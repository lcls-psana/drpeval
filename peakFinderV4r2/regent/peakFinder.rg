import "regent"

local c = regentlib.c
local SlacConfig = require("slac_config")
local AlImgProc = require("AlImgProc")

-- local EVENTS = 1000
local SHOTS = 32
local HEIGHT = 185
local WIDTH = 388
local MAX_PEAKS = 200
local rank = 4
local r0 = 5
local dr = 0.05
local THR_LOW = 10
local THR_HIGH = 150
-- clustter test
local EVENTS = 100
local data_file = "/reg/d/psdm/cxi/cxitut13/scratch/cpo/test1000.bin"
-- local test
-- local EVENTS = 5
-- local data_file = "small_test"

terra read_float(f : &c.FILE, number : &float)
  return c.fread(number, sizeof(float), 1, f) == 1
end

task parallel_load_data(r_data : region(ispace(int3d), Pixel))
where reads writes (r_data)
do
  var sizeofFloat : int64 = 4
  var offset : int64 = [int64](r_data.bounds.lo.z) * [int64](WIDTH) * [int64](HEIGHT) * [int64](sizeofFloat)
  var f = c.fopen(data_file, "rb")
  c.fseek(f,offset,c.SEEK_SET)
  c.printf("Loading %s into panels %d through %d, offset:%ld\n", data_file, r_data.bounds.lo.z, r_data.bounds.hi.z, offset)
  var x : float[1]
  -- c.printf("r_data.bounds.lo.z, r_data.bounds.hi.z + 1 = %d, %d\n",r_data.bounds.lo.z, r_data.bounds.hi.z + 1)
  for i = r_data.bounds.lo.z, r_data.bounds.hi.z + 1 do
    for row = 0, HEIGHT do
      for col = 0, WIDTH do
        if not read_float(f, x) then
          c.printf("parallel_load_data: Couldn't read data in shot %d\n", i)
          return -1
        end
        r_data[{col,row,i}].cspad = x[0]
      end
    end
  end
  return 0
end

task writePeaks(r_peaks : region(ispace(int3d), Peak), color : int3d, parallelism : int)
where
  reads writes(r_peaks)
do
  var filename : int8[256]
  c.sprintf([&int8](filename), "peaks_%d/peaks_%d_%d", parallelism, r_peaks.bounds.lo.z,r_peaks.bounds.hi.z)
  c.printf("write to %s\n", filename)
  var f = c.fopen(filename,'w')
  -- c.printf("color:%d,%d,%d\n", color.x, color.y, color.z)
  var hdr = 'Evt Seg  Row  Col  Npix      Amax      Atot   rcent   ccent rsigma  csigma rmin rmax cmin cmax    bkgd     rms     son\n'
  for j = r_peaks.bounds.lo.z, r_peaks.bounds.hi.z+1 do
    var event = j / SHOTS
    var shot = j % SHOTS
    for i = 0, MAX_PEAKS do
      var peak : Peak = r_peaks[{0, i, j}]
      if peak.valid then 
        c.fprintf(f,"%3d %3d %4d %4d  %4d  %8.1f  %8.1f  %6.1f  %6.1f %6.2f  %6.2f %4d %4d %4d %4d  %6.2f  %6.2f  %6.2f\n", event,
          [int](peak.seg), [int](peak.row), [int](peak.col), [int](peak.npix), peak.amp_max, peak.amp_tot, peak.row_cgrav, peak.col_cgrav, peak.row_sigma, peak.col_sigma, 
          [int](peak.row_min), [int](peak.row_max), [int](peak.col_min), [int](peak.col_max), peak.bkgd, peak.noise, peak.son)
      end
    end
  end
  c.fclose(f)
end

terra wait_for(x : int)
  return x
end

task main()
  -- Configure --
  c.printf("start main task\n")
  
  var config : SlacConfig
  config:initialize_from_command()
  
  var p_colors = ispace(int3d, {1, 1, config.parallelism})
  var r_peaks = region(ispace(int3d, {1, MAX_PEAKS, EVENTS * SHOTS}), Peak)
  var p_peaks = partition(equal, r_peaks, p_colors)

	c.printf("Loading in %d batches\n", config.parallelism)

  var m_win : WinType
  m_win:init(0,0,WIDTH-1,HEIGHT-1)
	c.printf("Processing in %d batches\n", config.parallelism)

  var r_data = region(ispace(int3d, {WIDTH, HEIGHT, EVENTS * SHOTS}), Pixel)
  var p_data = partition(equal, r_data, p_colors)
  var is = ispace(int1d, EVENTS * SHOTS)

  -- load data
  var ts_start = c.legion_get_current_time_in_micros()
  c.printf("Start sending loading tasks at %.4f\n", (ts_start) * 1e-6)
  do
    var _ = 0
    for color in p_data.colors do
      _ += parallel_load_data(p_data[color])
    end
    wait_for(_)
  end

  -- cpu code
  do
    __demand(__spmd)
    for color in p_data.colors do
      if config.flood_only then
        AlImgProc.peakFinderV4r2_flood(p_data[color], p_peaks[color], rank, m_win, THR_HIGH, THR_LOW, r0, dr)
      else
        AlImgProc.peakFinderV4r2(p_data[color], p_peaks[color], rank, m_win, THR_HIGH, THR_LOW, r0, dr)
      end
    end
  end
  -- gpu code
  -- do
  --   var r_conmap = region(ispace(int3d, {WIDTH, HEIGHT, EVENTS * SHOTS}), uint32)
  --   var r_stop = region(ispace(int1d, EVENTS * SHOTS), bool)
  --   var r_acc_stop = region(ispace(int1d, EVENTS * SHOTS), bool)
  --   AlImgProc.floodFill_gpu(r_data, r_conmap, r_stop, r_acc_stop, rank, m_win, THR_HIGH, THR_LOW, r0, dr)
  -- end
  -- write peaks
  -- for color in p_data.colors do
  --   writePeaks(p_peaks[color], color, config.parallelism)
  -- end

  return 0
end

regentlib.start(main)


