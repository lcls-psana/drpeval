import "regent"

-- Helper module to handle command line arguments
local SlacConfig = require("slac_config")
local Queue = require("queue")
local Peak = require("peak")
local PeakHelper = require("peakHelper")

local sqrt = regentlib.sqrt(double)

local c = regentlib.c

--------------------------------------------------------------------------------
--                               Configuration                                --
--------------------------------------------------------------------------------
local AlImgProc = {}

local WIDTH = 388
local HEIGHT = 185
local SHOTS = 32
local MAX_PEAKS = 200
local npix_min = 2
local npix_max = 50
local amax_thr = 10
local atot_thr = 20
local son_min = 5
local SON_MIN = 5
local BUFFER_SIZE = 400
local HEIGHT = 185
local WIDTH = 388

terra ternary(cond : bool, T : int, F : int)
  if cond then return T else return F end
end
--------------------------------------------------------------------------------
--                                   Structs                                  --
--------------------------------------------------------------------------------

fspace Pixel {
	cspad : float
}

fspace Mask {
  mask : int;
}

struct WinType{
  left : int;
  top : int;
  right : int;
  bot : int
}

terra WinType:init(left : int, top : int, right : int, bot : int)
  self.left = left
  self.right = right
  self.top = top
  self.bot = bot
end

--------------------------------------------------------------------------------
--                                 Processing                                 --
--------------------------------------------------------------------------------

terra in_ring(dx : int32, dy : int32, r0 : double, dr : double)
  var dist = dx * dx + dy * dy
  return dist >= r0 * r0 and dist <= (r0 + dr) * (r0 + dr)
end

terra peakIsPreSelected(peak : Peak)
  if peak.son < son_min then return false end
  if peak.npix < npix_min then return false end
  if peak.npix > npix_max then return false end
  if peak.amp_max < amax_thr then return false end
  if peak.amp_tot < atot_thr then return false end
  return true
end

task AlImgProc.peakFinderV4r2(data : region(ispace(int3d), Pixel), peaks : region(ispace(int3d), Peak),  rank : int, win : WinType, thr_high : double, thr_low : double, r0 : double, dr : double)
where
	reads (data), writes(peaks)
do
  var ts_start = c.legion_get_current_time_in_micros()
	var index : uint32 = 1
  var half_width : int = [int](r0 + dr)
  for i = data.bounds.lo.z, data.bounds.hi.z + 1 do
    for j = 0, MAX_PEAKS do
      peaks[{0,j,i}].valid = false
    end 
  end
  
  var num_peaks = 0  
  for p_i = data.bounds.lo.z, data.bounds.hi.z + 1 do
    var idx_x : int[BUFFER_SIZE]
    var idx_y : int[BUFFER_SIZE]
    var conmap : uint32[WIDTH*HEIGHT]
    for i = 0, WIDTH*HEIGHT do
      conmap[i] = 0
    end

    var queue : Queue
    queue:init()
    var shot_count = 0   
    for i = 0, WIDTH*HEIGHT do
      conmap[i] = 0
    end

    for row = win.top, win.bot + 1 do
			for col = win.left, win.right + 1 do
				if data[{col, row, p_i}].cspad > thr_high and conmap[row*WIDTH+col] <= 0 and shot_count <= MAX_PEAKS then
          index += 1
          var set = index
          var significant = true
          var average : double = 0.0
          var variance : double = 0.0
          var count = 0
          -- check significance
          var r_min = ternary(row - half_width < win.top, win.top - row, -half_width)
          var r_max = ternary(row + half_width > win.bot, win.bot - row, half_width )
          var c_min = ternary(col - half_width < win.left, win.left - col, -half_width )
          var c_max = ternary(col + half_width > win.right, win.right - col, half_width )

          for r = r_min, r_max + 1 do
            for c = c_min, c_max + 1 do
              if in_ring(c,r,r0,dr) and data[{c + col, r + row, p_i}].cspad < thr_low then
                var cspad : double = data[{c + col, r + row, p_i}].cspad
                average += cspad
                variance += cspad * cspad
                count += 1
              end
            end
          end
          var stddev : double = 0.0
          if count > 0 then
            average /= [double](count)
            variance = variance / [double](count) - average * average
            stddev = sqrt(variance)
          end
					
          if significant then
            -- clear buffer
            for i = 0, (2*rank+1)*(2*rank+1) do
              idx_x[i] = -1
              idx_y[i] = -1
            end

            r_min = ternary(win.top < row - rank, row - rank, win.top)
            r_max = ternary(win.bot > row + rank, row + rank, win.bot)
            c_min = ternary(win.left < col - rank, col - rank, win.left)
            c_max = ternary(win.right > col + rank, col + rank, win.right)

            var pix_counter = 0
            var peak_helper : PeakHelper
            

            queue:clear()
            queue:enqueue({col, row, p_i})
            conmap[row*WIDTH+col] = set
            idx_x[pix_counter] = col
            idx_y[pix_counter] = row
            pix_counter += 1
            var init_cspad = data[{col, row, p_i}].cspad
            var is_local_maximum = true
            while not queue:empty() do
              if not is_local_maximum then break end
              var p = queue:dequeue()
              if p.x - 1 >= c_min then
                var t : int3d = {p.x - 1, p.y, p.z}
                var cspad = data[t].cspad
                if cspad > thr_low and conmap[t.y*WIDTH+t.x] == 0 then
                  is_local_maximum = cspad <= init_cspad
                  if not is_local_maximum then break end
                  queue:enqueue(t)
                  conmap[t.y*WIDTH+t.x] = set
                  idx_x[pix_counter] = t.x
                  idx_y[pix_counter] = t.y
                  pix_counter += 1
                end
              end
              if p.x + 1 <= c_max then
                var t : int3d  = {p.x + 1, p.y, p.z}
                var cspad = data[t].cspad
                if cspad > thr_low and conmap[t.y*WIDTH+t.x] == 0 then
                  is_local_maximum = cspad <= init_cspad
                  if not is_local_maximum then break end
                  queue:enqueue(t)
                  conmap[t.y*WIDTH+t.x] = set
                  idx_x[pix_counter] = t.x
                  idx_y[pix_counter] = t.y
                  pix_counter += 1
                end
              end
              if p.y - 1 >= r_min then
                var t : int3d  = {p.x, p.y - 1, p.z}
                var cspad = data[t].cspad
                if cspad > thr_low and conmap[t.y*WIDTH+t.x] == 0 then
                  is_local_maximum = cspad <= init_cspad
                  if not is_local_maximum then break end
                  queue:enqueue(t)
                  conmap[t.y*WIDTH+t.x] = set
                  idx_x[pix_counter] = t.x
                  idx_y[pix_counter] = t.y
                  pix_counter += 1
                end
              end
              if p.y + 1 <= r_max then
                var t : int3d  = {p.x, p.y + 1, p.z}
                var cspad = data[t].cspad
                if cspad > thr_low and conmap[t.y*WIDTH+t.x] == 0 then
                  is_local_maximum = cspad <= init_cspad
                  if not is_local_maximum then break end
                  queue:enqueue(t)
                  conmap[t.y*WIDTH+t.x] = set
                  idx_x[pix_counter] = t.x
                  idx_y[pix_counter] = t.y
                  pix_counter += 1
                end
              end
            end
            var add_point_success = false
            if is_local_maximum then

              peak_helper:init(row,col,data[{col,row,p_i}].cspad,average,stddev, p_i%SHOTS, WIDTH,HEIGHT)
              for i = 0, pix_counter do
                var x = idx_x[i]
                var y = idx_y[i]
                peak_helper:add_point(data[{x,y,p_i}].cspad, y, x)
              end
              var peak : Peak = peak_helper:get_peak()
              if peakIsPreSelected(peak) then
                peak.valid = true
                peaks[{0, shot_count, p_i}] = peak
                add_point_success = true
                shot_count += 1
                num_peaks += 1
              end
            else
              for i = 0, pix_counter do
                conmap[idx_y[i]*WIDTH+idx_x[i]] = 0
              end
            end
          end
				end
			end
		end
	end
	var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("peakFinderTask: (%d - %d) starts from %.4f, ends at %.4f, num_peaks:%d\n", data.bounds.lo.z, data.bounds.hi.z + 1, (ts_start) * 1e-6, (ts_stop) * 1e-6, num_peaks)
	return 0
end

local write_flood_fill = false
task AlImgProc.peakFinderV4r2_flood(data : region(ispace(int3d), Pixel), peaks : region(ispace(int3d), Peak),  rank : int, win : WinType, thr_high : double, thr_low : double, r0 : double, dr : double)
where
  reads (data), writes(peaks)
do
  var ts_start = c.legion_get_current_time_in_micros()
  var index : uint32 = 1
  var half_width : int = [int](r0 + dr)
  -- c.printf("p_data.bounds.lo.x:%d, p_data.bounds.hi.x:%d\n",peaks.bounds.lo.x, peaks.bounds.hi.x)
  -- var HEIGHT = data.bounds.hi.y + 1
  -- var WIDTH = data.bounds.hi.x + 1
  var idx_x : int[BUFFER_SIZE]
  var idx_y : int[BUFFER_SIZE]
  var num_peaks = 0
  var conmap : int[WIDTH*HEIGHT]

  var f : &c.FILE
  if write_flood_fill then
    f = c.fopen("peaks.regent.img","w")
  end

  -- c.printf("height:%d,width:%d\n",HEIGHT,WIDTH)
  -- var r_conmap = region(ispace(int2d, {WIDTH,HEIGHT}), uint32)
  -- for p_i in is do
  for i = data.bounds.lo.z, data.bounds.hi.z + 1 do
    for j = 0, MAX_PEAKS do
      peaks[{0,j,i}].valid = false
    end 
  end
  
  for p_i = data.bounds.lo.z, data.bounds.hi.z + 1 do
    var queue : Queue
    queue:init()
    var shot_count = 0   
    for i = 0, WIDTH*HEIGHT do
      conmap[i] = 0
    end
    var set = 0
    -- flood fill
    for row = win.top, win.bot + 1 do
      for col = win.left, win.right + 1 do
        if data[{col, row, p_i}].cspad > thr_high and conmap[row*WIDTH+col] == 0 then
          set += 1
          conmap[row * WIDTH + col] = set
          queue:clear()
          queue:enqueue({col, row, p_i})
          while not queue:empty() do
            var p = queue:dequeue()
            if p.x - 1 >= win.left then
              var t : int3d = {p.x - 1, p.y, p.z}
              if conmap[t.y*WIDTH+t.x] == 0 and data[t].cspad > thr_low then
                queue:enqueue(t)
                conmap[t.y*WIDTH+t.x] = set
              end
            end
            if p.x + 1 <= win.right then
              var t : int3d = {p.x + 1, p.y, p.z}
              if conmap[t.y*WIDTH+t.x] == 0 and data[t].cspad > thr_low  then
                queue:enqueue(t)
                conmap[t.y*WIDTH+t.x] = set
              end
            end
            if p.y - 1 >= win.top then
              var t : int3d = {p.x, p.y - 1, p.z}
              if conmap[t.y*WIDTH+t.x] == 0 and data[t].cspad > thr_low  then
                queue:enqueue(t)
                conmap[t.y*WIDTH+t.x] = set
              end
            end
            if p.y + 1 <= win.bot then
              var t : int3d = {p.x, p.y + 1, p.z}
              if conmap[t.y*WIDTH+t.x] == 0 and data[t].cspad > thr_low then
                queue:enqueue(t)
                conmap[t.y*WIDTH+t.x] = set
              end
            end
          end
        end
      end
    end
    if write_flood_fill then
      var buf = [&float](c.malloc(WIDTH * HEIGHT * 4))
      for i = 0, HEIGHT*WIDTH do
        if conmap[i] > 0 then
          buf[i] = 1
        else
          buf[i] = 0
        end
      end
      c.fwrite(buf, 4, WIDTH*HEIGHT, f)
      c.free(buf)
    end
  end
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("peakFinderTask: (%d - %d) starts from %.6f, ends at %.6f\n", data.bounds.lo.z, data.bounds.hi.z + 1, (ts_start) * 1e-6, (ts_stop) * 1e-6)
  return 0
end

__demand(__cuda)
task AlImgProc.floodFill_gpu(data : region(ispace(int3d), Pixel), conmap : region(ispace(int3d), uint32), stop : region(ispace(int1d), bool), acc_stop : region(ispace(int1d), bool), rank : int, win : WinType, thr_high : double, thr_low : double, r0 : double, dr : double)
where
  reads(data, conmap, stop, acc_stop), writes(conmap, stop, acc_stop)
do
  var is3d = ispace(int3d, {388, 185, 32 * 5})
  var is1d = ispace(int1d, 32 * 5)
  var ts_start = c.legion_get_current_time_in_micros();
  -- filted by thr_high
  for is in is3d do
    if data[is].cspad > thr_high then
      conmap[is] = is.z * (WIDTH * HEIGHT) + is.y * (WIDTH) + is.x
    else
      conmap[is] = 0
    end
  end 
  -- flood fill
  for is1 in is1d do
    acc_stop[is1] = false
  end
  for iter = 0, 2*rank do
    for is in is1d do
      stop[is] = acc_stop[is]
      acc_stop[is] = true
    end
    for is in is3d do
      if not stop[is.z] then
        var intensity = data[is].cspad
        if intensity > thr_low then
          var changed = false
          var status = conmap[is]
          if is.x > 0 then
            var tgt_status = conmap[{is.x - 1, is.y, is.z}]
            if tgt_status > 0 and (status == 0 or data[{tgt_status % WIDTH, tgt_status / WIDTH % HEIGHT, tgt_status / WIDTH / HEIGHT}].cspad > intensity) and is.x - tgt_status % WIDTH <= rank then
              conmap[is] = tgt_status
              changed = true
            end
          end
          if is.x < WIDTH - 1 then
            var tgt_status = conmap[{is.x + 1, is.y, is.z}]
            if tgt_status > 0 and (status == 0 or data[{tgt_status % WIDTH, tgt_status / WIDTH % HEIGHT, tgt_status / WIDTH / HEIGHT}].cspad > intensity) and tgt_status % WIDTH - is.x <= rank then
              conmap[is] = tgt_status
              changed = true
            end
          end
          if is.y > 0 then
            var tgt_status = conmap[{is.x, is.y - 1, is.z}]
            if tgt_status > 0 and (status == 0 or data[{tgt_status % WIDTH, tgt_status / WIDTH % HEIGHT, tgt_status / WIDTH / HEIGHT}].cspad > intensity) and is.y - tgt_status / WIDTH % HEIGHT <= rank then
              conmap[is] = tgt_status
              changed = true
            end
          end
          if is.y < HEIGHT - 1 then
            var tgt_status = conmap[{is.x, is.y + 1, is.z}]
            if tgt_status > 0 and (status == 0 or data[{tgt_status % WIDTH, tgt_status / WIDTH % HEIGHT, tgt_status / WIDTH / HEIGHT}].cspad > intensity) and tgt_status / WIDTH % HEIGHT - is.y <= rank then
              conmap[is] = tgt_status
              changed = true
            end
          end
          if changed then
            acc_stop[is.z] = false
          end
        end
      end
    end
  end
  var ts_stop = c.legion_get_current_time_in_micros();
  c.printf("peakFinderTask: %f miliseconds\n", (ts_stop - ts_start) * 1e-3)
end

task AlImgProc.peakFinder_gpuCompare(d_data : region(ispace(int3d), Pixel), peaks : region(ispace(int3d), Peak),  rank : int, win : WinType, thr_high : double, thr_low : double, r0 : double, dr : double)
where
  reads (d_data), writes(peaks)
do
  var ts_start = c.legion_get_current_time_in_micros() 
  -- filterByThrHigh_v2
  var FILTER_PATCH_WIDTH = 32;
  var FILTER_PATCH_HEIGHT = 4; 
  var FILTER_THREADS_PER_PATCH = FILTER_PATCH_WIDTH * FILTER_PATCH_HEIGHT;
  var FILTER_PATCH_ON_WIDTH = (WIDTH) / FILTER_PATCH_WIDTH;
  var FILTER_PATCH_ON_HEIGHT = (HEIGHT + FILTER_PATCH_HEIGHT - 1) / FILTER_PATCH_HEIGHT;
  var FILTER_PATCH_PER_IMAGE = FILTER_PATCH_ON_WIDTH * FILTER_PATCH_ON_HEIGHT;
  var NUM_NMS_AREA = FILTER_PATCH_WIDTH / FILTER_PATCH_HEIGHT;
  var CENTER_SIZE = (d_data.bounds.hi.z+1 - d_data.bounds.lo.z) * FILTER_PATCH_PER_IMAGE * NUM_NMS_AREA
  var d_centers = [&uint32](c.malloc(CENTER_SIZE * 4))
  var center_idx = 0
  var npix = 0
  var num_peaks = 0
  for imgId = d_data.bounds.lo.z, d_data.bounds.hi.z+1 do
    for patch_y = 0, FILTER_PATCH_ON_HEIGHT do
      for patch_x = 0, FILTER_PATCH_ON_WIDTH do
        var data : float[32][4]
        for y = 0, FILTER_PATCH_HEIGHT do
          for x = 0, FILTER_PATCH_WIDTH do
            var col = patch_x * FILTER_PATCH_WIDTH + x
            var row = patch_y * FILTER_PATCH_HEIGHT + y
            data[y][x] = d_data[{col, row, imgId}].cspad
          end
        end
        for area = 0, NUM_NMS_AREA do
          var max_v : float = 0
          var max_id : uint32 = 0
          for y = 0, FILTER_PATCH_HEIGHT do
            for x = area * FILTER_PATCH_HEIGHT, area * FILTER_PATCH_HEIGHT + FILTER_PATCH_HEIGHT do
              if (data[y][x] > max_v) then
                var col = patch_x * FILTER_PATCH_WIDTH + x
                var row = patch_y * FILTER_PATCH_HEIGHT + y
                max_v = data[y][x]
                max_id = imgId * (WIDTH * HEIGHT) + row * WIDTH + col
              end
            end
          end
          if max_v > thr_high then
            d_centers[center_idx] = max_id
            npix += 1
          else
            d_centers[center_idx] = 0
          end
          center_idx += 1
        end
      end
    end
  end
  -- stream compaction
  var d_dense_centers = [&uint32](c.malloc(npix * 4))
  center_idx = 0
  for i = 0, CENTER_SIZE do
    if d_centers[i] > 0 then
      d_dense_centers[center_idx] = d_centers[i]
      center_idx += 1
    end
  end
  c.free(d_centers)
  d_centers = d_dense_centers

  -- floodFill_v2
  var HALF_WIDTH = 5;
  var PATCH_WIDTH = (2 * HALF_WIDTH + 1);
  var FF_LOAD_THREADS_PER_CENTER = 64;
  var FF_THREADS_PER_CENTER = 32;
  var FF_INFO_THREADS_PER_CENTER = FF_THREADS_PER_CENTER;
  var FF_THREADS_PER_BLOCK = 64;
  var FF_LOAD_PASS = (2 * HALF_WIDTH + 1) * (2 * HALF_WIDTH + 1) / FF_LOAD_THREADS_PER_CENTER + 1;
  var FF_CENTERS_PER_BLOCK = FF_THREADS_PER_BLOCK / FF_LOAD_THREADS_PER_CENTER;
  for blockIdx = 0, npix do
    -- load data
    var shot_count = 0
    var center_id = d_centers[blockIdx];
    var img_id = center_id / (WIDTH * HEIGHT);
    var crow = center_id / WIDTH % HEIGHT;
    var ccol = center_id % WIDTH;
    var data : float[11][11];
    var status : uint[11][11];
    for irow = 0, PATCH_WIDTH do
      for icol = 0, PATCH_WIDTH do
        var drow = crow + irow - HALF_WIDTH
        var dcol = ccol + icol - HALF_WIDTH
        if (drow >= 0 and drow < HEIGHT and dcol >= 0 and dcol < WIDTH) then
          data[irow][icol] = d_data[{dcol, drow, img_id}].cspad;
        elseif irow < PATCH_WIDTH then
          data[irow][icol] = 0;
        end
        status[irow][icol] = 0
      end
    end
    status[HALF_WIDTH][HALF_WIDTH] = center_id
    -- flood fill
    var FF_SCAN_LENGTH = FF_THREADS_PER_CENTER / 8;
    var sign_x : int[8]
    var sign_y : int[8]
    sign_x[0] = -1; sign_x[1] = 1; sign_x[2] = 1; sign_x[3] = -1; sign_x[4] = 1; sign_x[5] = 1; sign_x[6] = -1; sign_x[7] = -1;
    sign_y[0] = 1; sign_y[1] = 1; sign_y[2] = -1; sign_y[3] = -1; sign_y[4] = 1; sign_y[5] = -1; sign_y[6] = -1; sign_y[7] = 1; 
    var icol : int[32]
    var irow : int[32]
    var base_v : int[32]
    var dxs : int[4];
    var dys : int[4];
    dxs[0] = -1; dxs[1] = 1; dxs[2] = 0; dxs[3] = 0; 
    dys[0] = 0; dys[1] = 0; dys[2] = 1; dys[3] = -1; 
    var dx : int[32]
    var dy : int[32]
    for threadIdx = 0, 32 do
      var scanline_id = threadIdx / FF_SCAN_LENGTH
      var id_in_grp = threadIdx % (2 * FF_SCAN_LENGTH);
      base_v[threadIdx] = id_in_grp - FF_SCAN_LENGTH;
      icol[threadIdx] = base_v[threadIdx] * sign_x[scanline_id] + HALF_WIDTH;
      irow[threadIdx] = base_v[threadIdx] * sign_y[scanline_id] + HALF_WIDTH;
      var scangrp_id = threadIdx / (2 * FF_SCAN_LENGTH);
      dx[threadIdx] = dxs[scangrp_id];
      dy[threadIdx] = dys[scangrp_id];
    end
    var center_intensity = data[HALF_WIDTH][HALF_WIDTH];
    var is_local_maximum = true
    for i = 1, rank+1 do
      if not is_local_maximum then break end
      for threadIdx = 0, 32 do
        icol[threadIdx] += dx[threadIdx];
        irow[threadIdx] += dy[threadIdx];
        if (data[irow[threadIdx]][icol[threadIdx]] > center_intensity) then
          is_local_maximum = false;
        end
        if (data[irow[threadIdx]][icol[threadIdx]] > thr_low) then
          if (status[irow[threadIdx]-dy[threadIdx]][icol[threadIdx]-dx[threadIdx]] == center_id) then
            status[irow[threadIdx]][icol[threadIdx]] = center_id;
          end
        end
      end
    end
    for i = 1, FF_SCAN_LENGTH do
      if not is_local_maximum then break end
      for threadIdx = 0, 32 do
        var bound = base_v[threadIdx]
        if bound < 0 then bound = -bound end
        if i <= bound then
          icol[threadIdx] += dx[threadIdx];
          irow[threadIdx] += dy[threadIdx];
          if (data[irow[threadIdx]][icol[threadIdx]] > center_intensity) then
            is_local_maximum = false;
          end
          if (data[irow[threadIdx]][icol[threadIdx]] > thr_low) then
            if (status[irow[threadIdx]-dy[threadIdx]][icol[threadIdx]-dx[threadIdx]] == center_id) then
              status[irow[threadIdx]][icol[threadIdx]] = center_id;
            end
          end
        end
      end
    end
    if is_local_maximum then
      var r_min = ternary(crow - HALF_WIDTH < win.top, win.top - crow, -HALF_WIDTH)
      var r_max = ternary(crow + HALF_WIDTH > win.bot, win.bot - crow, HALF_WIDTH )
      var c_min = ternary(ccol - HALF_WIDTH < win.left, win.left - ccol, -HALF_WIDTH )
      var c_max = ternary(ccol + HALF_WIDTH > win.right, win.right - ccol, HALF_WIDTH )
      var average : float = 0
      var variance : float = 0
      var count : int = 0
      for r = r_min, r_max + 1 do
        for c = c_min, c_max + 1 do
          if in_ring(c,r,r0,dr) and d_data[{c + ccol, r + crow, img_id}].cspad < thr_low then
            var cspad : double = d_data[{c + ccol, r + crow, img_id}].cspad
            average += cspad
            variance += cspad * cspad
            count += 1
          end
        end
      end
      var stddev : double = 0.0
      if count > 0 then
        average /= [double](count)
        variance = variance / [double](count) - average * average
        stddev = sqrt(variance)
      end
      var peak_helper : PeakHelper
      peak_helper:init(crow,ccol,d_data[{ccol, crow, img_id}].cspad,average,stddev, img_id%SHOTS, WIDTH,HEIGHT)
      for r = 0, PATCH_WIDTH do
        for c = 0, PATCH_WIDTH do
          if status[r][c] == center_id then
            var drow = crow + r - HALF_WIDTH
            var dcol = ccol + c - HALF_WIDTH
            if (drow >= 0 and drow < HEIGHT and dcol >= 0 and dcol < WIDTH) then
              peak_helper:add_point(d_data[{dcol, drow, img_id}].cspad, drow, dcol)
            end
          end
        end
      end
      var peak : Peak = peak_helper:get_peak()
      if peakIsPreSelected(peak) then
        peak.valid = true
        peaks[{0, shot_count, img_id}] = peak
        shot_count += 1
        num_peaks += 1
      end
    end
  end
  var ts_stop = c.legion_get_current_time_in_micros()
  c.printf("peakFinder_gpuCompare: (%d - %d) starts from %.6f, ends at %.6f, num_peaks:%d\n", d_data.bounds.lo.z, d_data.bounds.hi.z + 1, (ts_start) * 1e-6, (ts_stop) * 1e-6, num_peaks)
end

return AlImgProc

