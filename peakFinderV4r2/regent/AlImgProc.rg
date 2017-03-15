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

return AlImgProc

