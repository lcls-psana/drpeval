import "regent"

local c = regentlib.c
local Peak = require("peak")
local sqrt = regentlib.sqrt(double)


struct PeakHelper {
	seg  : int;
    npix : int;
    samp : double;
    sac1 : double;
    sac2 : double;
    sar1 : double;
    sar2 : double;
    rmin : int;
    cmin : int;
    rmax : int;
    cmax : int;
    r0 : int;
    c0 : int;
    a0 : double;
    bg_avg : double;
    bg_rms : double;
    WIDTH : int;
    HEIGHT : int;
}

terra PeakHelper:init(r0 : int, c0 :int, a0 : double, bg_avg : double, bg_rms : double, seg : double, WIDTH : int, HEIGHT : int)
	self.r0 = r0
	self.c0 = c0
	self.a0 = a0
	self.bg_avg = bg_avg
	self.bg_rms = bg_rms
	self.seg = seg
	self.WIDTH = WIDTH
	self.HEIGHT = HEIGHT
	self.npix = 0
	self.samp = 0
	self.sac1 = 0
	self.sac2 = 0
	self.sar1 = 0
	self.sar2 = 0
	self.rmin = WIDTH
	self.cmin = HEIGHT
	self.rmax = 0
	self.cmax = 0
end

terra PeakHelper:add_point(intensity : double, row : int, col : int)
	var a : double = intensity
	var r : int = row
	var c : int = col
	if r < self.rmin then self.rmin = r end
	if r > self.rmax then self.rmax = r end
	if c < self.cmin then self.cmin = c end
	if c > self.cmax then self.cmax = c end

	self.npix = self.npix + 1
	self.samp = self.samp + a
	self.sar1 = self.sar1 + a * r
	self.sac1 = self.sac1 + a * c
	self.sar2 = self.sar2 + a * r * r
	self.sac2 = self.sac2 + a * c * c
end

terra ternary(cond : bool, T : double, F : double)
  if cond then return T else return F end
end

terra PeakHelper:get_peak()
	var peak : Peak
	peak.seg = self.seg
	peak.row = self.r0
	peak.col = self.c0
	peak.npix = self.npix
	peak.amp_max = self.a0 - self.bg_avg
	peak.amp_tot = self.samp - self.bg_avg * self.npix

	if self.samp > 0 then
		self.sar1 = self.sar1 / self.samp
		self.sac1 = self.sac1 / self.samp
		self.sar2 = self.sar2 / self.samp - self.sar1 * self.sar1
		self.sac2 = self.sac2 / self.samp - self.sac1 * self.sac1
		peak.row_cgrav = self.sar1
		peak.col_cgrav = self.sac1
		peak.row_sigma = ternary(self.npix > 1 and self.sar2 > 0, sqrt(self.sar2), 0)
		peak.col_sigma = ternary(self.npix > 1 and self.sac2 > 0, sqrt(self.sac2), 0)
	else
		peak.row_cgrav = self.r0
		peak.col_cgrav = self.c0
		peak.row_sigma = 0
		peak.col_sigma = 0
	end

	peak.row_min = self.rmin
	peak.row_max = self.rmax
	peak.col_min = self.cmin
	peak.col_max = self.cmax
	peak.bkgd = self.bg_avg
	peak.noise = self.bg_rms
	var noise_tot : double = self.bg_rms * sqrt(self.npix)
	peak.son = ternary(noise_tot > 0, peak.amp_tot / noise_tot, 0)

	return peak
end

return PeakHelper