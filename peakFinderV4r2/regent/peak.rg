import "regent"

local c = regentlib.c

struct Peak {
  seg : double;
  row : double;
  col : double;
  npix : double;
  npos : double;
  amp_max : double;
  amp_tot : double;
  row_cgrav : double;
  col_cgrav : double;
  row_sigma : double;
  col_sigma : double;
  row_min : double;
  row_max : double;
  col_min : double;
  col_max : double;
  bkgd : double;
  noise : double;
  son : double;
  valid  : bool;
}

return Peak