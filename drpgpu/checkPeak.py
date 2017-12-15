import cPickle as pickle
import math

"""
for i in range(20):
  p = pickle.load(open('dials/r0106/099/out/idx-'+str(i+1).zfill(4)+'_indexed.pickle', 'r'))
  print i+1, p.nrows()
"""
i = 5
peakFilename = '/reg/d/psdm/cxi/cxid9114/scratch/mona/l2/discovery/dials/r0106/099/out/idx-'+str(i+1).zfill(4)+'_indexed.pickle'
p = pickle.load(open(peakFilename, 'r'))

print '%12s %12s'%('cctbx','psana')
print '%5s %4s %4s %4s %4s %4s %10s'%('Panel','Fast','Slow','Seg','Row','Col','Iobs')
for i in range(p.nrows()):
  # cctbx format: panel (64), fast (194), slow (185)
  panelId = p['panel'][i]
  Iobs = p['intensity.sum.value'][i]
  fast, slow, _ = p['xyzobs.px.value'][i]

  #convert to sector (32), row (185), and column (388)
  row = slow
  col = fast + (194 * (panelId % 2))
  sector = (panelId - (panelId % 2)) // 2

  print '%5d %4d %4d %4d %4d %4d %10.2f'%(panelId, fast, slow, sector, row, col, Iobs)


"""
with open('zhenglin_evt01.txt','r') as f:
  data = f.read().splitlines()

for row in data:
  cols = row.split(',')
  dcols = [float(col.strip()) for col in cols]
  (_, _, seg, r, c, npix, amax, atot) = tuple(dcols)
  cctbxPanel = int(math.floor(seg * 2 + c/194))
  cctbxFast = c - (194 * (c//194))
  cctbxSlow = r
  print cctbxPanel, cctbxFast, cctbxSlow, atot

from IPython import embed
embed()
"""
