#!/bin/bash
awk '/\s*(-?[0-9]\s+){3}/{print $8,$9,$10,$4}' amb-overpredict.stream  | ./cop agipd_2442_v2_gv_mg_41-45.geom | awk '{print $1,$2,$5}' > tmp
awk 'BEGIN{d=64}{r=int((($1/d)**2+($2/d)**2)**0.5);b[r]+=$3;++m[r];a[int($1/d)][int($2/d)]+=$3;++n[int($1/d)][int($2/d)]}END{for (x in n) for (y in n[x]) { r=int(sqrt(x**2+y**2));if (r in b) print x,y,a[x][y]*m[r]/(b[r]*n[x][y])}}' tmp > peaksum
# notice that intensity decreases along y but not x
