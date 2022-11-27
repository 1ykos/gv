#!/bin/awk -f
BEGIN{sumw=mean=var=0}
/^\s*([+-]?[0-9]+\s+){3}/{
  w = 1;
  x = log(1+$4**2);
  sumw += w;
  dx    = x-mean;
  mean += dx*w/sumw;
  var  += ((x-mean)*dx-var)*w/sumw;
}
END { print exp(0.5*mean)-1; }
