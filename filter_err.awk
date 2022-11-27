#!/bin/awk -f
length($0)>80{s=" ("length($0)")";print substr($0,0,80-length(s))s;next}
{print $0}
