#!/bin/bash
gnuplot << EOF
set terminal postscript eps size 5, 3 enhanced color \
    font 'Helvetica,20' linewidth 2
unset colorbox
set xrange [-10000:60000]
set yrange [-10000:60000]
set xlabel "predicted intensity"
set ylabel "measured intensity"
set output "predicted_vs_measured_subset0.eps"
a=1.0/16;
p "<shuf -n 10000 subset0/premerge" u 7:5 lc rgb "black" pt 7 notitle
EOF
gnuplot << EOF
set terminal postscript eps size 5, 3 enhanced color \
    font 'Helvetica,20' linewidth 2
unset colorbox
set xrange [-10000:60000]
set yrange [-10000:60000]
set xlabel "predicted intensity"
set ylabel "measured intensity"
set output "predicted_vs_measured_subset1.eps"
a=1.0/16;
p "<shuf -n 10000 subset1/premerge" u 7:5 lc rgb "black" pt 7 notitle
EOF
