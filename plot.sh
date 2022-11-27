#!/bin/bash
#awk '{x=$7/256;x=x>=0?int(x):x==int(x)?x:int(x)-1;y=$5/256;y=y>=0?int(y):y==int(y)?y:int(y)-1;++a[x][y]}END{for (x in a) for (y in a[x]) print x,y,a[x][y]}' premerge > tmp
#awk '{a[$1][$2]+=$3}END{for (x=-32;x!=512-32;++x) {for (y=-32;y!=512-32;++y) printf("%d ",1*a[x][y]);print ""} }' tmp > tmp2
gnuplot << EOF
set terminal postscript eps size 5, 3 enhanced color \
    font 'Helvetica,20' linewidth 2
unset colorbox
set xlabel "predicted partiality"
set ylabel "measured partiality"
set palette defined ( 0 "red", 1 "blue")
set output "partiality_predicted_vs_measured.eps"
a=1.0/16;
set yrange [-0.5:1.5]
p "<head -n 80000 premerge" u (\$7>0&&\$11>0&&sqrt(\$6)/\$11<0.25?\$7/\$11:1/0):(\$5/\$11):(sqrt(\$6)/\$11) lc rgb "black" pt 7 ps 1 notitle w errorbars, x lc rgb "black" lw 2 notitle
EOF
gnuplot << EOF
set terminal postscript eps size 5, 3 enhanced color \
    font 'Helvetica,20' linewidth 2
unset colorbox
set xlabel "predicted intensity"
set ylabel "measured intensity"
set palette defined ( 0 "red", 1 "blue")
set output "predicted_vs_measured_errorbars.eps"
a=1.0/16;
p "<head -n 6000 premerge" u 7:5:(sqrt(\$6)):((1-a)*\$9/((1-a)*\$9+a*\$10)) w errorbars lc palette pt 7 ps 1 notitle, x lc rgb "black" lw 2 notitle
EOF
gnuplot << EOF
set terminal postscript eps size 5, 3 enhanced color \
    font 'Helvetica,20' linewidth 2
unset colorbox
set xlabel "predicted partiality"
set ylabel "measured partiality"
set output "partiality_predicted_vs_measured_scatterplot_8_I_over_sigma.eps"
a=1.0/16;
set yrange [-0.5:1.5]
set xrange [0:1]
p "<head -n 500000 premerge" u (\$7>0&&\$11>0&&sqrt(\$6)/\$11<0.25?\$7/\$11:1/0):(\$5/\$11) lc rgb "black" pt 7 ps 0.2 notitle
EOF
