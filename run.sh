nohup ./to_easy.awk source <(awk '{printf($0" ")}NR%7==0{print ""}' crystls_latest ) amb-overpredict.stream | ./mergeparm2bin | ./merge 197 > log 2> err &
