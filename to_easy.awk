#!/bin/awk -f
BEGIN{profile_radius=1e-6;n=0;m=0}
ARGIND==1{source=source$0"\n";next}
ARGIND==2{crystls[n]=$0"\n";n++;next}
/^profile_radius/{pr=$3;next}
/^photon_energy_eV\s*=\s*[0-9]+\.[0-9]+/{if($3>0)l=1239.84/$3;next}
/^astar/{a0=$3*1.0;a1=$4*1.0;a2=$5*1.0;next}
/^bstar/{b0=$3*1.0;b1=$4*1.0;b2=$5*1.0;next}
/^cstar/{c0=$3*1.0;c1=$4*1.0;c2=$5*1.0;next}
/^   h    k    l          I /{
  print source;
  if (m in crystls) {
    print crystls[m];
  } else {
    print "<";
    printf("%16g %16g %16g\n",a0,  b0,  c0);
    printf("%16g %16g %16g\n",a1,  b1,  c1);
    printf("%16g %16g %16g\n",a2,  b2,  c2);
    printf("%16g %16g %16g\n",pr,   0,   0);
    printf("%33g %16g\n"     ,     pr,   0);
    printf("%50g\n"          ,          pr);
    print 1e-4, 1e-9, 1, 0.001, 200, 1;
  }
  ++m;
  next
}
/^\s*(-?[0-9]+\s+){3}/{
  printf("%d %d %d %.2f %.2f %.0f %.0f\n",$1,$2,$3,$4,$5,$8,$9);
  next
}
