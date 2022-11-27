awk 'BEGIN{n=0}/^\s*(-?[0-9]+\s+){3}/{n=!n;if(n) print $0;next}{print $0}'
