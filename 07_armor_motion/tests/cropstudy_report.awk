# Usage: awk -f tests/cropstudy_report.awk baseline.csv variant.csv
# Counts decision changes, NOT correctness, recall, or physical identities.
BEGIN { FS="," }
FNR==1 {
  if ($0 != "frame,index,crop_clipped,label_id,label,confidence,above_threshold,status") {
    print "invalid header" > "/dev/stderr"; failed=1; exit 1
  }
  next
}
NR==FNR {
  key=$1 SUBSEP $2
  if (key in base) { print "duplicate baseline" > "/dev/stderr"; failed=1; exit 1 }
  base[key]=($4==6 && $7==1 && $8=="classified"); label[key]=$4; expected++
  next
}
{
  key=$1 SUBSEP $2
  if (!(key in base) || key in seen) { print "unmatched/duplicate variant" > "/dev/stderr"; failed=1; exit 1 }
  seen[key]=1; count++; accepted=($4==6 && $7==1 && $8=="classified")
  added+=(!base[key] && accepted); removed+=(base[key] && !accepted)
  changed+=($4!=label[key]); total+=accepted
}
END {
  if (failed) exit 1
  if(count!=expected || count==0) { print "incomplete comparison" > "/dev/stderr"; exit 1 }
  printf "candidates=%d target_accepted=%d added=%d removed=%d top_class_changed=%d\n",count,total,added,removed,changed
}
