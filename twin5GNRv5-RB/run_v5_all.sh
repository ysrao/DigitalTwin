#!/usr/bin/env bash
# Run every v5 grid: 4 sites x 5 bands x {street, roof}. Resumable: chunks
# already recorded in a grid's .npz are skipped, finished grids are skipped.
# Order: one site at a time (all bands), so each site becomes complete in turn.
cd "$(dirname "$0")"
for s in NW NE SW SE; do
  for f in 3.6 0.7 2.5 7.125 28; do
    for rx in street roof; do
      fstr="f$(python3 -c "print(int(round($f*1000)))")"
      out="data_v5_$(echo $s | tr A-Z a-z)_${fstr}_${rx}.npz"
      if [ -f "$out" ] && python3 -c "import numpy as np,sys; d=np.load('$out'); sys.exit(0 if 'rsrp_tilt' in d else 1)"; then
        continue
      fi
      for k in 0 1 2 3 4 5 6 7 8; do
        if [ -f "$out" ] && python3 -c "import numpy as np,sys; sys.exit(0 if $k in np.load('$out')['done'] else 1)"; then
          continue
        fi
        python3 rt_v5.py --site Site-$s --freq-ghz $f --rx $rx --chunk $k 2>&1 | grep -v -i warn
      done
      python3 rt_v5.py --site Site-$s --freq-ghz $f --rx $rx --post
    done
  done
done
echo ALL_DONE
