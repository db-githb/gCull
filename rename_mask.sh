#!/bin/bash

ls mask_*.png \
  | sort -r \
  | while read f; do
      # extract the 4-digit number and bump it
      num="${f#mask_}"; num="${num%.png}"
      new=$(printf "%04d" $((10#$num + 1)))
      mv "$f" "mask_${new}.png"
    done
