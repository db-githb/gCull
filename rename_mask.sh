#!/bin/bash

ls mask_*.png | while read f; do
  num="${f#mask_}"
  num="${num%.png}"

  if [[ ${#num} -eq 5 && ${num:0:1} == "0" ]]; then
    new_num="${num:1}"  # remove the first digit only
    mv "$f" "mask_${new_num}.png"
  fi
done
