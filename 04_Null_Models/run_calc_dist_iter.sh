#!/bin/bash

for IDX in {0..11845}

do
  echo $IDX
  python calc_dist_iter.py -idx $IDX
done
