#!/bin/sh

for i in $(seq 4 1 12);
do
  python data_prep.py -E $i
done

