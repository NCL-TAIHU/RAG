#!/bin/bash

for year in {108..112}; do
    echo "Converting year $year..."
    python data/action/xlsx_to_jsonl.py \
        ~/data/ncl/${year}.xlsx \
        /home/share/data/ncl_json/${year}.jsonl
done
