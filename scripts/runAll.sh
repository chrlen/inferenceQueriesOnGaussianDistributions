#! /bin/bash

cd ..
find intel -name "*.csv" -exec rm -f {} \;
python generateSparseModels.py


cd scripts



sh timeOnly.sh
#sh plotOnly.sh







