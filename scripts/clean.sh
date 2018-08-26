#! /bin/bash

cd ..
find intel -name "*.csv" -exec rm -f {} \;
find intel -name "*.png" -exec rm -f {} \;
find models -name "*.csv" -exec rm -f {} \;
