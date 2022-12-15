#!/bin/bash
mkdir -p data
pushd data
wget https://snap.stanford.edu/data/soc-sign-bitcoinotc.csv.gz
gzip -d soc-sign-bitcoinotc.csv.gz
popd
make && ./bin/main
