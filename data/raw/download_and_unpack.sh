#!/bin/bash

wget http://snap.stanford.edu/decagon/bio-decagon-ppi.tar.gz
wget http://snap.stanford.edu/decagon/bio-decagon-targets.tar.gz

#wget http://snap.stanford.edu/decagon/bio-decagon-targets-all.tar.gz
#Note that including this file in analysis introduces ~1400 drug nodes and ~112k drug-target edges MORE than are reported in the paper. For this reason it will be excluded from all analyses.

wget http://snap.stanford.edu/decagon/bio-decagon-combo.tar.gz
wget http://snap.stanford.edu/decagon/bio-decagon-mono.tar.gz
#wget http://snap.stanford.edu/decagon/bio-decagon-effectcategories.tar.gz -- This is unusued in this analysis

for f in *.gz;
do
    tar -xzvf $f
    rm -f $f
done