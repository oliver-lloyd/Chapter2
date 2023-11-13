#!/bin/bash

cd ../../../../kge/local/experiments/chapter2_experiments
for dir in *; 
do 
    mv_dir=../../../../Chapter2/analysis/assessment/all_trials_assess/temp/$dir;
    mkdir $mv_dir;
    cd $dir; 
    for subdir in 0*; 
    do 
        mkdir ../$mv_dir/$subdir; 
        cp $subdir/checkpoint_best.pt ../$mv_dir/$subdir; 
    done; 
    cd -; 
done