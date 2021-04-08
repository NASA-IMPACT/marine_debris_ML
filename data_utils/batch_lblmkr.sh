#!/usr/bin/env bash

for d in */; do
    # Will print */ if no directories are available
    echo "$d"
    cd $d
    label-maker labels
    label-maker preview -n 10
    label-maker images
    cd ../
done

