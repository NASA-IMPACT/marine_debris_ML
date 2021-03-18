#!/usr/bin/env bash

files=./*.tif

for f in $files; do
rio color -d uint8 -j 2 $f $f gamma R 1.5 gamma G 1.5 gamma B 1.5 sigmoidal RGB 9 0.35 saturation 1.0
  echo " $f color enhenced!"
done
