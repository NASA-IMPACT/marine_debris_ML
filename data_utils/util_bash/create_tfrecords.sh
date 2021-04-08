#!bin/bash

# under color_enhanced folder they are all the label-maker created labels.npz and tiles
# run under data
# docker run -it -v ${PWD}:/mnt/data  covid_sat_ml_tf /bin/bash
# run "bash create_tfrecords.sh"

mkdir -p training_data_4cls_csvs
mkdir -p tfrecords_4cls
mkdir -p tf_records
# for d in color_enhanced/*; do
target_dir=./color_enhenced/*
#   echo $d
# for f in $target_dir/*/; do
for f in $(find $target_dir -maxdepth 1 -type d); do
  # if [[ -d "$f" && ! -L "$f"]]; then
  echo $f
  python3 utils_convert_tfrecords.py \
          --label_input=$f/labels.npz   \
          --data_dir=tf_records/$f   \
          --tiles_dir=$f/tiles    \
          --pbtxt=xview_hl_4cls.pbtxt
  mv *.csv training_data_4cls_csvs/
  mv *.records tfrecords_4cls
  # fi;
done
