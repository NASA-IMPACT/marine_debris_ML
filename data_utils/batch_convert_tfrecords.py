import os, sys
from os import makedirs, path as op
import shutil
from glob import glob


tif_tiles = glob(f'/path/to/planet/scenes/'+'/**/*.tif',recursive=True)

labels_dir = f'/path/to/labelmaker/data/'

for tif in tif_tiles:
    filename_split = os.path.splitext(tif) 
    filename_zero, fileext = filename_split 
    basename = os.path.basename(filename_zero) 
    basename = basename[15:]
    basename =basename.replace('_3B_Visual','')
    basename = basename[0:8]+'_'+basename[9:]
    
    split_vals = [.7, .2, .1]
    
    cmd = f"python3 utils_convert_tfrecords_jpg.py --label_input={labels_dir}/{basename}/labels.npz --data_dir={labels_dir}/tf_records --tiles_dir={labels_dir}/{basename}/tiles --pbtxt=data/marine_debris.pbtxt --record_name {basename} --split_vals .7, .2, .1"
    
    os.system(cmd)
