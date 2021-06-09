import os, sys
from os import makedirs, path as op
import shutil
from glob import glob
import subprocess

tif_tiles = glob(f''+'/**/*.tif',recursive=True)

labels_dir = f''

for tif in tif_tiles:
    filename_split = os.path.splitext(tif) 
    filename_zero, fileext = filename_split 
    basename = os.path.basename(filename_zero) 
    basename = basename[15:]
    basename =basename.replace('_3B_Visual','')
    basename = basename[0:8]+'_'+basename[9:]
    
    subprocess.run([f'python', 'utils_convert_tfrecords.py',f'--label_input={labels_dir}/{basename}/labels.npz', f'--data_dir={labels_dir}/tf_records', f'--tiles_dir={labels_dir}/{basename}/tiles', f'--pbtxt=data/marine_debris.pbtxt', f'--record_name={basename}'])
