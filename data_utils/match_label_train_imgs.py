"""
To match label and train images from xView since labels were only
selected for euro, china and europ

auother: @developmentseed

match_label_train_imgs.py

~~~
usage:
     python3 match_label_train_imgs.py --geojson=xView/xView_training_aus_select-1_hl_cls.geojson \
                                       --train_images=xView/train_images
"""

import os
import sys
from os import makedirs, path as op

import geopandas as gpd
import shutil
import argparse

def get_region_image_ids(geojson, train_images):
    """only gether the images ids for the region
    """
    g_df = gpd.read_file(geojson)
    images = list(g_df['image_id'])
    nm = op.splitext(geojson)[0]
    if not op.isdir(nm):
        makedirs(nm)
    for image in images:
        if op.exists(op.join(train_images, image)):
            shutil.move(op.join(train_images, image), op.join(nm, image))
            print(f"{image} moved from {train_images} to new path {nm} !")

def parse_arg(args):
    desc = "get_region_image_ids"
    dhf = argparse.RawTextHelpFormatter
    parse0 = argparse.ArgumentParser(description= desc, formatter_class=dhf)
    parse0.add_argument('--geojson', help="selected geojson file of labels")
    parse0.add_argument('--train_images', help='directory to the train images downloaded from xView')
    return vars(parse0.parse_args(args))

def main(geojson, train_images):
    get_region_image_ids(geojson, train_images)


def cli():
    args = parse_arg(sys.argv[1:])
    main(**args)


if __name__ == "__main__":
    cli()
