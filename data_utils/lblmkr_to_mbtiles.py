
"""
Script to construct an mbtiles folder from label-maker tiles output.
See https://github.com/developmentseed/label-maker

Author: @developmentseed
Run:
    python3 lblmkr_to_mbtiles.py --tiles_dir=data/tiles/ \
            --mbtiles_dir=data/tiles_mb/ \
            --zoom=16
"""

import os, sys, glob, argparse
from shutil import copyfile

def main(tiles_dir, mbtiles_dir, zoom):
    """
    reads in a tiles directory exported from label-maker, parses the tile names
    and separates the x and y coordinates as well as zoom level from the filenames,
    renaming the tile paths to a nested structure of zoom/x/y.jpg
    """
    if not os.path.exists(mbtiles_dir):
        os.makedirs(mbtiles_dir)
    z_dir = os.path.join(mbtiles_dir,str(zoom))

    if not os.path.exists(z_dir):
        os.makedirs(z_dir)

    x_list = []
    y_list = []

    tiles_list = glob.glob(tiles_dir+'*.jpg')

    for t in tiles_list:
        filename_split = os.path.splitext(t)
        filename_zero, fileext = filename_split
        basename = os.path.basename(filename_zero)
        t_split = basename.split("-")
        x_list.append(t_split[0])
        y_list.append(t_split[1])
        x_dir = os.path.join(z_dir,str(t_split[0]))
        if not os.path.exists(x_dir):
          os.makedirs(x_dir)
        dst_t = x_dir+'/'+t_split[1]+'.jpg'
        copyfile(t, dst_t)

def parse_arg(args):
    desc = "construct mbtiles folder from label-maker tiles output"
    dhf = argparse.RawTextHelpFormatter
    parse0 = argparse.ArgumentParser(description= desc, formatter_class=dhf)
    parse0.add_argument('--tiles_dir', help="data/tiles/")
    parse0.add_argument('--mbtiles_dir', help="data/tiles_mb/")
    parse0.add_argument('--zoom', help="16")
    return vars(parse0.parse_args(args))


def cli():
    args = parse_arg(sys.argv[1:])
    main(**args)

if __name__ == "__main__":
    cli()
