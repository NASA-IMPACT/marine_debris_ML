"""
dap_send-images.py

Populate an SQS queue from the gaint list of objects/images from S3 bucket.

@author:developmentseed

use:
python3 dap_send-images.py --imgs_txt_files img_inds.txt
"""


import numpy as np
import boto3
from tqdm import tqdm
from joblib import Parallel, delayed
import argparse
import sys


def send_tile_msg_batch(queue, msg_template, batch_inds):
    """Send a batch of message."""
    entries = []
    for id_num, (scene_id, x, y, z) in enumerate(batch_inds):
        entries.append({'MessageBody':msg_template.format(scene_id=scene_id, x=x, y=y, z=z),
                        'Id':str(id_num)})

    try:
        response = queue.send_messages(Entries=entries)

        if response.get('Failed'):
            print(response.get('Failed'))

    except Exception as e:
        print('Error in pushing tiles: {}. Error:\n{}'.format(entries, e))


def send_tile_msg(queue, msg_template, scene_id, x, y, z):
    """Send a single message to SQS queue."""
    msg_body = msg_template.format(scene_id=scene_id, x=x, y=y, z=z)
    try:
        response = queue.send_message(MessageBody=msg_body)

        if response.get('Failed'):
            print(response.get('Failed'))

    except Exception as e:
        print('Error in pushing tile: {}. Error:\n{}'.format(msg_body, e))

def main(files):
    # files = list(files)
    msg_template='{{"scene_id":{scene_id}, "x":{x},"y":{y},"z":{z}}}'
    sqs = boto3.resource('sqs', region_name='us-east-1')
    queue = sqs.Queue('	https://sqs.us-east-1.amazonaws.com/552819999234/covidmlTileQueue')
    for tile_ind_file in files:
        print('\nProcessing tile index file: {}'.format(tile_ind_file))
       # Get tiles from text file if pre-computed
        with open(tile_ind_file, 'r') as tile_file:
            tile_inds = [ind.strip().split('-') for ind in tile_file if len(ind)]
        batch_size = 10
        batch_inds = len(tile_inds) // batch_size + 1
        tile_ind_batches = [tile_inds[b1 * batch_size:b2 * batch_size]
                            for b1, b2 in zip(np.arange(0, batch_inds - 1),
                                              np.arange(1, batch_inds))]
        print('Found {} tiles, pushing to SQS Queue: {}'.format(len(tile_inds), queue.url))
        Parallel(32, prefer='threads')(delayed(send_tile_msg_batch)(
        queue, msg_template, tile_ind_batch) for tile_ind_batch
        in tqdm(tile_ind_batches))

def parse_arg(args):
    desc = "populate_sqs"
    dhf = argparse.RawTextHelpFormatter
    parse0 = argparse.ArgumentParser(description= desc, formatter_class=dhf)
    parse0.add_argument('--inds_txt_files', nargs='*', help='txt files for tiles index from geodex')
    return vars(parse0.parse_args(args))

def cli():
    args = parse_arg(sys.argv[1:])
    main(args['inds_txt_files'])

if __name__=="__main__":
    cli()
