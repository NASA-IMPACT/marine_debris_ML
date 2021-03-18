"""
This code was modified on top of Google tensorflow
(https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/using_your_own_dataset.md)
This code works similar to `label-maker package` when used with Label Maker and Tensor Flow object detection API.
To create a correct training data set for Tensor Flow Object Detection, we recommend you:

1. After running `label-maker images`, do `git clone https://github.com/tensorflow/models.git`
2. Install TensorFlow object detection by following this: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md

author: @developmentseed

usage:

folder=asia_p1
python3 utils_convert_tfrecords.py    \
        --label_input=$folder/labels.npz   \
        --data_dir=tf_records   \
        --tiles_dir=$folder/tiles    \
        --pbtxt=airplane_1class.pbtxt   \
        --record_name=$scene_id
"""

import os
import io
import glob
import numpy as np
from os import makedirs, path as op
import shutil

import pandas as pd
import tensorflow as tf
#import tensorflow.compat.v1 as tf

from PIL import Image
from object_detection.utils import dataset_util
from collections import namedtuple
from object_detection.utils import label_map_util

label_map_util.tf = tf.compat.v1
tf.gfile = tf.io.gfile

flags = tf.app.flags
flags.DEFINE_string('label_input', '', 'Path to the labels.npz input')
flags.DEFINE_string('data_dir', '', 'Directory to write tfrecords into')
flags.DEFINE_string('tiles_dir', '', 'Path to tiles dir that corresponds with labels.npz')
flags.DEFINE_string('pbtxt', '', 'Path to pbtxt of TF object detection')
flags.DEFINE_list('split_names', 'train, test, val', 'List of names for data set to be divided into')
flags.DEFINE_list('split_vals', '1.0, 0, 0', 'List of ratios for data set to be divided into')
flags.DEFINE_string('record_name', '', 'TFrecord output prefix')
FLAGS = flags.FLAGS

def class_text_to_int(pbtxt):
    labels = label_map_util.create_category_index_from_labelmap(
        pbtxt, use_display_name=True)
    return labels

def split(df, group):
    data = namedtuple('data', ['filename', 'object'])
    gb = df.groupby(group)
    return [data(filename, gb.get_group(x)) for filename, x in zip(gb.groups.keys(), gb.groups)]

def _move_files(df, tile_path, des_dir):
    for tile in df['filename']:
        tile_ = op.join(tile_path, tile)
        if op.exists(tile_):
            shutil.copy(tile_, op.join(des_dir,tile))

def remove_blank_tiles(labels, tiles_dir):
    removed_tiles = []
    tile_names = [tile for tile in labels.files]
    tile_names.sort()
    for tile_nm in tile_names:
        f_nm = op.basename(tile_nm)
        img= op.join(tiles_dir, 'train', f'{f_nm}.png')
        if not op.exists(img):
            removed_tiles.append(f_nm)
            # if file size < 7, remove
            #if op.getsize(img) < 7*1024:
                #removed_tiles.append(f_nm)
    print(removed_tiles)
    print(f"{len(removed_tiles)} tiles will be removed!")
    return removed_tiles

def _diff(lst1, lst2):
    return list(set(lst1) - set(lst2))


def create_tf_example(group, path):
    """Creates a tf.Example proto from sample tower image tile.

    Args:
     encoded tower_image_data: The jpg/png encoded data of the tower image.
    Returns:
     example: The created tf.Example.
    """
    with tf.gfile.GFile(op.join(path, '{}'.format(group.filename)), 'rb') as fid:
        encoded_jpg = fid.read()
    encoded_jpg_io = io.BytesIO(encoded_jpg)
    image = Image.open(encoded_jpg_io)
    width, height = image.size
    filename = group.filename.encode('utf8')
    image_format = 'png'
    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for _, row in group.object.iterrows():

        #normalize the bbox
        xmins.append(row['xmin'] / width)
        xmaxs.append(row['xmax'] / width)
        ymins.append(row['ymin'] / height)
        ymaxs.append(row['ymax'] / height)
        classes_text.append(row['class'].encode('utf8'))
        classes.append(int(row['class_id']))

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(image_format.encode('utf8')),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    labels = np.load(op.join(os.getcwd(), FLAGS.label_input))
    tiles_dir =op.join(os.getcwd(), FLAGS.tiles_dir)
    #all_tile_names = [tile for tile in labels.files]
    all_tile_names = [tile for tile in glob.glob(FLAGS.tiles_dir+'train/'+'*.png')]
    print(f'we have total {len(all_tile_names)} tiles')
    removed_tiles = remove_blank_tiles(labels, tiles_dir)
    tile_names = _diff(all_tile_names, removed_tiles)
    tile_names.sort()
    tiles = np.array(tile_names)
    region = FLAGS.label_input.split("/")[2]

    tf_tiles_info = []
    cl= class_text_to_int(FLAGS.pbtxt)
    print(cl)
    # re-assign the class map to 4 classes
    cl_id_dict = {1:1,4:2, 5:3, 6:2, 7:4, 11:1}
    for tile in tiles:
        tile = op.basename(tile)
        tile = tile[:-4]
        bboxes = labels[tile].tolist()

        width, height = 256, 256

        if bboxes:
            for bbox in bboxes:
                cl_id = int(bbox[4])
                if cl_id in [1, 4, 5, 6, 7, 11]:
                    cl_id=cl_id_dict[cl_id]
                    cl_str =cl[cl_id]['name']
                    print(cl_id, cl_str)
                    bbox = [max(0, min(255, x)) for x in bbox[0:4]]
                    y = [f'{tile}.png', width, height, cl_str, cl_id, bbox[0], bbox[1], bbox[2], bbox[3]]
                    tf_tiles_info.append(y)

    split_n_samps = [len(tf_tiles_info) * float(val) for val in FLAGS.split_vals]

    split_inds = np.cumsum(split_n_samps).astype(np.integer)

    column_name = ['filename', 'width', 'height', 'class', 'class_id', 'xmin', 'ymin', 'xmax', 'ymax']
    df = pd.DataFrame(tf_tiles_info, columns=column_name)
    # shuffle the dataframe
    df = df.sample(frac=1)
    print(df.head(20))

    split_dfs = np.split(df, split_inds[:-1])

    train_df = split_dfs[0]
    print('train samples {}'.format(train_df.shape[0]))

    test_df = split_dfs[1]
    print('test samples {}'.format(test_df.shape[0]))

    val_df = split_dfs[2]
    print('val samples {}'.format(val_df.shape[0]))

    ### saving for training data stats
    train_df.to_csv(f'{FLAGS.data_dir}'+'/'+f'{FLAGS.record_name}_train.csv')
    test_df.to_csv(f'{FLAGS.data_dir}'+'/'+f'{FLAGS.record_name}_test.csv')
    val_df.to_csv(f'{FLAGS.data_dir}'+'/'+f'{FLAGS.record_name}_val.csv')

    train_dir = op.join(FLAGS.tiles_dir, 'train')
    test_dir = op.join(FLAGS.tiles_dir, 'test')
    val_dir = op.join(FLAGS.tiles_dir, 'val')

    dir_list = train_dir, test_dir, val_dir
    tile_path = FLAGS.tiles_dir

    for d in dir_list:
        if not op.isdir(d):
            makedirs(d)
        #move train images

        _move_files(train_df, tile_path, d)

        # move test images
        _move_files(test_df, tile_path, d)

        # move validation images
        _move_files(val_df, tile_path, d)

    # train TFRecords Creation
    writer = tf.python_io.TFRecordWriter(f'{FLAGS.data_dir}'+'/'+f'{FLAGS.record_name}_train.records')
    grouped = split(train_df, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, train_dir)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created train tf records.')

    # Test TFRecords Creation
    writer = tf.python_io.TFRecordWriter(f'{FLAGS.data_dir}'+'/'+f'{FLAGS.record_name}_test.records')
    grouped = split(test_df, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, test_dir)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created test tf records.' )

    # Val TFRecords Creation
    writer = tf.python_io.TFRecordWriter(f'{FLAGS.data_dir}'+'/'+f'{FLAGS.record_name}_val.records')
    grouped = split(val_df, 'filename')
    for group in grouped:
        tf_example = create_tf_example(group, val_dir)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print('Successfully created val tf records.')

if __name__ == '__main__':
    tf.app.run()
