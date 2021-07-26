"""
This is adapted from Tensorflow (https://github.com/tensorflow/models/tree/master/research/object_detection);
Save this code under the directory `models/research/object_detection/`
To use, run:
python3 tf_od_predict_image_aug_to_geo_corrected.py --model_name=marine_debris \
                         --path_to_label=data/marine_debris.pbtxt \
                         --test_image_path=test_images
"""
import os
from os import makedirs, path as op
import sys
import glob
import tensorflow as tf

from PIL import Image


import numpy as np
import json

from utils import label_map_util
from utils import visualization_utils as vis_util

from skimage import exposure

from geojson import Feature, FeatureCollection as fc
import mercantile
import affine
import shapely
from shapely import geometry


import pandas as pd

flags = tf.app.flags
flags.DEFINE_string('model_name', '', 'Path to frozen detection graph')
flags.DEFINE_string('path_to_label', '', 'Path to label file')
flags.DEFINE_string('test_image_path', '', 'Path to test imgs and output diractory')
flags.DEFINE_string('scene_id', '', 'Geojson output prefix')
FLAGS = flags.FLAGS

def darken_img(image):
    gamma_corrected = exposure.adjust_gamma(image, 2)
    return gamma_corrected


def load_image_into_numpy_array(image):
    (im_width, im_height) = image.size
    return np.array(image.getdata()).reshape((im_height, im_width, 3)).astype(np.uint8)


def tf_od_pred():
    geoname = test_image_path.split('/')[-1]
    features = []
    with detection_graph.as_default():
        with tf.Session(graph=detection_graph) as sess:
            # Definite input and output Tensors for detection_graph
            image_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
            # Each box represents a part of the image where a particular object was detected.
            detection_boxes = detection_graph.get_tensor_by_name('detection_boxes:0')
            # Each score represent how level of confidence for each of the objects.
            # Score is shown on the result image, together with the class label.
            detection_scores = detection_graph.get_tensor_by_name('detection_scores:0')
            detection_classes = detection_graph.get_tensor_by_name('detection_classes:0')
            num_detections = detection_graph.get_tensor_by_name('num_detections:0')
            # idx = 0
            for image_path in test_imgs:
                if op.getsize(image_path) <= 4*1024:
                    continue
                image = Image.open(image_path)
                image_np = load_image_into_numpy_array(image)
                #image_np = darken_img(image_np)
                print("image_path: ", image_path)
                # the array based representation of the image will be used later in order to prepare the
                # result image with boxes and labels on it.
                # Expand dimensions since the model expects images to have shape: [1, None, None, 3]
                image_np_expanded = np.expand_dims(image_np, axis=0)
                # Actual detection.
                (boxes, scores, classes, num) = sess.run(
                  [detection_boxes, detection_scores, detection_classes, num_detections],
                  feed_dict={image_tensor: image_np_expanded})
                boxes = np.squeeze(boxes)
                classes = np.squeeze(classes).astype(np.int32)
                scores = np.squeeze(scores)
                indices = np.argwhere(scores>=0.2)
                bboxes=np.squeeze(boxes[indices])
                scores = np.squeeze(scores[indices])
                classes = np.squeeze(classes[indices])

                basen = op.basename(image_path)
                basename = op.splitext(basen)[0]
                tile_x, tile_y, tile_z = [int(x) for x in basename.split('-')]
                b = mercantile.bounds(tile_x, tile_y, tile_z)
                width = b[2] - b[0]
                height = b[3] - b[1]

                a = affine.Affine(width / 256, 0.0, b[0], 0.0, (0 - height / 256), b[3])
                a_lst = [a.a, a.b, a.d, a.e, a.xoff, a.yoff]
                bbox_256 = (bboxes * 256).astype(np.int)
                bboxes_256 = np.squeeze(bbox_256)
                print(f"bboxes_256: {bboxes_256}")
                try:

                    for i, bbox in enumerate(bboxes_256.tolist()):
                        print("bbox before: ", bbox)
                        pred = [bbox[1], bbox[0], bbox[3], bbox[2]]
                        print("bbox after: ", pred)

                        geographic_bbox = shapely.affinity.affine_transform(geometry.box(*pred), a_lst)
                        features.append(Feature(geometry=geographic_bbox,
                                properties=dict(tile=basename, cls=int(classes[i]), score=float(scores[i]))))
                except TypeError:
                    continue

    
    geoname = FLAGS.scene_id
    print(f"features for {geoname} are {features}")
    with open(f"{FLAGS.test_image_path}/{geoname}.geojson", 'w') as geoj:
        json.dump(fc(features), geoj)


if __name__ =='__main__':
    # load your own trained model inference graph. This inference graph was generated from
    # export_inference_graph.py under model directory, see `models/research/object_detection/`
    model_name = op.join(os.getcwd(), FLAGS.model_name)
    # Path to frozen detection graph.
    path_to_ckpt = op.join(model_name,  'frozen_inference_graph.pb')
    # Path to the label file
    path_to_label = op.join(os.getcwd(), FLAGS.path_to_label)
    #only train on buildings
    num_classes = 1
    #Directory to test images path
    #test_image_path = op.join(os.getcwd(), FLAGS.test_image_path)
    test_image_path = FLAGS.test_image_path
    test_imgs = glob.glob(test_image_path + "/*.jpg")
    print(f"test_imgs: {test_imgs}")
    ############
    #Load the frozen tensorflow model
    #############
    detection_graph = tf.Graph()
    with detection_graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(path_to_ckpt, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='')
    ############
    #Load the label file
    #############
    label_map = label_map_util.load_labelmap(path_to_label)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=num_classes, use_display_name=True)
    category_index = label_map_util.create_category_index(categories)
    tf_od_pred()
