import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from object_detection.core import standard_fields
from object_detection.metrics import tf_example_parser
from object_detection.utils import label_map_util

flags = tf.app.flags

flags.DEFINE_string('label_map', None, 'Path to the label map')
flags.DEFINE_string('detections_record', None, 'Path to the detections record file')
flags.DEFINE_string('output_path', None, 'Path to the output the results in a csv.')

FLAGS = flags.FLAGS

IOU_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5

def compute_iou(groundtruth_box, detection_box):
    g_ymin, g_xmin, g_ymax, g_xmax = tuple(groundtruth_box.tolist())
    d_ymin, d_xmin, d_ymax, d_xmax = tuple(detection_box.tolist())

    xa = max(g_xmin, d_xmin)
    ya = max(g_ymin, d_ymin)
    xb = min(g_xmax, d_xmax)
    yb = min(g_ymax, d_ymax)

    intersection = max(0, xb - xa + 1) * max(0, yb - ya + 1)

    boxAArea = (g_xmax - g_xmin + 1) * (g_ymax - g_ymin + 1)
    boxBArea = (d_xmax - d_xmin + 1) * (d_ymax - d_ymin + 1)

    return intersection / float(boxAArea + boxBArea - intersection)

def process_detections(detections_record, categories):
    record_iterator = tf.python_io.tf_record_iterator(path=detections_record)
    data_parser = tf_example_parser.TfExampleDetectionAndGTParser()

    confusion_matrix = np.zeros(shape=(len(categories) + 1, len(categories) + 1))

    image_index = 0
    for string_record in record_iterator:
        example = tf.train.Example()
        example.ParseFromString(string_record)
        decoded_dict = data_parser.parse(example)

        image_index += 1

        if decoded_dict:
            groundtruth_boxes = decoded_dict[standard_fields.InputDataFields.groundtruth_boxes]
            groundtruth_classes = decoded_dict[standard_fields.InputDataFields.groundtruth_classes]

            detection_scores = decoded_dict[standard_fields.DetectionResultFields.detection_scores]
            detection_classes = decoded_dict[standard_fields.DetectionResultFields.detection_classes][detection_scores >= CONFIDENCE_THRESHOLD]
            detection_boxes = decoded_dict[standard_fields.DetectionResultFields.detection_boxes][detection_scores >= CONFIDENCE_THRESHOLD]

            matches = []

            if image_index % 100 == 0:
                print("Processed %d images" %(image_index))

            for i in range(len(groundtruth_boxes)):
                for j in range(len(detection_boxes)):
                    iou = compute_iou(groundtruth_boxes[i], detection_boxes[j])

                    if iou > IOU_THRESHOLD:
                        matches.append([i, j, iou])

            matches = np.array(matches)
            if matches.shape[0] > 0:
                # Sort list of matches by descending IOU so we can remove duplicate detections
                # while keeping the highest IOU entry.
                matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

                # Remove duplicate detections from the list.
                matches = matches[np.unique(matches[:,1], return_index=True)[1]]

                # Sort the list again by descending IOU. Removing duplicates doesn't preserve
                # our previous sort.
                matches = matches[matches[:, 2].argsort()[::-1][:len(matches)]]

                # Remove duplicate ground truths from the list.
                matches = matches[np.unique(matches[:,0], return_index=True)[1]]

            for i in range(len(groundtruth_boxes)):
                if matches.shape[0] > 0 and matches[matches[:,0] == i].shape[0] == 1:
                    confusion_matrix[groundtruth_classes[i] - 1][detection_classes[int(matches[matches[:,0] == i, 1][0])] - 1] += 1
                else:
                    confusion_matrix[groundtruth_classes[i] - 1][confusion_matrix.shape[1] - 1] += 1

            for i in range(len(detection_boxes)):
                if matches.shape[0] > 0 and matches[matches[:,1] == i].shape[0] == 0:
                    confusion_matrix[confusion_matrix.shape[0] - 1][detection_classes[i] - 1] += 1
        else:
            print("Skipped image %d" % (image_index))

    print("Processed %d images" % (image_index))

    return confusion_matrix

def display(confusion_matrix, categories, output_path):
    print("\nConfusion Matrix:")
    print(confusion_matrix, "\n")
    results = []

    for i in range(len(categories)):
        id = categories[i]["id"] - 1
        name = categories[i]["name"]
        
        total_target = np.sum(confusion_matrix[id,:])
        total_predicted = np.sum(confusion_matrix[:,id])

        precision = float(confusion_matrix[id, id] / total_predicted)
        recall = float(confusion_matrix[id, id] / total_target)
        f1 = 2 * precision * recall / (precision + recall)

        recalls = []
        prec_at_rec = []
        for recall_level in np.linspace(0.0, 1.0, 11):
            args = np.argwhere(recall >= recall_level).flatten()
            for row in args:
                recalls.append(recall)
                prec = precision #[row]
                prec_at_rec.append(prec)
        prec_at_rec = np.array(prec_at_rec)
        avg_prec = np.mean(prec_at_rec)


        results.append({'category' : name, 'precision_@{}IOU'.format(IOU_THRESHOLD) : precision, 'recall_@{}IOU'.format(IOU_THRESHOLD) : recall, 'map_@{}IOU'.format(IOU_THRESHOLD) : avg_prec, 'f1_@{}IOU'.format(IOU_THRESHOLD) : f1})

    df = pd.DataFrame(results)
    print(df)
    df.to_csv(output_path)

def main(argv):
    del argv
    required_flags = ['detections_record', 'label_map', 'output_path']
    for flag_name in required_flags:
        if not getattr(FLAGS, flag_name):
            raise ValueError('Flag --{} is required'.format(flag_name))

    label_map = label_map_util.load_labelmap(FLAGS.label_map)
    categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=100, use_display_name=True)

    confusion_matrix = process_detections(FLAGS.detections_record, categories)

    display(confusion_matrix, categories, FLAGS.output_path)

if __name__ == '__main__':
    tf.app.run(main)
