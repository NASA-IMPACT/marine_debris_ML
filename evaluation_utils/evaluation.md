# Object Detection Model Evaluation Guide
This is a quick guide to run evaluation with the accompanying code.

## **Part 1**
We are using https://github.com/tensorflow/models/blob/master/research/object_detection/inference/infer_detections.py to run inference directly on test dataset tfrecords.

That script requires arguments for the input tfrecords files (comma separated list), output tfrecords path (containes detection boxes and ground truth boxes) and frozen inference graph. 

Example usage:

from ` % models/research/object_detection/` run `python3 inference/infer_detections.py --input_tfrecord_paths=testrecords/test.records,testrecords/test1.records,testrecords/test2.records,testrecords/test3.records --output_tfrecord_path=tf_object_detection_cm/detections.record --inference_graph=export/frozen_inference_graph.pb`

Once this script has finished you should have your compiled detections tfrecords file. This file is to be used in part 2.

## **Part 2**

For this part you will probably need to apply a patch fix to a local TFOD script like I did (due to a bug with the tensorflow version that the object detection API depends on ref: https://github.com/tensorflow/models/issues/3252).  To solve, in `models/research/object_detection/metrics/tf_example_parser.py` replace the former with the latter:

```
class StringParser(data_parser.DataToNumpyParser):

  def __init__(self, field_name):
    self.field_name = field_name

  def parse(self, tf_example):
    return "".join(tf_example.features.feature[self.field_name]
                   .bytes_list.value) if tf_example.features.feature[
                       self.field_name].HasField("bytes_list") else None
```
to
```
class StringParser(data_parser.DataToNumpyParser):
  """Tensorflow Example string parser."""
  def __init__(self, field_name):
    self.field_name = field_name
  def parse(self, tf_example):
    if tf_example.features.feature[self.field_name].HasField("bytes_list"):
        result = tf_example.features.feature[self.field_name].bytes_list.value
        result = "".join([x if type(x)=='str' else x.decode('utf-8') for x in result])
    else:
        result = None
    return result
```

After that, you are ready to run the evaluation script. It will generate a confusion matrix (columns are predicted, rows are ground truth), precision, recall, f1 and mAP scores.

The script requires input for the detections tfrecords file (which we get from part 1), a label map (.pbtxt file), and a path for the output csv containing the confusion matrix and scores.

Example usage:

`python3 eval_cmatrix_f1_map.py --detections_record=tf_object_detection_cm/detections.record --label_map=marine_debris.pbtxt --output_path=marine_debris_cm.csv`

You should get a confusion matrix printed out as well as the scores.

Formatted example of resulting metrics:

| | Predicted debris | Predicted none | 
|---| ---| ---| 
| **True debris** | 38 | 16 | 
| **True none** | 11 | 0 | 


| True Positive |  False Positive |  False Negative | 
|---| ---| ---| 
| 38 | 11 | 16 | 


| category |  precision_@0.5IOU |  recall_@0.5IOU | map_@0.5IOU  | f1_@0.5IOU | 
|---| ---| ---| ---|  ---| 
| marine_debris | 0.78  | 0.70  | 0.78 | 0.74
