"""Example AWS Lambda function for chip-n-scale"""
import os
import pg8000
from typing import Dict, Any
import numpy as np
from download_and_predict.base import DownloadAndPredict
from download_and_predict.custom_types import SQSEvent


class OD_DownloadAndPredict(DownloadAndPredict):
    """
    base object DownloadAndPredict implementing all necessary methods to
    make machine learning predictions
    """

    def __init__(self, imagery: str, db: str, prediction_endpoint: str):
        super(DownloadAndPredict, self).__init__()
        self.imagery = imagery
        self.db = db
        self.prediction_endpoint = prediction_endpoint

    def format_od_preds(self, content: Dict):
        preds = content['predictions']
        preds_out = []
        for _, pred in enumerate(preds):
            # print(f'\nPrediction number {pi}')
            n_ods = int(pred['num_detections'])
            pred_list = []
            for i in range(n_ods):
                pred_i = {}
                if pred['detection_scores'][i] > 0.1:
                    # print(pred['detection_classes'][i], pred['detection_scores'][i], pred['detection_boxes'][i])
                    pred_i["detection_classes"]= pred['detection_classes'][i]
                    pred_i["detection_scores"] = pred['detection_scores'][i]
                    pred_i["detection_boxes"] = pred['detection_boxes'][i]
                    pred_list.append(pred_i)
            preds_out.append(pred_list)
        return preds_out


def handler(event: SQSEvent, context: Dict[str, Any]) -> None:
    # read all our environment variables to throw errors early
    imagery = os.getenv('TILE_ENDPOINT')
    db = os.getenv('DATABASE_URL')
    prediction_endpoint = os.getenv('PREDICTION_ENDPOINT')
    assert (imagery)
    assert (db)
    assert (prediction_endpoint)
    # instantiate our DownloadAndPredict class
    dap = OD_DownloadAndPredict(
        imagery=imagery,
        db=db,
        prediction_endpoint=prediction_endpoint
    )
    # get tiles from our SQS event
    tiles = dap.get_tiles(event)
    print(tiles)
    # construct a payload for our prediction endpoint
    tile_indices, payload = dap.get_prediction_payload(tiles)
    # send prediction request
    content = dap.post_prediction(payload)
    preds = dap.format_od_preds(content)
    # save prediction request to db
    dap.save_to_db(
        tile_indices,
        preds,
        result_wrapper=lambda x: pg8000.PGJsonb(x)
    )
