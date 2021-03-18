"""
Lambda for downloading images, packaging them for prediction, sending them
to a remote ML serving image, and saving them
@author:Development Seed
"""

import json
from base64 import b64encode
from urllib.parse import urlparse
from typing import Dict, List, NamedTuple, Callable, Optional, Tuple, Any, Iterator

from mercantile import Tile
import requests
import pg8000

from download_and_predict.custom_types import SQSEvent


class DownloadAndPredict(object):
    """
    base object DownloadAndPredict implementing all necessary methods to
    make machine learning predictions
    """

    def __init__(self, imagery: str, db: str, prediction_endpoint: str):
        super(DownloadAndPredict, self).__init__()
        self.imagery = imagery
        self.db = db
        self.prediction_endpoint = prediction_endpoint

    @staticmethod
    def get_tiles(event: SQSEvent) -> List[Tile]:
        """
        Return the body of our incoming SQS messages as an array of mercantile Tiles
        Expects events of the following format:

        { 'Records': [ { "body": '{"scene_id": 123x, "x": 4, "y": 5, "z":3 }' }] }

        """
        return [
            json.loads(record['body'])
            for record
            in event['Records']
        ]

    @staticmethod
    def b64encode_image(image_binary: bytes) -> str:
        return b64encode(image_binary).decode('utf-8')

    def get_images(self, tiles: List[Dict]) -> Iterator[Tuple[Tile, bytes]]:
        for tile in tiles:
            url = self.imagery.format(scene_id=tile['scene_id'], x=tile['x'], y=tile['y'], z=tile['z'])
            r = requests.get(url)
            new_tile = f"{tile['scene_id']}-{tile['x']}-{tile['y']}-{tile['z']}"
            yield (new_tile, r.content)

    def get_prediction_payload(self, tiles: List[Dict]) -> Tuple[List[Tile], str]:
        """
        tiles: list mercantile Tiles
        imagery: str an imagery API endpoint with three variables {z}/{x}/{y} to replace

        Return:
        - an array of b64 encoded images to send to our prediction endpoint
        - a corresponding array of tile indices

        These arrays are returned together because they are parallel operations: we
        need to match up the tile indicies with their corresponding images
        """
        tiles_and_images = self.get_images(tiles)
        tile_indices, images = zip(*tiles_and_images)

        #inputs for tf-serving instead of image_bytes for object detection
        instances = [dict(inputs=dict(b64=self.b64encode_image(img))) for img in images]
        payload = json.dumps(dict(instances=instances))

        return (list(tile_indices), payload)

    def post_prediction(self, payload: str) -> Dict[str, Any]:
        r = requests.post(self.prediction_endpoint, data=payload)
        r.raise_for_status()
        return r.json()

    def save_to_db(self, tiles: List[Tile], results: List[Any], result_wrapper: Optional[Callable] = None) -> None:

        # tile and results need to be lists of the same length! (see line 124)

        """
        Save our prediction results to the provided database
        tiles: list mercantile Tiles
        results: list of predictions
        db: str database connection string

        """
        db = urlparse(self.db)

        conn = pg8000.connect(
            user=db.username,
            password=db.password,
            host=db.hostname,
            database=db.path[1:],
            port=db.port
        )
        cursor = conn.cursor()

        for i, output in enumerate(results):
            result = result_wrapper(output) if result_wrapper else output
            cursor.execute("INSERT INTO results VALUES (%s, %s) ON CONFLICT (tile) DO UPDATE SET output = %s",
                           (tiles[i], result, result))

        conn.commit()
        conn.close()
