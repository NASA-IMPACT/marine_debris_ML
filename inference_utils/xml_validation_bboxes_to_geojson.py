"""
Write bboxes that validated by expert mappers from CVAT tool to geojson

Author: @developmentseed

Run:
    python3 xml_validation_bboxes_to_geojson.py --xml_path=Airplanes_data_team_validated
"""

import os
from os import path as op
import json
import pandas as pd
import xml.etree.ElementTree as etree
import mercantile
import affine
import shapely
from shapely import geometry
import numpy as np
from geojson import Feature, FeatureCollection as fc
import click

def get_bboxes(xml):
    root = etree.parse(xml).getroot()
    image_entries = root.findall('image')
    bboxes = []
    for image in image_entries:
        for bb in image.findall('box'):
            name = image.get('name')
            bbox = [round(float(bb.get(coord_key))) for coord_key in ['ytl', 'xtl', 'ybr', 'xbr']]
            score = bb.findall("./*[@name='score']")[0].text
            bboxes.append([name, bbox, score])
    return bboxes

def detections_to_geo(df_date, xml_dir, date):
    """write prediction bboxes, scores, and cls to geo features
    -----df_date.head()--------
    file_name, bbox, score
    {path}/{scene_id}/x-y-z.png, [10, 157, 47, 189], 0.93
    ----------------------------
    ---
    Args:
        melted_df: dataframe
        date: date e.g. 2020_0130
    Returns:
        features: predictions saved as features
    """
    features = []
    date_info = [(file_nm, bbox, score) for
               file_nm, bbox, score in zip(df_date.file_name, df_date.bbox, df_date.score)]
    for i, info in enumerate(date_info):
        file_, scene_id, image = info[0].split('/')
        tile = image.split('.')[0]
        x, y, z = tile.split('-')
        b = mercantile.bounds(int(x), int(y), 16)
        width = b[2] - b[0]
        height = b[3] - b[1]
        a = affine.Affine(width / 256, 0.0, b[0], 0.0, (0 - height / 256), b[3])
        a_lst = [a.a, a.b, a.d, a.e, a.xoff, a.yoff]
        bbox = info[1]
        scene_id = scene_id
        tile_id = tile
        score = info[2]
        cls = 1
        geographic_bbox = shapely.affinity.affine_transform(geometry.box(*bbox), a_lst)
        features.append(Feature(geometry=geographic_bbox,
                                properties=dict(tile=tile_id, scene_id=scene_id, cls=cls, score=score)))
    with open(f'{xml_dir}/{date}_validated.geojson', 'w') as out_geo:
        json.dump(fc(features), out_geo)

@click.command(short_help="create geojson for ml detection from human validated xml files")
@click.option('--xml_path', help="the directory that contains xml files")

def main(xml_path):
    xmls = [op.join(xml_path, xml) for xml in os.listdir(xml_path) if xml.endswith('.xml')]
    for xml in xmls:
        date = (op.basename(xml).split('.')[0]).split('_')[1]
        col_names = ['file_name', 'bbox', 'score']
        bboxes = get_bboxes(xml)
        df = pd.DataFrame(bboxes, columns=col_names)
        detections_to_geo(df, xml_path, date)


if __name__=="__main__":
    main()
