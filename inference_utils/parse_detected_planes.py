"""
The following scripts do:
- re-struct detection results from RDS;
- write bboxes as geometry;
- intersect bboxes with airport boundary, and only keep the predictions within the boundary;
- save the prediction at daily bases.

Run:
   python3 parse_detected_planes.py

Author: @developmentseed.
"""

import json
import pandas as pd
import numpy as np
import mercantile
import affine
import shapely
from shapely import geometry
import numpy as np
from geojson import Feature, FeatureCollection as fc
import geopandas as gpd

def read_large_csv(csv):
    """read large csv
    ---
    Args:
        csv: results from RDS database exported as csv
    Returns:
        df: pandas dataframe
    """
    c_size = 5000
    dfs = [chunck_df for chunck_df in pd.read_csv(csv, chunksize=c_size)]
    df = pd.concat(dfs)
    return df
def melt_restruct_df(df):
    """melt and restructure dataframe
    ---
    Args:
        df: dataframe
    Returns:
        df_melted: melted and reconstructed dataframe
    """
    df['output'] = df.output.apply(json.loads)
    df = df[df.output.apply(bool)]
    df['scene_id'] = df.tile.apply(lambda x: x.split('-')[0])
    df['tile_id'] = df.tile.apply(lambda x: '-'.join(x.split('-')[1:]))
    outputs = df.output.apply(pd.Series)
    outputs = outputs.rename(columns = lambda x : 'outputs_' + str(x))
    df_output = pd.concat([df[:], outputs[:]], axis=1).drop(["tile", "output"], axis=1)
    df_melted = df_output.melt(id_vars=["tile_id", "scene_id"], var_name="detection_index", value_name="detection")
    df_melted = df_melted.drop('detection_index', axis=1).dropna()
    df_melted['date'] = df_melted['scene_id'].apply(lambda x: x.split('_')[0])

    return df_melted

def detections_to_geo(melted_df, date):
    """write prediction bboxes, scores, and cls to geo features
    ---
    Args:
        melted_df: dataframe
        date: date e.g. 2020_0130
    Returns:
        features: predictions saved as features
    """
    features = []
    df_date = melted_df[melted_df['date']==date]
    for i, row in df_date.iterrows():
        tile_id = row['tile_id']
        x, y, z = tile_id.split('-')
        b = mercantile.bounds(int(x), int(y), 16)
        width = b[2] - b[0]
        height = b[3] - b[1]
        a = affine.Affine(width / 256, 0.0, b[0], 0.0, (0 - height / 256), b[3])
        a_lst = [a.a, a.b, a.d, a.e, a.xoff, a.yoff]
        bbox=row['detection']['detection_boxes']
        bbox_rev= [bbox[1], bbox[0], bbox[3], bbox[2]]
        score=row['detection']['detection_scores']
        scene_id = row['scene_id']
        tile_id = row['tile_id']
        bbox_256 = (np.array(bbox_rev)*256).astype(np.int)
        cls = int(row['detection']['detection_classes'])
        geographic_bbox = shapely.affinity.affine_transform(geometry.box(*bbox_256), a_lst)
        features.append(Feature(geometry=geographic_bbox,
                                properties=dict(tile=tile_id, scene_id=scene_id, cls=cls, score=float(score))))
    return features

def map_plane_airports(melted_df, airport_geo, threshod):
    """only keep predictions that intersect with airport polygons
    ---
    Args:
        melted_df: dataframe
        airport_geo: airport AOIs path
    Returns:
        (None): intersected ML predictiom in geojson format
    """
    airport_aois = gpd.read_file(airport_geo)
    for date in np.unique(melted_df['date']):
        plane_preds = detections_to_geo(melted_df, date)
        planes_df = gpd.GeoDataFrame(plane_preds)
        planes_df['tile'] = planes_df['properties'].apply(lambda x: x['tile'])
        planes_df['scene_id']= planes_df['properties'].apply(lambda x: x['scene_id'])
        planes_df['score']= planes_df['properties'].apply(lambda x: x['score'])
        planes_df = planes_df.drop(columns=['type', 'properties'])
        planes_df = planes_df[planes_df['score']>=threshod]
        print(planes_df.head())
        airp_planes = gpd.overlay(airport_aois, planes_df, how='intersection', make_valid=True, keep_geom_type=True)
        print(airp_planes.head())
        airp_planes.to_file(f'{date}_intersection.geojson', driver='GeoJSON')
        print('##'*30)
        print(f'{date}_inters.geojson wrote to the file!')

if __name__== "__main__":
    csv = 'results_more_to_may.csv'
    th_score = 0.2
    df = read_large_csv(csv)
    df_melt =melt_restruct_df(df)
    airports = "Covid_7AOIs_Airports.geojson"
    map_plane_airports(df_melt, airports, th_score)
