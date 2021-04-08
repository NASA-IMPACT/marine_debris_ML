"""
The following scripts do:
- re-struct detection results from RDS;
- write bboxes as geometry;
- intersect bboxes with airport boundary, and only keep the predictions within the boundary;
- save the prediction at daily bases.

Run:
   python3 parse_detected_planes.py --csv predictions_from_RDS.csv --aoi airports.geojson --threshold_score 0.2 --output_path ./covid_ml_detections/

Author: @developmentseed.
"""
import os
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
import argparse

def read_large_csv(csv):
    """read large csv
    ---
    Args:
        csv: results from RDS database exported as csv
    Returns:
        df: pandas dataframe
    """
    c_size = 5000
    first_c = pd.read_csv(csv, chunksize=100, sep='delimiter')
    dfs = [chunck_df for chunck_df in pd.read_csv(csv, chunksize=c_size, sep='delimiter')]
    df = pd.concat(dfs)
    df['list']  = df['tile,output'].str.split(',',1).tolist()
    df = pd.DataFrame(df["list"].to_list(), columns=['tile', 'output'])

    print("Count of empty detection rows: ", df.loc[df.output == '[]', 'output'].count())
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
    print("LENGTH OF df_date: ", len(df_date))
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

def detections_to_geo_all(melted_df):
    """write prediction bboxes, scores, and cls to geo features
    ---
    Args:
        melted_df: dataframe
        date: date e.g. 2020_0130
    Returns:
        features: predictions saved as features
    """
    features = []
    df_date = melted_df
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

def map_plane_airports(melted_df, airport_geo, threshod, output_path):
    """only keep predictions that intersect with airport polygons
    ---
    Args:
        melted_df: dataframe
        airport_geo: airport AOIs path
    Returns:
        (None): intersected ML predictiom in geojson format
    """
    airport_aois = gpd.read_file(airport_geo)
    airport_aois_names = airport_aois.name
    for n in airport_aois_names:
        d = output_path+n
        if not os.path.exists(d):
            os.makedirs(d)
    plane_preds = detections_to_geo_all(melted_df)
    planes_df = gpd.GeoDataFrame(plane_preds)

    for name in airport_aois_names:
        airport_aois = airport_aois[airport_aois.name == name]
        for date in np.unique(melted_df['date']):
            plane_preds = detections_to_geo(melted_df, date)
            planes_df = gpd.GeoDataFrame(plane_preds)
            planes_df['tile'] = planes_df['properties'].apply(lambda x: x['tile'])
            planes_df['scene_id']= planes_df['properties'].apply(lambda x: x['scene_id'])
            planes_df['score']= planes_df['properties'].apply(lambda x: x['score'])
            planes_df = planes_df.drop(columns=['type', 'properties'])
            planes_df = planes_df[planes_df['score']>=threshod]
            try:
                airp_planes = gpd.overlay(airport_aois, planes_df, how='intersection', make_valid=True, keep_geom_type=True)
                print("merged sample: ", airp_planes.head())
                airp_planes.to_file(f'{output_path}/{name}/{date}_intersection.geojson', driver='GeoJSON')
                print('##'*30)
                print(f'{date}_inters.geojson wrote to the file!')
            except:
                print("didn't merge")
                continue

def parse_arg(args):
    desc = "get_planet_tiles"
    dhf = argparse.RawTextHelpFormatter
    parse0 = argparse.ArgumentParser(description= desc, formatter_class=dhf)
    parse0.add_argument('--csv', help="Raw predictions from RDS")
    parse0.add_argument('--aoi', help="AOI Geojson")
    parse0.add_argument('--threshold_score', help="Minimum confidence score allowed for valid detections")
    parse0.add_argument('--output_path', help="Where to write output detection geojsons")
    return vars(parse0.parse_args(args))


if __name__== "__main__":
    args = parse_arg(sys.argv[1:])
    csv, aoi, th_score, output_path = **args
    df = read_large_csv(csv)
    print("Length of dataframe: ", len(df))
    df_melt =melt_restruct_df(df)
    map_plane_airports(df_melt, aoi, th_score, output_path)
