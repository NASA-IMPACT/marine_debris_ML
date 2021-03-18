"""
Use label maker to create training dataset

author: @developmentseed

under built docker image 'covid_sat_ml':

run:
    python3 label_maker_get_training_data.py
"""

import os
from os import makedirs, path as op
import json
import shutil

def create_training_data(config, new_geo_dics):
    """create lables.npz and tiles from AOIs

    ~~~~
    Args:
        config: label maker config.json file.
        new_geo_dics (dict): dictionary contain geojson and its bbox
    Returns (none): a directory contains tiles and labels.npz from label maker
    """
    with open(config, 'r') as con_j:
        config_json = json.load(con_j)
    for key, value in new_geo_dics.items():
        print(key, value)
        geojson = key
        base_nm = geojson.split('.')[0]
        config_json['bounding_box'] = value
        config_json['geojson'] = f"geojsons/{geojson}"
        with open(config, 'w') as con_n_j:
            json.dump(config_json, con_n_j)
        print(config_json)
        cmd1 ="label-maker labels"
        os.system(cmd1)
        cmd2 ="label-maker images"
        os.system(cmd2)
        if not op.isdir(base_nm):
            makedirs(f'data/{base_nm}')
        shutil.move('data/labels.npz', f'data/{base_nm}')
        shutil.move('data/tiles', f'data/{base_nm}/tiles')

if __name__=="__main__":
    new_geo_dics={}
    # new_geo_dics['central_americal_p3.geojson'] = [-81.36164890009647,19.282763121285523,-81.3513101505485,19.299540491823304]
    # new_geo_dics['central_americal_p4.geojson'] = [-77.49591171624593,25.034085425966715,-77.45154078237647,25.060320759100083]
    # new_geo_dics['central_americal_p5.geojson'] = [-69.6856664772408,18.412247861357823,-69.66250376229063,18.45125465137286]
    # new_geo_dics['central_americal_p6.geojson'] = [-61.549291099979605,16.259461670443763,-61.502746502822866,16.277411454116898]
    # new_geo_dics['central_americal_p7.geojson'] = [-59.5146279879555,13.06557068900878,-59.46892258296427,13.083677647616769]
    # new_geo_dics['central_americal_p8.geojson'] = [-90.54141007131278,14.56095172754877,-86.83497978783647,15.756689140170963]
    # new_geo_dics['central_americal_p9.geojson'] = [-90.54141007131278,14.56095172754877,-90.51338895890112,14.597748935294478]
    new_geo_dics['asia_p1.geojson'] = [120.2505337230577,22.523855094567587,120.37511105074056,22.64947339043939]
    new_geo_dics['asia_p2.geojson'] = [100.58450427578242,13.6388562453059,100.78626023817888,13.945848799980968]
    new_geo_dics['asia_p3.geojson'] = [103.95941908193554,1.313448236662519,104.03132412723458,1.394274672234017]
    new_geo_dics['asia_p4.geojson'] = [98.84844186161708,3.619590902542954,98.90246283713661,3.662421634677365]
    new_geo_dics['asia_p5.geojson'] = [112.77148613317378,-7.39747942617161,112.81693397068814,-7.361404474085646]
    config = "config.json"
    create_training_data(config, new_geo_dics)
