"""
Use label maker (https://github.com/developmentseed/label-maker) 
to create training dataset.

author: @developmentseed @NASA-IMPACT

run:
    python3 label_maker_get_training_data_from_S3_IL.py
"""

import os, shutil, json, requests, subprocess, base64
from os import makedirs, path as op
import boto3
import numpy as np
import rasterio
import geojson
from glob import glob
from rasterio.io import MemoryFile
from rasterio.warp import reproject, calculate_default_transform, Resampling
from zipfile import ZipFile

ACCOUNT = os.environ['AWS_ACCOUNT_NUMBER']

DEFAULT_CRS = 'EPSG:4326'
DOWNLOAD_FOLDER = 'downloaded_files'

GEOJSON_FOLDER = 'geojsons'

S3_URL = f"s3://marine-litter-observations"
class Processor:
    def __init__(self, profile_name):
        """
        Initializer
        Args:
            profile_name (str): AWS account profile name
        """
        Processor.mkdir('updated')


    def create_training_data(self, config, new_geo_dics, basename, filename):
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
            geojson = key
            base_nm = basename 
            print("bounding_box: ", value)
            config_json['bounding_box'] = value
            config_json['geojson'] = geojson 
            config_json['imagery'] = filename+'.tif'
            with open(config, 'w') as con_n_j:
                json.dump(config_json, con_n_j)
                print(config_json)
            cmd1 ="label-maker labels"
            os.system(cmd1)
            cmd2 ="label-maker preview -n 10"
            os.system(cmd2)
            cmd3 ="label-maker images"
            os.system(cmd3)
            if not op.isdir(base_nm):
                makedirs(f'data/{base_nm}')
            shutil.move('data/labels.npz', f'data/{base_nm}')
            shutil.move('data/tiles', f'data/{base_nm}/tiles')
            
            tif_tiles = glob(f'data/{base_nm}/tiles'+'/**/*.tif',recursive=True)
            for t in tif_tiles:
                im = Image.open(t)
                im.save(i[:-4]+'.jpg', "JPEG")
                os.remove(t)

    def get_bounding_box(self, geoj):
        with open(geoj) as f:
            xcoords = []
            ycoords = []
            data = json.load(f)
            for feature in data['features']:
                geom = feature['geometry']
                for coord in geom['coordinates']:
                    if type(coord) == float:  # then its a point feature
                        xcoords.append(geom['coordinates'][0])
                        ycoords.append(geom['coordinates'][1])
                    elif type(coord) == list:
                        for c in coord:
                            if type(c) == float:  # then its a linestring feature
                                xcoords.append(coord[0])
                                ycoords.append(coord[1])
                            elif type(c) == list:  # then its a polygon feature
                                xcoords.append(c[0])
                                ycoords.append(c[1])
                coords = np.array(list(geojson.utils.coords(geom)))
        return [min(xcoords), min(ycoords), max(xcoords), max(ycoords)]

    def read_geotiffs(self, file_name):
        """
        Upload geotiffs into imagelabeler
        Args:
            file_name (str): path to downloaded geotiff.
        """
        foldername, _ = os.path.splitext(file_name)
        Uploader.mkdir(foldername)

        with ZipFile(file_name) as zip_file:
            print("================ Reading files ================")
            compressed_files = zip_file.namelist()
            for compressed_file in compressed_files:
                compressed_file = str(compressed_file)
                _, extension = os.path.splitext(compressed_file)
                if extension == '.tif':
                    self.process_geotiff(
                        compressed_file,
                        zip_file,
                        foldername
                    )

    def process_geotiff(self, compressed_file, zip_file, foldername):
        """
        Reproject and upload geotiff into imagelabeler
        Args:
            compressed_file (str): path of tif file in zip file
            zip_file (zipfile.ZipFile): zipfile instance
            foldername (str): foldername of where to store file
        """
        split = compressed_file.split('/')[-1].split('_')
        updated_filename = f"marine_plastic_{'T'.join(split[0:2])}_{'_'.join(split[2:])}"
        filename = f"{foldername}/{updated_filename}"
        mem_tiff = zip_file.read(compressed_file)
        tiff_file = MemoryFile(mem_tiff).open()
        updated_profile = self.calculate_updated_profile(tiff_file)
        with rasterio.open(updated_filename, 'w', **updated_profile) as dst:
            for band in range(1, 4):
                dest = reproject(
                    source=rasterio.band(tiff_file, band),
                    destination=rasterio.band(dst, band),
                    src_transform=tiff_file.transform,
                    src_crs=tiff_file.crs,
                    dst_transform=updated_profile['transform'],
                    dst_crs=DEFAULT_CRS,
                    resampling=Resampling.nearest
                )
        config = "config.json"
        new_geo_dics={}
        filename = filename[:-4]
        filename_split = os.path.splitext(filename) 
        filename_zero, fileext = filename_split 
        basename = os.path.basename(filename_zero) 
        basename = basename[15:]
        basename =basename.replace('_3B_Visual','')
        basename = basename[0:8]+'_'+basename[9:]
        print(f'geojson: {GEOJSON_FOLDER}/{basename}.geojson')
        extent = self.get_bounding_box(f'{GEOJSON_FOLDER}/{basename}.geojson')
        print(f"extent: {extent}")
        new_geo_dics[f'{GEOJSON_FOLDER}/{basename}.geojson'] = extent
        self.create_training_data(config, new_geo_dics, basename, filename)
        #os.remove(filename)
        print(f"{filename} processed")

    def calculate_updated_profile(self, tiff_file):
        """
        Create updated profile for the provided tiff_file
        Args:
            tiff_file (rasterio.io.MemoryFile): rasterio memoryfile.
        Returns:
            dict: updated profile for new tiff file
        """
        profile = tiff_file.profile
        transform, width, height = calculate_default_transform(
            tiff_file.crs,
            DEFAULT_CRS,
            tiff_file.width,
            tiff_file.height,
            *tiff_file.bounds
        )
        profile.update(
            crs=DEFAULT_CRS,
            transform=transform,
            width=width,
            height=height,
            count=3,
            nodata=0,
            compress='lzw',
            dtype='uint8'
        )
        return profile


    @classmethod
    def mkdir(cls, dirname):
        if not os.path.exists(dirname):
            os.mkdir(dirname)
            print(f'directory created: {dirname}')

def main(profile_name):
    session = boto3.session.Session(profile_name=profile_name)
    s3_connection = session.resource('s3')
    bucket = s3_connection.Bucket('marine-litter-observations')
    processor = Processor(profile_name)
    Processor.mkdir(DOWNLOAD_FOLDER)
    for s3_object in bucket.objects.all():
        if '.zip' in s3_object.key:
            filename = s3_object.key.split('/')[-1]
            print(f"================ Downloading file: {filename} ================")
            zip_filename = f"{DOWNLOAD_FOLDER}/{filename}"
            bucket.download_file(s3_object.key, zip_filename)
            print("================ Download complete ================ ")
            print("================ In progress ================")
            processor.read_geotiffs(zip_filename)
            print("================ Process Complete ================")
            #os.remove(f'{DOWNLOAD_FOLDER}/{filename}.tif')

main(profile_name=os.environ['AWS_PROFILE_NAME'])

