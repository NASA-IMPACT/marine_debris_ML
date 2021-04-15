"""
Script to batch process the raw analytic planet scenes into 3x NIR stacked images.
author: @developmentseed @NASA-IMPACT

run:
    python3 process_analytic_images_from_S3.py
"""

import os, sys
from os import makedirs, path as op
import shutil

from PIL import Image
import base64
import boto3
import json
import os
import traceback
import numpy as np
import rasterio
import requests
import subprocess
import geojson
from glob import glob
from rasterio.io import MemoryFile
from rasterio.warp import reproject, calculate_default_transform, Resampling
from zipfile import ZipFile

ACCOUNT = ''

DEFAULT_CRS = 'EPSG:4326'
DOWNLOAD_FOLDER = 'downloaded_files'


S3_URL = f"s3://marine-litter-observations"
class Processor:
    def __init__(self, profile_name):
        """
        Initializer
        Args:
            profile_name (str): AWS account profile name
        """
        Processor.mkdir('updated')


    def read_geotiffs(self, file_name):
        """
        Upload geotiffs into imagelabeler
        Args:
            file_name (str): path to downloaded geotiff.
        """
        foldername, _ = os.path.splitext(file_name)
        Processor.mkdir(foldername)

        with ZipFile(file_name) as zip_file:
            print("================ Reading files ================")
            compressed_files = zip_file.namelist()
            print("compressed_files count: ", len(compressed_files))
            
            for compressed_file in compressed_files:
                compressed_file = str(compressed_file)
                _, extension = os.path.splitext(compressed_file)
                if extension == '.tif':
                    if '_DN_udm' not in compressed_file:
                        print("compressed_file: ", compressed_file)
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
        
        tiff_file_r = tiff_file.read([4])
        tiff_file_r = tiff_file_r #.astype('uint8')
        
        tiff_file_t = tiff_file_r.transpose(1,2,0)

        tiff_file_stack = np.dstack((tiff_file_t, tiff_file_t, tiff_file_t))
        tiff_file_stack = tiff_file_stack.transpose(2,0,1)
    

        updated_profile = self.calculate_updated_profile(tiff_file)
        with rasterio.open(updated_filename, 'w', **updated_profile) as dst:
            
            for i, band in enumerate(tiff_file_r, 1):
                dest = np.zeros_like(band)
                
                reproject(
                    band,
                    dest,
                    src_transform=tiff_file.transform,
                    src_crs=tiff_file.crs,
                    dst_transform=updated_profile['transform'],
                    dst_crs=DEFAULT_CRS,
                    resampling=Resampling.nearest)

                dst.write(dest, indexes=i)
            
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
            dtype= 'uint16' #'uint8'
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
            #filename = s3_object.key.split('/')[-1]
            filename = 'marine_litter_order_Mar-31-2021.zip'
            print(f"================ Downloading file: {filename} ================")
            #zip_filename = f"{DOWNLOAD_FOLDER}/{filename}"
            #bucket.download_file(s3_object.key, zip_filename)
            print("================ Download complete ================ ")
            print("================ Process in progress ================")
            #processor.read_geotiffs(zip_filename)
            
            try: 
                processor.read_geotiffs(zip_filename)
            except:
                traceback.print_exc()
                pass

            
            print("================ Process Complete ================")



main(profile_name='nasa')

