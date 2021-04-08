import base64
import boto3
import json
import os
import rasterio
import requests
import subprocess
from glob import glob
from rasterio.io import MemoryFile
from rasterio.warp import reproject, calculate_default_transform, Resampling
from zipfile import ZipFile

ACCOUNT = os.environ['AWS_ACCOUNT_NUMBER']
BASE_URL = "https://labeler.nasa-impact.net"
DEFAULT_CRS = 'EPSG:4326'
DOWNLOAD_FOLDER = 'downloaded_files'
LOGIN_URL = f"{BASE_URL}/accounts/login/"
IL_URL = {
    'geotiff': f"{BASE_URL}/api/geotiffs"
}
S3_URL = f"s3://marine-litter-observations"
class Uploader:
    def __init__(self, username, password, client_id, client_secret):
        """
        Initializer
        Args:
            username (str): ImageLabeler Username
            password (str): ImageLabeler Password
        """
        self.request_token(username, password, client_id, client_secret)
        Uploader.mkdir('updated')
    def upload_geotiffs(self, file_name):
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
        with rasterio.open(filename, 'w', **updated_profile) as dst:
            for band in range(1, 4):
                reproject(
                    source=rasterio.band(tiff_file, band),
                    destination=rasterio.band(dst, band),
                    src_transform=tiff_file.transform,
                    src_crs=tiff_file.crs,
                    dst_transform=updated_profile['transform'],
                    dst_crs=DEFAULT_CRS,
                    resampling=Resampling.nearest
                )
        _, status_code = self.upload_to_image_labeler(filename)
        if status_code == 200:
            os.remove(filename)
        print(f"{filename} uploaded to imagelabeler with: {status_code}")
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
    def request_token(self, username, password, client_id, client_secret):
        """
        this funtion will return an authentication token for users to use
        Args:
            username (string) : registered username of the user using the script
            password (string) : password associated with the user
        Exceptions:
            UserNotFound: Given user does not exist
        Returns:
            headers (dict): {
                "Authorization": "Bearer ..."
            }
        """
        payload = {
            "username": username,
            "password": password,
            "grant_type": "password"
        }
        response = requests.post(
            f"{BASE_URL}/authentication/token/",
            data=payload,
            auth=(client_id, client_secret)
        )
        access_token = json.loads(response.text)['access_token']
        self.headers = {
            'Authorization': f"Bearer {access_token}",
        }
    def upload_to_image_labeler(self, file_name, file_type='geotiff'):
        """
        Uploads a single shapefile to the image labeler
        Args:
            file_name : name of zip file containing shapefiles
        Returns:
            response (tuple[string]): response text, response code
        """
        with open(file_name, 'rb') as upload_file_name:
            file_headers = {
                **self.headers,
            }
            files = {
                'file': (file_name, upload_file_name),
            }
            response = requests.post(
                IL_URL[file_type],
                files=files,
                headers=file_headers
            )
            return response.text, response.status_code
    @classmethod
    def mkdir(cls, dirname):
        if not os.path.exists(dirname):
            os.mkdir(dirname)
            print(f'directory created: {dirname}')
def main(profile_name, username, password, client_id, client_secret):
    session = boto3.session.Session(profile_name=profile_name)
    s3_connection = session.resource('s3')
    bucket = s3_connection.Bucket('marine-litter-observations')
    uploader = Uploader(username, password, client_id, client_secret)
    Uploader.mkdir(DOWNLOAD_FOLDER)
    for s3_object in bucket.objects.all():
        if '.zip' in s3_object.key:
            filename = s3_object.key.split('/')[-1]
            print(f"================ Downloading file: {filename} ================")
            zip_filename = f"{DOWNLOAD_FOLDER}/{filename}"
            bucket.download_file(s3_object.key, zip_filename)
            print("================ Download complete ================ ")
            print("================ Upload in progress ================")
            uploader.upload_geotiffs(zip_filename)
            print("================ Upload Complete ================")


main(profile_name=os.environ['AWS_PROFILE_NAME'], client_id=os.environ['AWS_ACCESS_KEY'], client_secret=os.environ['AWS_SECRET_ACCESS_KEY'])

