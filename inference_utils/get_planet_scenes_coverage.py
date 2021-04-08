"""
Script to get all the scene ids for Planetscopes
"""
import os
import requests
from requests.auth import HTTPBasicAuth
import numpy as np
import json
import requests
import mercantile
import numpy as np


def get_sites(url):
    geojson = requests.get(url).json()

    features = [
        {
            'type': 'Feature',
            'geometry': site["polygon"],
            'bbox': site["bounding_box"],
            'properties': {"id": site["id"], "label": site["label"]}
        }
        for site in geojson["sites"]
    ]

    geojson = {
        "type": "FeatureCollection",
        "features": features}
    return geojson

def stats(geometry, collections, start_date, end_date, cc, PL_API_KEY):
    """Retrieve Stats

    ----
    Args:
        collections: ['PSOrthoTile', 'REOrthoTile', 'PSScene3Band', 'PSScene4Band', 'SkySatScene']
        geometry: geojson for the sites
        start_date: "2020-04-01T00:00:00.000Z"
        end_date: same format as start_date
        cc: cloud cover in 0.05 (5%)
    """
    # filter for items the overlap with our chosen geometry
    geometry_filter = {
      "type": "GeometryFilter",
      "field_name": "geometry",
      "config": geometry
    }

    # filter images acquired in a certain date range
    date_range_filter = {
      "type": "DateRangeFilter",
      "field_name": "acquired",
      "config": {
        "gte": start_date,
        "lte": end_date
      }
    }

    # filter any images which are more than 50% clouds
    cloud_cover_filter = {
      "type": "RangeFilter",
      "field_name": "cloud_cover",
      "config": {
        "lte": cc
      }
    }

    config = {
      "type": "AndFilter",
      "config": [geometry_filter, cloud_cover_filter, date_range_filter]
    }

    # Stats API request object
    stats_endpoint_request = {
      "interval": "day", "item_types": collections, "filter": config
    }

    # fire off the POST request
    result = requests.post('https://api.planet.com/data/v1/stats', auth=HTTPBasicAuth(PL_API_KEY, ''), json=stats_endpoint_request)
    return result.json()


def search(geometry, collections, start_date, end_date, cc, PL_API_KEY
):
    """Search for Data."""
    # filter for items the overlap with our chosen geometry
    geometry_filter = {
      "type": "GeometryFilter",
      "field_name": "geometry",
      "config": geometry
    }

    # filter images acquired in a certain date range
    date_range_filter = {
      "type": "DateRangeFilter",
      "field_name": "acquired",
      "config": {
        "gte": start_date,
        "lte":end_date
      }
    }

    # filter any images which are more than 50% clouds
    cloud_cover_filter = {
      "type": "RangeFilter",
      "field_name": "cloud_cover",
      "config": {
        "lte": cc
      }
    }

    config = {
      "type": "AndFilter",
      "config": [geometry_filter, cloud_cover_filter, date_range_filter]
    }

    # Stats API request object
    stats_endpoint_request = {
      "interval": "day",
      "item_types": collections,
      "filter": config
    }

    # fire off the POST request
    result = requests.post('https://api.planet.com/data/v1/quick-search', auth=HTTPBasicAuth(PL_API_KEY, ''), json=stats_endpoint_request)
    return result.json()


def get_scene_ids_aoi(site_aois, start_date, end_date, cc, collections, api_key):
    results = {feat["properties"]["label"] : stats(feat["geometry"], collections, start_date, end_date, cc, api_key)
           for feat in site_aois["features"]}
    results_PS = {feat["properties"]["label"] : search(feat["geometry"], collections, start_date, end_date, cc, api_key)
                  for feat in site_aois["features"]}

    aois_scene_ids = [{
        aoi: [[f['id'], f['geometry']['coordinates']]for f in results_PS[aoi]['features']]
    } for aoi in results_PS.keys()]
    print(f"Total of {collections} scenes per sites")
    for k, r in results.items():
        print(k)
        im = r.get("buckets", [])
        total = sum([f["count"] for f in im])
        print(total)
        print("---")

    return aois_scene_ids



api_key = os.getenv("PL_API_KEY")
aoi_url = "https://8ib71h0627.execute-api.us-east-1.amazonaws.com/v1/sites"
geojson = get_sites(aoi_url)

collections=["PSScene3Band"]

start_d = "2020-04-01T00:00:00.000Z"
end_d = "2020-04-01T00:00:00.000Z"
cloud_cover = 0.05
scene_ids_aois = get_scene_ids_aoi(geojson, start_d, end_d, cloud_cover, collections, api_key)
