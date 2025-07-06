from fastapi import APIRouter, UploadFile as Upload, File, HTTPException
from pydantic import BaseModel
import requests
from dotenv import load_dotenv
load_dotenv()
import os

MAPBOX_ACCESS_TOKEN = os.getenv("mapbox_key")

def generate_bbox(lat:float, lng:float, delta=10):
    # ~10km quanh vị trí người dùng
    min_lng = lng - delta
    min_lat = lat - delta
    max_lng = lng + delta
    max_lat = lat + delta
    return f"{min_lng},{min_lat},{max_lng},{max_lat}"

async def search_pharmacies(lat:float,lng:float):
    bbox = generate_bbox(lat, lng)
    url = "https://api.mapbox.com/geocoding/v5/mapbox.places/pharmacy.json"
    params = {
        "proximity": f"{lng},{lat}",
        "bbox": bbox,
        "country": "VN",  # Giới hạn trong Việt Nam
        "access_token": MAPBOX_ACCESS_TOKEN,
        "limit": 10
    }

    response = requests.get(url, params=params)
    data = response.json()

    results = []
    for feature in data.get("features", []):
        results.append({
            "name": feature["text"],
            "address": feature.get("place_name", ""),
            "lat": feature["center"][1],
            "lng": feature["center"][0],
        })

    return results