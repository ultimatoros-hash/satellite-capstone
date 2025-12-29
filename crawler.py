import os
import math
import csv
import requests
import concurrent.futures
from PIL import Image
from io import BytesIO
import random
import time

# --- CONFIGURATION ---
OUTPUT_DIR = "data/raw/images"
CSV_FILE = "data/dataset.csv"
ZOOM_LEVEL = 15   
MAX_WORKERS = 12  # Moderate speed to prevent server blocks
STEP_SIZE = 0.0045 # Tuned for ~20k total images

# --- BROWSER MASQUERADE (Prevents 403 Forbidden) ---
HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8',
    'Referer': 'https://www.google.com/'
}

# --- TILE PROVIDERS ---
PROVIDERS = {
    "esri": "https://services.arcgisonline.com/arcgis/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    "usgs": "https://basemap.nationalmap.gov/arcgis/rest/services/USGSImageryOnly/MapServer/tile/{z}/{y}/{x}",
    "nasa": "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/BlueMarble_ShadedRelief_Bathymetry/default/2022-12-01/GoogleMapsCompatible_Level8/{z}/{y}/{x}.jpg"
}

# --- REGIONS OF INTEREST ---
TARGETS = [
    # --- URBAN (High Density Cities) ---
    ("urban", 40.7000, -74.0200, 40.8000, -73.9500, "esri"),   # NYC
    ("urban", 35.6800, 139.7000, 35.7500, 139.8000, "esri"),   # Tokyo
    ("urban", 48.8300, 2.2500, 48.9000, 2.4000, "esri"),       # Paris
    ("urban", 51.4500, -0.1500, 51.5500, 0.0500, "esri"),      # London
    ("urban", 19.4000, -99.1500, 19.5000, -99.1000, "esri"),   # Mexico City
    ("urban", 31.2000, 121.4000, 31.3000, 121.5000, "esri"),   # Shanghai
    ("urban", -33.8600, 151.2000, -33.8000, 151.3000, "esri"), # Sydney

    # --- FOREST (Massive Green Zones) ---
    ("forest", -3.4653, -62.2159, -3.4000, -62.1000, "esri"),  # Amazon 1
    ("forest", -4.0000, -65.0000, -3.9000, -64.9000, "esri"),  # Amazon 2
    ("forest", 48.4647, 7.9552, 48.5500, 8.0500, "esri"),      # Black Forest
    ("forest", 45.0000, -122.0000, 45.1000, -121.8000, "usgs"), # Oregon
    ("forest", 0.3000, 20.0000, 0.4000, 20.1000, "esri"),      # Congo Basin
    ("forest", 61.0000, 99.0000, 61.1000, 99.1000, "esri"),    # Taiga

    # --- WATER (Oceans & Lakes) ---
    ("water", 25.0343, -77.3963, 25.1000, -77.3000, "esri"),   # Bahamas
    ("water", 34.0000, -119.0000, 34.1000, -118.9000, "nasa"), # Pacific (NASA)
    ("water", -20.0000, 57.5000, -19.9000, 57.6000, "esri"),   # Mauritius
    ("water", 43.0000, 6.0000, 43.1000, 6.1000, "nasa"),       # Med Sea (NASA)
    ("water", 47.7000, -87.5000, 47.8000, -87.4000, "usgs"),   # Lake Superior

    # --- DESERT (Sand & Rock) ---
    ("desert", 24.6857, 46.7023, 24.8000, 46.8000, "esri"),    # Saudi
    ("desert", 36.1699, -115.1398, 36.3000, -115.0000, "usgs"), # Nevada
    ("desert", 21.0000, 10.0000, 21.1000, 10.1000, "nasa"),    # Sahara (NASA)
    ("desert", -25.3000, 131.0000, -25.2000, 131.1000, "esri"), # Outback
    ("desert", 40.0000, 90.0000, 40.1000, 90.1000, "nasa"),    # Gobi (NASA)
]

def lat_lon_to_tile(lat, lon, zoom):
    lat_rad = math.radians(lat)
    n = 2.0 ** zoom
    xtile = int((lon + 180.0) / 360.0 * n)
    ytile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return xtile, ytile

def generate_grid(lat_min, lon_min, lat_max, lon_max, step):
    points = []
    lat = lat_min
    while lat < lat_max:
        lon = lon_min
        while lon < lon_max:
            points.append((lat, lon))
            lon += step
        lat += step
    return points

def download_task(args):
    lat, lon, label, provider_name = args
    xtile, ytile = lat_lon_to_tile(lat, lon, ZOOM_LEVEL)
    
    url_template = PROVIDERS.get(provider_name, PROVIDERS['esri'])
    url = url_template.format(z=ZOOM_LEVEL, y=ytile, x=xtile)
    
    filename = f"{label}_{provider_name}_{xtile}_{ytile}.jpg"
    save_dir = os.path.join(OUTPUT_DIR, label)
    full_path = os.path.join(save_dir, filename)
    
    # Check if exists (and valid size)
    if os.path.exists(full_path) and os.path.getsize(full_path) > 1000:
        return None

    try:
        # Headers are CRITICAL to prevent 403 Forbidden errors
        r = requests.get(url, headers=HEADERS, timeout=5)
        
        if r.status_code == 200:
            # Verify image integrity
            img = Image.open(BytesIO(r.content))
            img.verify()
            
            # Re-open to save
            img = Image.open(BytesIO(r.content))
            os.makedirs(save_dir, exist_ok=True)
            img.convert('RGB').save(full_path)
            return [filename, label, lat, lon, provider_name, url]
    except Exception:
        pass # Skip bad tiles silently to keep speed up
    return None

if __name__ == "__main__":
    # --- RESET OLD DATA ---
    print("🚀 Initializing Crawler...")
    if not os.path.exists(CSV_FILE):
        os.makedirs(os.path.dirname(CSV_FILE), exist_ok=True)
        with open(CSV_FILE, 'w', newline='') as f:
            csv.writer(f).writerow(["filename", "label", "latitude", "longitude", "provider", "url"])

    tasks = []
    for label, lat_min, lon_min, lat_max, lon_max, provider in TARGETS:
        points = generate_grid(lat_min, lon_min, lat_max, lon_max, STEP_SIZE)
        for lat, lon in points:
            tasks.append((lat, lon, label, provider))
            
    print(f"📡 Target List: {len(tasks)} potential tiles.")
    print("⏳ Starting download (this will take 5-10 minutes)...")
    
    count = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = executor.map(download_task, tasks)
        
        with open(CSV_FILE, 'a', newline='') as f:
            writer = csv.writer(f)
            for res in results:
                if res:
                    writer.writerow(res)
                    count += 1
                    if count % 200 == 0: 
                        print(f"  [+] Saved {count} images...")

    print(f"\n✅ Crawl Complete. Total Images Saved: {count}")
