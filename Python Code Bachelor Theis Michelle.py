#!/usr/bin/env python
# coding: utf-8

# # Python Code Bachelor Thesis

# ### Michelle Köthe

# ##### The structure and syntax of this code were generated with the aid of ChatGPT-4o

# ### Inspect data

# In[49]:


import rasterio
import numpy as np

# Path to GLCLU
path = "/Users/michelle/Desktop/BA/LCLU/20S_040E_2000.tif"

# Open and read raster
with rasterio.open(path) as src:
    data = src.read(1)
    crs = src.crs
    transform = src.transform
    res = src.res
    nodata = src.nodata

# Get unique values (classes)
unique_values = np.unique(data)

print("Unique land cover/use classes in the raster:")
print(unique_values)

# Print some raster info
print("\n Resolution:", res)
print("CRS:", crs)
print("Transform (origin, pixel size):", transform)
print("Nodata value:", nodata)
print("Shape (rows, cols):", data.shape)


# ### Merge same year together

# In[50]:


from rasterio.merge import merge
from glob import glob
import os

# Input folder
input_dir = "/Users/michelle/Desktop/BA/LCLU/"

# Output folder
output_dir = "/Users/michelle/Desktop/BA/LCLU/merged/"
os.makedirs(output_dir, exist_ok=True)

# Target years
years = ["2000", "2005", "2010", "2015", "2020"]

for year in years:
    print(f"Merging tiles for year {year}...")

    # Find all files for this year
    search_pattern = os.path.join(input_dir, f"*_{year}.tif")
    files = sorted(glob(search_pattern))

    if not files:
        print(f"No files found for year {year}")
        continue

# Open all rasters
    src_files_to_mosaic = [rasterio.open(fp) for fp in files]

# Merge tiles
    mosaic, out_transform = merge(src_files_to_mosaic)

# Copy metadata and update
    out_meta = src_files_to_mosaic[0].meta.copy()
    out_meta.update({
        "driver": "GTiff",
        "height": mosaic.shape[1],
        "width": mosaic.shape[2],
        "transform": out_transform
    })

 # Save the merged raster
    out_path = os.path.join(output_dir, f"LCLU_{year}_merged.tif")
    with rasterio.open(out_path, "w", **out_meta) as dest:
        dest.write(mosaic)

    print(f"Saved: {out_path}")


# ### Clip merged layers to Madagascar

# In[51]:


from rasterio.mask import mask
import geopandas as gpd
from glob import glob

# Paths
merged_dir = "/Users/michelle/Desktop/BA/LCLU/merged/"
output_dir = "/Users/michelle/Desktop/BA/LCLU/clipped/"
os.makedirs(output_dir, exist_ok=True)

shapefile_path = "/Users/michelle/Desktop/BA/Boundaries/gadm41_MDG_shp/gadm41_MDG_0.shp"
gdf = gpd.read_file(shapefile_path)
geometry = gdf.geometry.values

# Loop through merged rasters and clip
merged_files = sorted(glob(os.path.join(merged_dir, "LCLU_*_merged.tif")))

for raster_path in merged_files:
    print(f"Clipping {os.path.basename(raster_path)}...")

    with rasterio.open(raster_path) as src:
        out_image, out_transform = mask(src, geometry, crop=True)
        out_meta = src.meta.copy()

        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })

        # Output path
        filename = os.path.basename(raster_path).replace("_merged", "_clipped")
        output_path = os.path.join(output_dir, filename)

        with rasterio.open(output_path, "w", **out_meta) as dest:
            dest.write(out_image)

        print(f"Saved: {output_path}")


# ### Classify forest for year 2000

# In[52]:


# Input and output
input_raster = "/Users/michelle/Desktop/BA/LCLU/clipped/LCLU_2000_clipped.tif"
output_raster = "/Users/michelle/Desktop/BA/LCLU/forest_mask_2000_correct.tif"

# Forest class codes
forest_classes = list(range(27, 49))  #minimum height 5m

with rasterio.open(input_raster) as src:
    lc = src.read(1)
    meta = src.meta.copy()

    # Create forest mask: 1 = forest, 0 = non-forest
    forest_mask = np.isin(lc, forest_classes).astype(np.uint8)

    meta.update(dtype="uint8", count=1, compress="lzw")

    with rasterio.open(output_raster, "w", **meta) as dst:
        dst.write(forest_mask, 1)

print(f"Forest mask saved to: {output_raster}")


# In[6]:


# Reproject
from rasterio.warp import calculate_default_transform, reproject, Resampling

# Input/output
src_path = "/Users/michelle/Desktop/BA/LCLU/forest_mask_2000_correct.tif"
dst_path = "/Users/michelle/Desktop/BA/LCLU/forest_mask_2000_utm.tif"
dst_crs = "EPSG:32738"  # UTM Zone 38S (Madagascar)

with rasterio.open(src_path) as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)

    kwargs = src.meta.copy()
    kwargs.update({
        "crs": dst_crs,
        "transform": transform,
        "width": width,
        "height": height
    })

    with rasterio.open(dst_path, "w", **kwargs) as dst:
        reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest)

print(f"Reprojected raster saved to:\n{dst_path}")


# In[7]:


# Calculate area

# Path to reprojected raster
raster_path = "/Users/michelle/Desktop/BA/LCLU/forest_mask_2000_utm.tif"

with rasterio.open(raster_path) as src:
    forest = src.read(1)
    pixel_size_x, pixel_size_y = src.res  # should be in meters now
    pixel_area_m2 = pixel_size_x * pixel_size_y
    pixel_area_km2 = pixel_area_m2 / 1_000_000  # m² → km²

# Count forest pixels (value == 1)
    forest_pixels = np.sum(forest == 1)

# Total area
    total_area_km2 = forest_pixels * pixel_area_km2
    total_area_ha = total_area_km2 * 100

print(f"Forest pixels: {forest_pixels:,}")
print(f"Forest area: {total_area_ha:,.2f} ha ({total_area_km2:,.2f} km²)")


# ### Classify forest for all the years

# In[8]:


import geopandas as gpd
from rasterio.mask import mask
from rasterio.warp import calculate_default_transform, reproject, Resampling
import pandas as pd

# Paths
base_dir = "/Users/michelle/Desktop/BA/LCLU/"
input_dir = os.path.join(base_dir, "clipped/")
output_dir = os.path.join(base_dir, "forest_masks/")
os.makedirs(output_dir, exist_ok=True)

boundary_path = "/Users/michelle/Desktop/BA/Boundaries/gadm41_MDG_shp/gadm41_MDG_0.shp"
gdf = gpd.read_file(boundary_path).to_crs("EPSG:32738")
geometry = gdf.geometry.values

# Forest class codes
forest_classes = list(range(27, 49)) 

# Years to process
years = ["2000", "2005", "2010", "2015", "2020"]
dst_crs = "EPSG:32738"

results = []

for year in years:
    input_path = os.path.join(input_dir, f"LCLU_{year}_clipped.tif")
    forest_mask_path = os.path.join(output_dir, f"forest_mask_{year}.tif")
    forest_mask_utm_path = os.path.join(output_dir, f"forest_mask_{year}_utm.tif")

    if not os.path.exists(input_path):
        print(f"Land cover file missing: {input_path}")
        continue

# Load LC & classify forest
    with rasterio.open(input_path) as src:
        lc = src.read(1)
        meta = src.meta.copy()
        forest_mask = np.isin(lc, forest_classes).astype(np.uint8)

# Save forest mask
    meta.update(dtype="uint8", count=1, compress="lzw")
    with rasterio.open(forest_mask_path, "w", **meta) as dst:
        dst.write(forest_mask, 1)

# Reproject to UTM
    with rasterio.open(forest_mask_path) as src:
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)

        kwargs = src.meta.copy()
        kwargs.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height
        })

        with rasterio.open(forest_mask_utm_path, "w", **kwargs) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=dst_crs,
                resampling=Resampling.nearest)

# Clip to Madagascar
    with rasterio.open(forest_mask_utm_path) as src:
        out_image, out_transform = mask(src, geometry, crop=True)
        out_meta = src.meta.copy()
        out_meta.update({
            "height": out_image.shape[1],
            "width": out_image.shape[2],
            "transform": out_transform
        })
        with rasterio.open(forest_mask_utm_path, "w", **out_meta) as dst:
            dst.write(out_image)

# Calculate area
    with rasterio.open(forest_mask_utm_path) as src:
        forest = src.read(1)
        pixel_area_m2 = src.res[0] * src.res[1]
        forest_pixels = np.sum(forest == 1)
        area_km2 = forest_pixels * pixel_area_m2 / 1_000_000
        area_ha = area_km2 * 100

    results.append({
        "year": int(year),
        "forest_pixels": forest_pixels,
        "area_ha": round(area_ha, 2),
        "area_km2": round(area_km2, 2)
    })

    print(f" {year}: {forest_pixels:,} px → {area_ha:,.2f} ha")

# Summary table
df = pd.DataFrame(results)
df_path = os.path.join(output_dir, "forest_area_summary.csv")
df.to_csv(df_path, index=False)

print("\n Summary saved to:", df_path)
print(df)


# ### Create a 5 Digits Trajectory Map 

# In[9]:


# Settings
input_dir = "/Users/michelle/Desktop/BA/LCLU/clipped/" 
output_path = "/Users/michelle/Desktop/BA/LCLU/forest_masks/forest_trajectory_year_based.tif"  
forest_classes = list(range(27, 49))   
years = [2000, 2005, 2010, 2015, 2020]  

# Load land cover rasters and classify forest (1) / non-forest (2) 
forest_status = []
meta_ref = None

for year in years:
    path = os.path.join(input_dir, f"LCLU_{year}_clipped.tif")  # Use the existing clipped land cover rasters
    if not os.path.exists(path):
        raise FileNotFoundError(f"File missing: {path}")

    with rasterio.open(path) as src:
        lc = src.read(1)  # Read land cover data (1-band)

        # Classify forest (1 for forest) and non-forest (2 for non-forest)
        forest = np.where(np.isin(lc, forest_classes), 1, 2).astype(np.uint8)
        forest_status.append(forest)
        if meta_ref is None:
            meta_ref = src.meta.copy() 

#  Build 5-digit trajectory code 
trajectory_code = (
    forest_status[0].astype(np.uint32) * 10000 + 
    forest_status[1].astype(np.uint32) * 1000 + 
    forest_status[2].astype(np.uint32) * 100 + 
    forest_status[3].astype(np.uint32) * 10 + 
    forest_status[4].astype(np.uint32)
)

# Save trajectory raster
meta_ref.update(dtype="uint32", count=1, compress="lzw", nodata=99999)  # Set dtype and nodata value
with rasterio.open(output_path, "w", **meta_ref) as dst:
    dst.write(trajectory_code, 1)  # Write the trajectory code to the output raster

print(f"Forest trajectory (year-based, 5-digit) saved to:\n{output_path}")


# ### Reclassify Trajectories into 5 Groups

# In[10]:


# Paths
input_raster = "/Users/michelle/Desktop/BA/LCLU/forest_masks/forest_trajectory_year_based.tif"
output_raster = "/Users/michelle/Desktop/BA/LCLU/forest_masks/trajectory_grouped_8class.tif"

# Class mapping
group_map = {
    # 1 = Stable forest
    11111: 1,

    # 2 = Stable non-forest
    22222: 2,

    # 3 = Early deforestation
    11222: 3, 12222: 3,

    # 4 = Late deforestation
    11112: 4, 11122: 4,

    # 5 = Early afforestation
    21111: 5, 22111: 5,

    # 6 = Late afforestation
    22211: 6, 22221: 6,

    # 7 = Forest recovery
    11211: 7, 12111: 7, 12211: 7,

    # 8 = Temporary (rest)
}

# Read raster 
with rasterio.open(input_raster) as src:
    profile = src.profile
    data = src.read(1)
    nodata = src.nodata if src.nodata is not None else -9999
    transform = src.transform

# Reclassify
grouped = np.full_like(data, 8, dtype=np.int32)  # default = 8 (Temporary)

# Apply mapping
for original_value, grouped_value in group_map.items():
    grouped[data == original_value] = grouped_value

# Restore nodata values
grouped[data == nodata] = nodata

# Save
profile.update(dtype=rasterio.int32, nodata=nodata, compress='lzw')

with rasterio.open(output_raster, "w", **profile) as dst:
    dst.write(grouped, 1)

print("Raster saved with 8 trajectory classes to:", output_raster)


# ### Clip to Madagascar 

# In[11]:


from rasterio.mask import mask

# File paths
input_raster = "/Users/michelle/Desktop/BA/LCLU/forest_masks/trajectory_grouped_8class.tif"
output_clipped = "/Users/michelle/Desktop/BA/LCLU/forest_masks/trajectory_grouped_MAD.tif"
boundary_vector =  "/Users/michelle/Desktop/BA/Boundaries/gadm41_MDG_shp/gadm41_MDG_0.shp"

# Load Madagascar boundary
madagascar = gpd.read_file(boundary_vector)

# Reproject to match raster CRS
with rasterio.open(input_raster) as src:
    madagascar = madagascar.to_crs(src.crs)

    # Get geometry in GeoJSON format
    geoms = [feature["geometry"] for feature in madagascar.__geo_interface__["features"]]

    # Clip raster 
    out_image, out_transform = mask(src, geoms, crop=True)
    out_meta = src.meta.copy()

# Update metadata and save
out_meta.update({
    "driver": "GTiff",
    "height": out_image.shape[1],
    "width": out_image.shape[2],
    "transform": out_transform
})

with rasterio.open(output_clipped, "w", **out_meta) as dest:
    dest.write(out_image)

print("Raster successfully clipped to Madagascar and saved to:")
print(output_clipped)



# ### Statistics for trajectory map

# In[12]:


from rasterio.warp import calculate_default_transform, reproject, Resampling


# Input/output paths
input_raster = "/Users/michelle/Desktop/BA/LCLU/forest_masks/trajectory_grouped_MAD.tif"
reprojected_raster = "/Users/michelle/Desktop/BA/LCLU/forest_masks/trajectory_grouped_MAD_UTM.tif"

# Reproject to EPSG:32738 (UTM zone 38S)
dst_crs = "EPSG:32738"  

with rasterio.open(input_raster) as src:
    transform, width, height = calculate_default_transform(
        src.crs, dst_crs, src.width, src.height, *src.bounds)

    kwargs = src.meta.copy()
    kwargs.update({
        'crs': dst_crs,
        'transform': transform,
        'width': width,
        'height': height,
        'compress': 'lzw'
    })

    with rasterio.open(reprojected_raster, 'w', **kwargs) as dst:
        reproject(
            source=rasterio.band(src, 1),
            destination=rasterio.band(dst, 1),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=transform,
            dst_crs=dst_crs,
            resampling=Resampling.nearest)

print("Raster successfully reprojected to EPSG:32738:")
print(reprojected_raster)


# In[13]:


# Path to reprojected raster
reprojected_raster = "/Users/michelle/Desktop/BA/LCLU/forest_masks/trajectory_grouped_MAD_UTM.tif"

# Labels for classes
labels = {
    1: "Stable forest",
    2: "Stable non-forest",
    3: "Early deforestation",
    4: "Late deforestation",
    5: "Early afforestation",
    6: "Late afforestation",
    7: "Forest recovery",
    8: "Temporary changes"
}

# Open raster and calculate pixel area
with rasterio.open(reprojected_raster) as src:
    data = src.read(1)
    nodata = src.nodata
    pixel_area_km2 = src.res[0] * src.res[1] / 1_000_000  # from m² to km²

# Mask out nodata values
valid_data = data[data != nodata]
unique, counts = np.unique(valid_data, return_counts=True)

# Total area for % calculation
total_area_km2 = sum(count * pixel_area_km2 for count in counts)

# Print header
print("Forest trajectory statistics for Madagascar (EPSG:32738)\n")
print(f"{'Class':<6} {'Label':<22} {'Pixels':>12} {'Area (km²)':>14} {'% of Total':>12}")

# Loop over each class
for cls in range(1, 9):
    label = labels.get(cls, "Unknown")
    if cls in unique:
        idx = unique.tolist().index(cls)
        pixel_count = counts[idx]
        area_km2 = pixel_count * pixel_area_km2
        percentage = (area_km2 / total_area_km2) * 100
    else:
        pixel_count = 0
        area_km2 = 0
        percentage = 0
    print(f"{cls:<6} {label:<22} {pixel_count:>12,} {area_km2:>14,.2f} {percentage:>11.2f}%")

print(f"\nTotal area: {total_area_km2:,.2f} km²")


# ### Create Binary Mask Deforestation and afforestation

# In[12]:


# File paths
input_trajectory_path = "/Users/michelle/Desktop/BA/LCLU/forest_masks/trajectory_grouped_8class.tif"
output_deforestation_path = "/Users/michelle/Desktop/BA/LCLU/forest_masks/deforestation_mask_2000_2020_c.tif"
output_afforestation_path = "/Users/michelle/Desktop/BA/LCLU/forest_masks/afforestation_mask_2000_2020_c.tif"

# Load trajectory raster
with rasterio.open(input_trajectory_path) as src:
    trajectory = src.read(1)  # Reading the trajectory raster

# Create binary mask for deforestation (late + early)
# Deforestation codes: 3, 4
deforestation_mask = np.isin(trajectory, [3, 4]).astype(np.uint8)

# Create binary mask for afforestation (late + early)
# Afforestation codes: 5, 6
afforestation_mask = np.isin(trajectory, [5, 6]).astype(np.uint8)

# Save the deforestation mask
meta_ref = src.meta.copy()  # Copy the metadata for the output raster
meta_ref.update(dtype="uint8", count=1, compress="lzw", nodata=0)

with rasterio.open(output_deforestation_path, "w", **meta_ref) as dst:
    dst.write(deforestation_mask, 1)

print(f"Deforestation mask saved to: {output_deforestation_path}")

# Save the afforestation mask
with rasterio.open(output_afforestation_path, "w", **meta_ref) as dst:
    dst.write(afforestation_mask, 1)

print(f"Afforestation mask saved to: {output_afforestation_path}")


# ### Delta Map for BII

# In[5]:


from rasterio.plot import show
from matplotlib.colors import ListedColormap
import numpy as np
import rasterio
import matplotlib.pyplot as plt

# Paths
bii_2000_path = "/Users/michelle/Desktop/BA/BII/bii-2000_v2-1-1_madagascar.tif"
bii_2020_path = "/Users/michelle/Desktop/BA/BII/bii-2020_v2-1-1_madagascar.tif"
bii_delta_path = "/Users/michelle/Desktop/BA/BII/BII_delta_2000-2020.tif"

# Load both rasters (read the data)
with rasterio.open(bii_2000_path) as src_2000, rasterio.open(bii_2020_path) as src_2020:
    bii_2000 = src_2000.read(1).astype(np.float32)
    bii_2020 = src_2020.read(1).astype(np.float32)
    meta = src_2020.meta.copy()

# Now you can safely check unique values
print("Unique values in BII 2000:", np.unique(bii_2000[~np.isnan(bii_2000)]))
print("Unique values in BII 2020:", np.unique(bii_2020[~np.isnan(bii_2020)]))

# Compute delta
bii_delta = bii_2020 - bii_2000
bii_delta = np.where(np.isnan(bii_2000) | np.isnan(bii_2020), np.nan, bii_delta)

# Save delta raster
meta.update(dtype='float32', nodata=np.nan, compress='lzw')
with rasterio.open(bii_delta_path, 'w', **meta) as dst:
    dst.write(bii_delta, 1)

print("Delta map saved to:", bii_delta_path)

# Plot
plt.figure(figsize=(10, 6))
plt.imshow(bii_delta, cmap='RdYlGn', vmin=-20, vmax=20)
plt.title("Change in Biodiversity Intactness (2000–2020)")
plt.colorbar(label='Δ BII (% points)')
plt.axis("off")
plt.tight_layout()
plt.show()

# Load raster and extract valid values
with rasterio.open(bii_delta_path) as src:
    bii_delta = src.read(1).astype(np.float32)
    valid_values = bii_delta[~np.isnan(bii_delta)]

# Get unique values and summary statistics
unique_values = np.unique(valid_values)
print("Unique values (first 20):", unique_values[:20])
print("Min:", valid_values.min())
print("Max:", valid_values.max())
print("Mean:", valid_values.mean())
print("Median:", np.median(valid_values))
print("Std Dev:", valid_values.std())
print("Total valid pixels:", len(valid_values))

# Check for exact zero values
zero_count = np.sum(valid_values == 0)
print(f"Number of pixels with exactly zero change: {zero_count}")


# ### Binary Mask for BII Change

# In[17]:


from rasterio.warp import reproject, Resampling
from rasterio.crs import CRS
from rasterio.mask import mask
from rasterio.io import MemoryFile
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Paths and threshold
bii_delta_path = "/Users/michelle/Desktop/BA/BII/BII_delta_2000-2020.tif"
threshold = 2

loss_out_path = "/Users/michelle/Desktop/BA/BII/BII_loss_mask_2000_2020.tif"
gain_out_path = "/Users/michelle/Desktop/BA/BII/BII_gain_mask_2000_2020.tif"

# Load delta layer
with rasterio.open(bii_delta_path) as src:
    bii_delta = src.read(1).astype(np.float32)
    bii_meta = src.meta.copy()
    transform = src.transform
    crs = src.crs
    pixel_size_x = transform.a

# Generate binary masks
bii_loss_mask = np.where(bii_delta < -threshold, 1, np.nan).astype(np.float32)
bii_loss_mask[np.isnan(bii_delta)] = np.nan

bii_gain_mask = np.where(bii_delta > threshold, 1, np.nan).astype(np.float32)
bii_gain_mask[np.isnan(bii_delta)] = np.nan

# Plot BII Loss
plt.figure(figsize=(10, 8))
plt.imshow(bii_loss_mask, cmap=ListedColormap(["white", "purple"]), vmin=0, vmax=1)
plt.title(f"BII Loss Mask (< -{threshold})")
plt.axis("off")
plt.tight_layout()
plt.show()

# Plot BII Gain
plt.figure(figsize=(10, 8))
plt.imshow(bii_gain_mask, cmap=ListedColormap(["white", "green"]), vmin=0, vmax=1)
plt.title(f"BII Gain Mask (> {threshold})")
plt.axis("off")
plt.tight_layout()
plt.show()

# Save masks
bii_meta.update(dtype='float32', nodata=np.nan, compress='lzw')

with rasterio.open(loss_out_path, 'w', **bii_meta) as dst:
    dst.write(bii_loss_mask, 1)

with rasterio.open(gain_out_path, 'w', **bii_meta) as dst:
    dst.write(bii_gain_mask, 1)

print(f"Exported: {loss_out_path}")
print(f"Exported: {gain_out_path}")

# Print summary
def print_mask_summary(mask, name, pixel_size_deg=None, pixel_size_m=None):
    valid_pixels = np.sum(mask == 1)
    total_pixels = np.sum(~np.isnan(mask))
    percent = (valid_pixels / total_pixels) * 100 if total_pixels > 0 else 0

    print(f"\n{name} Summary:")
    print(f"  Total pixels: {total_pixels:,}")
    print(f"  Affected pixels: {valid_pixels:,}")
    print(f"  Percent of area: {percent:.2f}%")

# Print summaries based on CRS
if crs.to_string().startswith("EPSG:4326"):
    print_mask_summary(bii_loss_mask, "BII Loss", pixel_size_deg=pixel_size_x)
    print_mask_summary(bii_gain_mask, "BII Gain", pixel_size_deg=pixel_size_x)
else:
    print_mask_summary(bii_loss_mask, "BII Loss", pixel_size_m=pixel_size_x)
    print_mask_summary(bii_gain_mask, "BII Gain", pixel_size_m=pixel_size_x)


# ### Look for correlation between afforestation/deforestation and BII gain/loss

# In[21]:


from rasterio.warp import reproject, Resampling

# Paths
def_path = "/Users/michelle/Desktop/BA/LCLU/forest_masks/deforestation_mask_2000_2020_c.tif"
aff_path = "/Users/michelle/Desktop/BA/LCLU/forest_masks/afforestation_mask_2000_2020_c.tif"

bii_loss_path = "/Users/michelle/Desktop/BA/BII/BII_loss_mask_2000_2020.tif"
bii_gain_path = "/Users/michelle/Desktop/BA/BII/BII_gain_mask_2000_2020.tif"

# Load forest masks (deforestation & afforestation)
with rasterio.open(def_path) as def_src:
    def_mask = def_src.read(1)
    ref_transform = def_src.transform
    ref_crs = def_src.crs
    forest_shape = def_mask.shape

with rasterio.open(aff_path) as aff_src:
    aff_mask = aff_src.read(1)

# Load BII gain/loss masks and metadata
with rasterio.open(bii_loss_path) as src:
    bii_loss = src.read(1)
    bii_meta = src.meta

with rasterio.open(bii_gain_path) as src:
    bii_gain = src.read(1)

# Resample BII masks to match forest shape if needed
if bii_loss.shape != forest_shape:
    bii_loss_resampled = np.empty(forest_shape, dtype=np.float32)
    bii_gain_resampled = np.empty(forest_shape, dtype=np.float32)

    reproject(
        source=bii_loss,
        destination=bii_loss_resampled,
        src_transform=bii_meta["transform"],
        src_crs=bii_meta["crs"],
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.nearest
    )

    reproject(
        source=bii_gain,
        destination=bii_gain_resampled,
        src_transform=bii_meta["transform"],
        src_crs=bii_meta["crs"],
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.nearest
    )
else:
    bii_loss_resampled = bii_loss
    bii_gain_resampled = bii_gain

# Create overlap masks
overlap_defo_bii = (def_mask == 1) & (bii_loss_resampled == 1)
overlap_affo_bii = (aff_mask == 1) & (bii_gain_resampled == 1)

# Pixel counts
total_defo = np.nansum(def_mask == 1)
total_affo = np.nansum(aff_mask == 1)
total_bii_loss = np.nansum(bii_loss_resampled == 1)
total_bii_gain = np.nansum(bii_gain_resampled == 1)
overlap_defo_bii_count = np.nansum(overlap_defo_bii)
overlap_affo_bii_count = np.nansum(overlap_affo_bii)

# Summary
print("CORRELATION SUMMARY")
print(f"Deforestation pixels: {total_defo:,}")
print(f"BII loss pixels:     {total_bii_loss:,}")
print(f"Overlap (Defo ∩ BII loss): {overlap_defo_bii_count:,} ({(overlap_defo_bii_count/total_defo*100 if total_defo else 0):.2f}%)")

print(f"\n Afforestation pixels: {total_affo:,}")
print(f"BII gain pixels:      {total_bii_gain:,}")
print(f"Overlap (Affo ∩ BII gain): {overlap_affo_bii_count:,} ({(overlap_affo_bii_count/total_affo*100 if total_affo else 0):.2f}%)")


# ### Statisical analysis

# In[3]:


import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from scipy.stats import chi2_contingency

# Paths
def_path = "/Users/michelle/Desktop/BA/LCLU/forest_masks/deforestation_mask_2000_2020_c.tif"
aff_path = "/Users/michelle/Desktop/BA/LCLU/forest_masks/afforestation_mask_2000_2020_c.tif"
bii_loss_path = "/Users/michelle/Desktop/BA/BII/BII_loss_mask_2000_2020.tif"
bii_gain_path = "/Users/michelle/Desktop/BA/BII/BII_gain_mask_2000_2020.tif"

# Load forest masks
with rasterio.open(def_path) as def_src:
    def_mask = def_src.read(1)
    ref_transform = def_src.transform
    ref_crs = def_src.crs
    forest_shape = def_mask.shape

with rasterio.open(aff_path) as aff_src:
    aff_mask = aff_src.read(1)

# Load BII gain/loss masks and metadata
with rasterio.open(bii_loss_path) as src:
    bii_loss = src.read(1)
    bii_meta = src.meta

with rasterio.open(bii_gain_path) as src:
    bii_gain = src.read(1)

# Resample BII masks to match forest shape if needed
if bii_loss.shape != forest_shape:
    bii_loss_resampled = np.empty(forest_shape, dtype=np.float32)
    bii_gain_resampled = np.empty(forest_shape, dtype=np.float32)

    reproject(
        source=bii_loss,
        destination=bii_loss_resampled,
        src_transform=bii_meta["transform"],
        src_crs=bii_meta["crs"],
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.nearest
    )

    reproject(
        source=bii_gain,
        destination=bii_gain_resampled,
        src_transform=bii_meta["transform"],
        src_crs=bii_meta["crs"],
        dst_transform=ref_transform,
        dst_crs=ref_crs,
        resampling=Resampling.nearest
    )
else:
    bii_loss_resampled = bii_loss
    bii_gain_resampled = bii_gain

# Create overlap masks
overlap_defo_bii = (def_mask == 1) & (bii_loss_resampled == 1)
overlap_affo_bii = (aff_mask == 1) & (bii_gain_resampled == 1)

# Pixel counts
total_defo = np.nansum(def_mask == 1)
total_affo = np.nansum(aff_mask == 1)
total_bii_loss = np.nansum(bii_loss_resampled == 1)
total_bii_gain = np.nansum(bii_gain_resampled == 1)
overlap_defo_bii_count = np.nansum(overlap_defo_bii)
overlap_affo_bii_count = np.nansum(overlap_affo_bii)

# Summary
print("CORRELATION SUMMARY")
print(f"Deforestation pixels:       {total_defo:,}")
print(f"BII loss pixels:            {total_bii_loss:,}")
print(f"Overlap (Defo ∩ BII loss):  {overlap_defo_bii_count:,} ({(overlap_defo_bii_count/total_defo*100 if total_defo else 0):.2f}%)")

print(f"\nAfforestation pixels:       {total_affo:,}")
print(f"BII gain pixels:            {total_bii_gain:,}")
print(f"Overlap (Affo ∩ BII gain):  {overlap_affo_bii_count:,} ({(overlap_affo_bii_count/total_affo*100 if total_affo else 0):.2f}%)")

# Chi-square test for deforestation vs BII loss
a = overlap_defo_bii_count
b = total_defo - a
c = total_bii_loss - a
valid_pixels_defo = (~np.isnan(def_mask)) & (~np.isnan(bii_loss_resampled))
total_valid_defo = np.sum(valid_pixels_defo)
d = total_valid_defo - (a + b + c)
if d < 0:
    d = 0  

contingency_defo = np.array([[a, b],
                             [c, d]])

chi2_defo, p_defo, _, _ = chi2_contingency(contingency_defo)

print("\nCHI-SQUARE TEST (Deforestation vs. BII loss)")
print(f"Chi2 statistic: {chi2_defo:.2f}, p-value: {p_defo:.4f}")

# Chi-square test for afforestation vs BII gain
a2 = overlap_affo_bii_count
b2 = total_affo - a2
c2 = total_bii_gain - a2
valid_pixels_affo = (~np.isnan(aff_mask)) & (~np.isnan(bii_gain_resampled))
total_valid_affo = np.sum(valid_pixels_affo)
d2 = total_valid_affo - (a2 + b2 + c2)
if d2 < 0:
    d2 = 0  

contingency_affo = np.array([[a2, b2],
                             [c2, d2]])

chi2_affo, p_affo, _, _ = chi2_contingency(contingency_affo)

print("\nCHI-SQUARE TEST (Afforestation vs. BII gain)")
print(f"Chi2 statistic: {chi2_affo:.2f}, p-value: {p_affo:.4f}")


# ### Hotspot classes area

# In[3]:


import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Path to hotspot raster and shapefile
hotspot_raster_path = "/Users/michelle/Desktop/BA/Hotspot/Hotspotmap.finale.tif"
shapefile_path = "/Users/michelle/Desktop/BA/Boundaries/gadm41_MDG_shp/gadm41_MDG_0.shp"

# Classify function
def classify_hotspot(value):
    if value <= 3000:
        return 1
    elif value <= 10000:
        return 2
    elif value <= 20000:
        return 3
    else:
        return 4

# Class labels
class_labels = {
    1: "Low (≤ 3,000)",
    2: "Moderate (3,000 – 10,000)",
    3: "High (10,000 – 20,000)",
    4: "Very High (> 20,000)"
}

# Load Madagascar shapefile using geopandas
gdf = gpd.read_file(shapefile_path)

# Ensure the shapefile and raster are in the same CRS
with rasterio.open(hotspot_raster_path) as src:
    raster_crs = src.crs

# Reproject the shapefile to match the raster's CRS
if gdf.crs != raster_crs:
    gdf = gdf.to_crs(raster_crs)

# Get the geometry of Madagascar (assuming it's the first feature in the shapefile)
geometry = [gdf.geometry.unary_union]

# Open the raster and apply the mask (clip the raster)
with rasterio.open(hotspot_raster_path) as src:
    # Mask the raster with the shapefile geometry
    out_image, out_transform = mask(src, geometry, crop=True)
    out_meta = src.meta

# Update metadata for the clipped raster
out_meta.update({"driver": "GTiff", "count": 1, "crs": raster_crs, "transform": out_transform})

# Save the clipped raster
clipped_raster_path = "/Users/michelle/Desktop/BA/Hotspot/clipped_hotspot.tif"
with rasterio.open(clipped_raster_path, "w", **out_meta) as dest:
    dest.write(out_image)

# Mask nodata
masked_data = out_image[0][out_image[0] != src.nodata]

# Classify each value
classified = np.vectorize(classify_hotspot)(masked_data)

# Count pixels per class
unique, counts = np.unique(classified, return_counts=True)
pixel_area_km2 = src.res[0] * src.res[1] / 1_000_000  # m² to km²
areas_km2 = counts * pixel_area_km2

# Create DataFrame with % 
df = pd.DataFrame({
    "Class": [class_labels[u] for u in unique],
    "Pixels": counts,
    "Area (km²)": np.round(areas_km2, 2),
    "Area (%)": np.round((areas_km2 / areas_km2.sum()) * 100, 2)
})

# Print the table
print(df.to_string(index=False))

# Optionally, plot the clipped raster
plt.imshow(out_image[0], cmap='viridis')
plt.title("Clipped Hotspot Layer")
plt.colorbar(label='Hotspot Intensity')
plt.show()

print(f"Clipped raster saved to {clipped_raster_path}")


# ### Deforestation in regions

# In[42]:


import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
from shapely.geometry import box

# Paths
shapefile_path = "/Users/michelle/Downloads/BNDA_MDG_2023-06-29_lastupdate/BNDA_MDG_2023-06-29_lastupdate.shp"
trajectory_raster = "/Users/michelle/Desktop/BA/LCLU/forest_masks/trajectory_grouped_MAD_UTM.tif"

# Load shapefile and extract Menabe region 
gdf = gpd.read_file(shapefile_path)
print("Shapefile loaded.")

# Check available region names and select Menabe from the correct column
print("Available columns:", gdf.columns)
print("Region names (adm2nm):", gdf["adm2nm"].unique())

# Correct filter for Menabe
menabe = gdf[gdf["adm2nm"] == "Menabe"]
print(f"Menabe geometry found: {len(menabe)} feature(s)")

# Open raster and process
with rasterio.open(trajectory_raster) as src:
    raster_crs = src.crs
    print("Raster CRS:", raster_crs)

    # Reproject Menabe to match raster
    menabe = menabe.to_crs(raster_crs)

    # Log bounding boxes for visual check
    raster_bounds = src.bounds
    menabe_bounds = menabe.total_bounds
    print("Raster bounds:", raster_bounds)
    print("Menabe bounds:", menabe_bounds)

    # Check intersection
    raster_bounds_geom = gpd.GeoDataFrame(geometry=[box(*raster_bounds)], crs=raster_crs)
    intersection = raster_bounds_geom.intersects(menabe.geometry.unary_union).values[0]

    if not intersection:
        raise ValueError("Menabe does not intersect raster extent!")

    # Mask raster to Menabe
    out_image, out_transform = mask(src, menabe.geometry, crop=True)
    pixel_area_km2 = src.res[0] * src.res[1] / 1_000_000  # m² to km²
    nodata = src.nodata

# Extract data and remove NoData values
data = out_image[0]
if nodata is not None:
    data = data[data != nodata]

# Count pixels per class
unique, counts = np.unique(data, return_counts=True)

# Calculate total area and deforestation
total_area_km2 = np.sum(counts * pixel_area_km2)
deforestation_pixels = sum(counts[i] for i, v in enumerate(unique) if v in [3, 4])
deforestation_area_km2 = deforestation_pixels * pixel_area_km2
deforestation_percent = (deforestation_area_km2 / total_area_km2) * 100

# Output
print(f"\n Total area of Menabe in raster: {total_area_km2:,.2f} km²")
print(f" Deforestation in Menabe (Classes 3 & 4): {deforestation_area_km2:,.2f} km² ({deforestation_percent:.2f}%)")


# In[43]:


import geopandas as gpd
import rasterio
from rasterio.mask import mask
import numpy as np
from shapely.geometry import box

# Paths
shapefile_path = "/Users/michelle/Downloads/BNDA_MDG_2023-06-29_lastupdate/BNDA_MDG_2023-06-29_lastupdate.shp"
trajectory_raster = "/Users/michelle/Desktop/BA/LCLU/forest_masks/trajectory_grouped_MAD_UTM.tif"

# Load shapefile and extract Sava region
gdf = gpd.read_file(shapefile_path)
sava = gdf[gdf["adm2nm"] == "Sava"]
print(f"Sava geometry found: {len(sava)} feature(s)")

# Open raster and process
with rasterio.open(trajectory_raster) as src:
    raster_crs = src.crs
    sava = sava.to_crs(raster_crs)

    # Check bounding box overlap
    raster_bounds_geom = gpd.GeoDataFrame(geometry=[box(*src.bounds)], crs=raster_crs)
    intersection = raster_bounds_geom.intersects(sava.geometry.unary_union).values[0]
    if not intersection:
        raise ValueError("Sava does not intersect raster extent!")

    # Mask raster to Sava geometry
    out_image, out_transform = mask(src, sava.geometry, crop=True)
    pixel_area_km2 = src.res[0] * src.res[1] / 1_000_000
    nodata = src.nodata

# Clean raster data
data = out_image[0]
if nodata is not None:
    data = data[data != nodata]

# Count pixel values and calculate area
unique, counts = np.unique(data, return_counts=True)
total_area_km2 = np.sum(counts * pixel_area_km2)
deforestation_pixels = sum(counts[i] for i, v in enumerate(unique) if v in [3, 4])
deforestation_area_km2 = deforestation_pixels * pixel_area_km2
deforestation_percent = (deforestation_area_km2 / total_area_km2) * 100

# Output
print(f"\n Total area of Sava in raster: {total_area_km2:,.2f} km²")
print(f" Deforestation in Sava (Classes 3 & 4): {deforestation_area_km2:,.2f} km² ({deforestation_percent:.2f}%)")

