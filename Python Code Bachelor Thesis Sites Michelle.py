#!/usr/bin/env python
# coding: utf-8

# # Python Code Bachelor Thesis: Sites Analysis

# ### Michelle Köthe

# ##### The structure and syntax of this code were generated with the aid of ChatGPT-4o

# ### Load and clean the data

# In[43]:


import pandas as pd
import geopandas as gpd
from shapely.geometry import Point

# Load the Excel file
df = pd.read_excel("/Users/michelle/Desktop/Visited Plots Michelle.xlsx")

# Split coordinates into Latitude and Longitude
df[['Latitude', 'Longitude']] = df['Coordinates (Lat, Long)'].str.strip().str.split(',', expand=True).astype(float)

# Create GeoDataFrame with Point geometry
geometry = [Point(xy) for xy in zip(df["Longitude"], df["Latitude"])]
resto_gdf = gpd.GeoDataFrame(df, geometry=geometry, crs="EPSG:4326")

# (Re)split Coordinates column in 2 
df[['Latitude', 'Longitude']] = df['Coordinates (Lat, Long)'].str.strip().str.split(',', expand=True).astype(float)

# Save as a new CSV
df.to_csv("/Users/michelle/Desktop/BA/restoration_sites_correct.csv", index=False)


# ### Group based on Land use type

# In[35]:


# Define the mapping dictionary
landuse_map = {
    "FF": "Forest",
    "RL": "Restoration",
    "UL": "Degraded"
}

# Map to new column
resto_gdf["landuse_category"] = resto_gdf["Land-use type"].map(landuse_map)

# Check for missing entries (should be none)
missing = resto_gdf[resto_gdf["landuse_category"].isna()]["Land-use type"].unique()
print("Unmapped land-use types:", missing)

resto_gdf.to_file("/Users/michelle/Desktop/BA/resto_sites_categorized.gpkg", driver="GPKG")

import seaborn as sns
import matplotlib.pyplot as plt

# Simple count of sites per land-use category
sns.countplot(data=resto_gdf, x="landuse_category", order=["Forest", "Restoration", "Degraded"])
plt.title("Number of Restoration Sites per Land-Use Category")
plt.xlabel("Land-Use Category")
plt.ylabel("Number of Sites")
plt.tight_layout()
plt.show()


# ### Reproject points to match the raster

# In[36]:


import rasterio
import numpy as np

raster_path = "/Users/michelle/Desktop/BA/LCLU/forest_masks/trajectory_grouped_MAD.tif"

# Open and sample inside the same block
with rasterio.open(raster_path) as src:
    # Reproject to raster CRS
    resto_projected = resto_gdf.to_crs(src.crs)

    # Create (x, y) coordinate list
    coords = [(geom.x, geom.y) for geom in resto_projected.geometry if not geom.is_empty]

    # Sample raster at each point
    sampled = list(src.sample(coords))

    # Handle nodata
    nodata = src.nodata if src.nodata is not None else -9999

# Assign values to new column
resto_gdf["trajectory_class"] = [val[0] if val and val[0] != nodata else np.nan for val in sampled]


# ### Extract trajectory value at each site

# In[37]:


with rasterio.open(raster_path) as src:
    # Reproject restoration sites to match raster CRS
    resto_projected = resto_gdf.to_crs(src.crs)

    # Get coordinates for sampling
    coords = [(geom.x, geom.y) for geom in resto_projected.geometry]

    # Sample raster values at these coordinates
    sampled = list(src.sample(coords))

    # Handle nodata value
    nodata = src.nodata if src.nodata is not None else -9999

# Assign values back to original GeoDataFrame
resto_gdf["trajectory"] = [val[0] if val and val[0] != nodata else np.nan for val in sampled]


# ### Analyze overlap between sites and forest trajectories

# In[38]:


# Open raster and sample values
with rasterio.open(raster_path) as src:
    # Reproject restoration sites to match raster CRS
    resto_projected = resto_gdf.to_crs(src.crs)

    # Get coordinates for sampling
    coords = [(geom.x, geom.y) for geom in resto_projected.geometry]

    # Sample raster values at these coordinates
    sampled = list(src.sample(coords))

    # Handle nodata value
    nodata = src.nodata if src.nodata is not None else -9999

    # Optional: print all unique values in the raster (excluding nodata)
    data = src.read(1)
    unique_vals = np.unique(data[data != nodata])
    print("All trajectory classes in raster:", unique_vals)

# Assign sampled values to GeoDataFrame
resto_gdf["trajectory"] = [val[0] if val and val[0] != nodata else np.nan for val in sampled]

# Drop rows with no valid value
resto_clean = resto_gdf.dropna(subset=["trajectory"])

# Summarize with all 8 classes shown
all_classes = [1, 2, 3, 4, 5, 6, 7, 8]  # define expected trajectory codes
trajectory_summary = resto_clean.groupby("Land-use type")["trajectory"]\
                                .value_counts()\
                                .unstack(fill_value=0)\
                                .reindex(columns=all_classes, fill_value=0)

# Display the result
print(trajectory_summary)

total_sites = trajectory_summary.sum().sum()
print("Total number of sites:", total_sites)


# ### Bar Chart

# In[39]:


import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Define all trajectory classes
all_classes = [1, 2, 3, 4, 5, 6, 7, 8]

# Group by land-use type and trajectory class
trajectory_summary = resto_clean.groupby("Land-use type")["trajectory"]\
    .value_counts().unstack(fill_value=0).reindex(columns=all_classes, fill_value=0)

# Human-readable labels for trajectory classes
trajectory_labels = {
    1: "Stable forest",
    2: "Stable non-forest",
    3: "Early deforestation",
    4: "Late deforestation",
    5: "Early afforestation",
    6: "Late afforestation",
    7: "Forest recovery",
    8: "Temporary change"
}

# Mapping land-use codes to full names
land_use_labels = {
    "FF": "Forest Fragment",
    "RL": "Restored Land",
    "UL": "Unrestored Land"
}

# Custom colors for each land-use type
custom_colors = {
    "Forest Fragment": "#1a6311",       # dark green
    "Restored Land": "#9ed26a",         # light green
    "Unrestored Land": "#f6eb5a"        # yellow
}

# Rename columns with human-readable trajectory class names
trajectory_named = trajectory_summary.copy()
trajectory_named.columns = [trajectory_labels.get(col, col) for col in trajectory_named.columns]

# Rename land-use types (index)
trajectory_named.index = trajectory_named.index.map(land_use_labels)

# Plot
ax = trajectory_named.T.plot(
    kind="bar", 
    figsize=(12, 6),
    color=[custom_colors[col] for col in trajectory_named.T.columns]  # fix here
)

# Frame all sides
for spine in ["top", "right", "bottom", "left"]:
    ax.spines[spine].set_visible(True)

# Grid style
ax.yaxis.grid(True, linestyle="--", alpha=0.5)
ax.xaxis.grid(False)

# Y-axis: only whole numbers
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# Labels and styling
plt.xlabel("Trajectory Class")
plt.ylabel("Number of Sites")
plt.xticks(rotation=30)
plt.legend(title="Land-Use Type", bbox_to_anchor=(1.05, 1), loc="upper left")

plt.tight_layout()
plt.savefig("trajectory_by_landuse.png", dpi=300, bbox_inches="tight")
plt.show()


# ### BII and Sites

# In[18]:


from rasterstats import point_query

# Load restoration GeoPackage
resto_gdf = gpd.read_file("/Users/michelle/Desktop/BA/resto_sites_categorized.gpkg")

# Load BII gain/loss masks and metadata
with rasterio.open("/Users/michelle/Desktop/BA/BII/BII_gain_mask_2000_2020.tif") as gain_src:
    bii_crs = gain_src.crs
    bii_transform = gain_src.transform
    bii_gain = gain_src.read(1)

with rasterio.open("/Users/michelle/Desktop/BA/BII/BII_loss_mask_2000_2020.tif") as loss_src:
    bii_loss = loss_src.read(1)

# Reproject restoration sites to match BII raster
resto_gdf = resto_gdf.to_crs(bii_crs)

# Sample BII gain/loss at restoration points
resto_gdf["BII_gain"] = point_query(resto_gdf, bii_gain, affine=bii_transform, nodata=np.nan)
resto_gdf["BII_loss"] = point_query(resto_gdf, bii_loss, affine=bii_transform, nodata=np.nan)

# Fill NAs for summarizing
resto_gdf["BII_gain"] = resto_gdf["BII_gain"].fillna(0).astype(int)
resto_gdf["BII_loss"] = resto_gdf["BII_loss"].fillna(0).astype(int)

# Create summary table by land-use type
summary = resto_gdf.groupby("Land-use type")[["BII_gain", "BII_loss"]].sum()
summary["Total sites"] = resto_gdf.groupby("Land-use type").size()
summary["Gain %"] = (summary["BII_gain"] / summary["Total sites"] * 100).round(1)
summary["Loss %"] = (summary["BII_loss"] / summary["Total sites"] * 100).round(1)

# Clean display
print("Correlation between restoration type and BII change:\n")
print(summary[["Total sites", "BII_gain", "Gain %", "BII_loss", "Loss %"]])


# In[20]:


import geopandas as gpd
import rasterio
from rasterstats import point_query
import numpy as np
import pandas as pd
from shapely.geometry import Point
import matplotlib.pyplot as plt

# Load restoration points
resto_gdf = gpd.read_file("/Users/michelle/Desktop/BA/resto_sites_categorized.gpkg")

# Load BII raster masks
with rasterio.open("/Users/michelle/Desktop/BA/BII/BII_gain_mask_2000_2020.tif") as gain_src:
    bii_crs = gain_src.crs
    bii_transform = gain_src.transform
    bii_gain = gain_src.read(1)

with rasterio.open("/Users/michelle/Desktop/BA/BII/BII_loss_mask_2000_2020.tif") as loss_src:
    bii_loss = loss_src.read(1)

# Reproject points to raster CRS
resto_gdf = resto_gdf.to_crs(bii_crs)

# Query gain/loss normally
gain_values = point_query(resto_gdf, bii_gain, affine=bii_transform, nodata=np.nan)
loss_values = point_query(resto_gdf, bii_loss, affine=bii_transform, nodata=np.nan)

# Replace NaNs with nearest pixel values (fallback)
gain_nearest = point_query(resto_gdf, bii_gain, affine=bii_transform, interpolate='nearest')
loss_nearest = point_query(resto_gdf, bii_loss, affine=bii_transform, interpolate='nearest')

# Combine: use nearest only where original is NaN
final_gain = [g if not pd.isna(g) else gn for g, gn in zip(gain_values, gain_nearest)]
final_loss = [l if not pd.isna(l) else ln for l, ln in zip(loss_values, loss_nearest)]

# Assign to GeoDataFrame
resto_gdf["BII_gain"] = pd.Series(final_gain).fillna(0).astype(int)
resto_gdf["BII_loss"] = pd.Series(final_loss).fillna(0).astype(int)

# Create summary
summary = resto_gdf.groupby("Land-use type")[["BII_gain", "BII_loss"]].sum()
summary["Total sites"] = resto_gdf.groupby("Land-use type").size()
summary["No change"] = summary["Total sites"] - summary["BII_gain"] - summary["BII_loss"]

# Prepare for grouped bar chart
# Rename land-use codes to full names
land_use_labels = {
    "FF": "Forest Fragment",
    "RL": "Restored Land",
    "UL": "Unrestored Land"
}
summary.index = summary.index.map(land_use_labels)
labels = summary.index.tolist()
gain = summary["BII_gain"].values
loss = summary["BII_loss"].values
no_change = summary["No change"].values

x = np.arange(len(labels))
width = 0.25

# Plot
fig, ax = plt.subplots(figsize=(10, 6))
bars1 = ax.bar(x - width, gain, width, label='BII Gain', color='#9b59b6', edgecolor='black')       # violet
bars2 = ax.bar(x, loss, width, label='BII Loss', color='orange', edgecolor='black')                # orange
bars3 = ax.bar(x + width, no_change, width, label='No Change', color='#fff9e3', edgecolor='black') # panna (off-white)

# Styling
ax.set_ylabel('Number of Sites', fontsize=13)
ax.set_xlabel('Land-use Type', fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(labels, rotation=30, ha='right')
ax.legend()
ax.grid(axis="y", linestyle="--", alpha=0.5) 
ax.set_yticks(np.arange(0, 26, 5))

# labels
def add_labels(bars):
    for bar in bars:
        height = bar.get_height()
        ax.annotate(f'{int(height)}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=9)

add_labels(bars1)
add_labels(bars2)
add_labels(bars3)

plt.tight_layout()
plt.show()


fig.savefig("/Users/michelle/Desktop/BA/BII_restoration_bar_chart.pdf", format="pdf", bbox_inches="tight")

print("PDF exported")


# ### Sites and hotspot map

# In[21]:


import matplotlib.pyplot as plt
# Load data
hotspotmap = "/Users/michelle/Desktop/BA/Hotspot/Heatmap_correct_mk.tif"
resto_sites_path = "/Users/michelle/Desktop/BA/resto_sites_categorized.gpkg"
resto_sites = gpd.read_file(resto_sites_path)

# Match CRS and query raster
with rasterio.open(hotspotmap) as src:
    resto_sites = resto_sites.to_crs(src.crs)
    resto_sites["deforestation_val"] = point_query(resto_sites, hotspotmap)

# Classification
def classify_hotspot(val):
    if pd.isna(val) or val == 0:
        return "No hotspot"
    elif val <= 3000:
        return "Low hotspot"
    elif val <= 10000:
        return "Moderate hotspot"
    elif val <= 20000:
        return "Strong hotspot"
    else:
        return "Very strong hotspot"
resto_sites["hotspot_class_5"] = resto_sites["deforestation_val"].apply(classify_hotspot)

# Ensure all 5 classes are represented
all_classes = [
    "No hotspot",
    "Low hotspot",
    "Moderate hotspot",
    "Strong hotspot",
    "Very strong hotspot"
]

class_counts = resto_sites["hotspot_class_5"].value_counts().reindex(all_classes, fill_value=0)

# Plot
plt.figure(figsize=(10, 6))
bars = plt.bar(class_counts.index, class_counts.values, color="orangered")
plt.title("Restoration Sites by 5-Class Deforestation Hotspot Intensity (Including Class 0)")
plt.ylabel("Number of Sites")
plt.xticks(rotation=15, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height + 0.5, f'{int(height)}', ha='center', va='bottom')

plt.tight_layout()
plt.show()


# In[22]:


# Display a table with relevant info
print(
    resto_sites[[
        "Fokontany (Village)",
        "landuse_category",
        "deforestation_val",
        "hotspot_class_5"
    ]]
)


# ### Restored sites and hotspot map

# In[41]:


# Load 
import matplotlib.pyplot as plt

hotspotmap = "/Users/michelle/Desktop/BA/Hotspot/Hotspot_correct.tif"
resto_sites_path = "/Users/michelle/Desktop/BA/resto_sites_categorized.gpkg"
resto_sites = gpd.read_file(resto_sites_path)

# Match CRS and sample raster values
with rasterio.open(hotspotmap) as src:
    resto_sites = resto_sites.to_crs(src.crs)
    resto_sites["deforestation_val"] = point_query(resto_sites, hotspotmap)

# Filter for "Restoration" sites only
restored_sites = resto_sites[resto_sites["landuse_category"] == "Restoration"].copy()

# === Step 4: Classify into simplified 4-class names ===
def classify_hotspot(val):
    if pd.isna(val) or val == 0:
        return "No hotspot"
    elif val <= 3000:
        return "Low hotspot"
    elif val <= 10000:
        return "Moderate hotspot"
    elif val <= 20000:
        return "Strong hotspot"
    else:
        return "Very strong hotspot"

restored_sites["hotspot_class_clean"] = restored_sites["deforestation_val"].apply(classify_hotspot)

# Remove 'No hotspot' class
filtered_restored = restored_sites[restored_sites["hotspot_class_clean"] != "No hotspot"]

# Define clean order
ordered_classes = ["Low hotspot", "Moderate hotspot", "Strong hotspot", "Very strong hotspot"]
class_counts = filtered_restored["hotspot_class_clean"].value_counts().reindex(ordered_classes, fill_value=0)

# Plot
plt.figure(figsize=(10, 7))
bars = plt.bar(class_counts.index, class_counts.values, color="seagreen")
plt.ylabel("Number of Restored Sites")
plt.xticks(rotation=0)

# Grid: only horizontal dashed lines
ax = plt.gca()  # Get current axis
ax.yaxis.grid(True, linestyle='--', alpha=0.7)
ax.xaxis.grid(False)  # ➔ Deaktiviert vertikale Linien korrekt

# Add labels on bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2., height + 0.5, f'{int(height)}', ha='center', va='bottom')

# Axis labels
plt.ylabel("Number of Restored Sites", fontsize=14)
plt.xticks(rotation=0, fontsize=12)
plt.yticks(fontsize=12)

plt.tight_layout()

# Save if you want
plt.savefig("/Users/michelle/Desktop/BA/restored_sites_by_hotspot_clean.pdf", dpi=300, bbox_inches="tight", format="pdf")

plt.show()


# ### Clean Exel table

# In[24]:


print(resto_gdf.columns.tolist())


# In[25]:


import rasterio
from rasterstats import point_query
import numpy as np
import pandas as pd

# File paths
trajectory_raster = "/Users/michelle/Desktop/BA/LCLU/forest_masks/trajectory_grouped_MAD.tif"
hotspot_raster = "/Users/michelle/Desktop/BA/Hotspot/Hotspot_correct.tif"

# Sample trajectory from raster
with rasterio.open(trajectory_raster) as src:
    resto_gdf = resto_gdf.to_crs(src.crs)
    coords = [(geom.x, geom.y) for geom in resto_gdf.geometry]
    values = list(src.sample(coords))
    nodata = src.nodata if src.nodata is not None else -9999
    resto_gdf["trajectory"] = [val[0] if val and val[0] != nodata else np.nan for val in values]

# Sample deforestation values from hotspot map
with rasterio.open(hotspot_raster) as src:
    resto_gdf = resto_gdf.to_crs(src.crs)  # reproject to match if needed
    resto_gdf["deforestation_val"] = point_query(resto_gdf, hotspot_raster)

# Classify hotspot intensity
def classify_hotspot(val):
    if pd.isna(val) or val == 0:
        return "No hotspot"
    elif val <= 3000:
        return "Low hotspot"
    elif val <= 10000:
        return "Moderate hotspot"
    elif val <= 20000:
        return "Strong hotspot"
    else:
        return "Very strong hotspot"

resto_gdf["hotspot_class_5"] = resto_gdf["deforestation_val"].apply(classify_hotspot)

# Updated export: include region
columns_to_export = {
    "region": "region",
    "Fokontany (Village)": "site_name",
    "landuse_category": "site_type",
    "trajectory": "trajectory_class",
    "BII_gain": "bii_gain",
    "BII_loss": "bii_loss",
    "deforestation_val": "deforestation_value",
    "hotspot_class_5": "hotspot_class"
}

# Keep only columns that exist
available = [col for col in columns_to_export.keys() if col in resto_gdf.columns]
df_export = resto_gdf[available].copy()
df_export.rename(columns={col: columns_to_export[col] for col in available}, inplace=True)

# Save to Excel
excel_path = "/Users/michelle/Desktop/BA/restoration_analysis_table.xlsx"
df_export.to_excel(excel_path, index=False)

print(f"Exported full analysis table (with region) to:\n{excel_path}")


# ### Statistical analysis restored land and hotspot map

# In[30]:


import pandas as pd
from scipy.stats import chisquare

# Load the Excel file into a DataFrame
restoration_analysis_table = pd.read_excel("/Users/michelle/Desktop/BA/restoration_analysis_table.xlsx")

# Check column names to confirm correct spelling
print("Available columns:", restoration_analysis_table.columns)

# Filter only restored sites (adjust column name if needed!)
restored = restoration_analysis_table[restoration_analysis_table["site_type"] == "Restoration"]

# Count restored sites per hotspot class (drop NaNs)
observed = restored["hotspot_class"].dropna().value_counts().sort_index()

# Show observed counts
print("\nObserved counts per hotspot class:\n", observed)

# Create expected uniform distribution
expected = [len(restored) / len(observed)] * len(observed)

# Run chi-square test
chi2, p = chisquare(f_obs=observed, f_exp=expected)

# Show result
print(f"\nChi-square test result:")
print(f"Chi² = {chi2:.2f}, p = {p:.4f}")



# ### Statistical Analysis visited sites and trajectory map

# In[31]:


import pandas as pd
from scipy.stats import chi2_contingency

# Load data
file_path = "/Users/michelle/Desktop/BA/restoration_analysis_table.xlsx"
df = pd.read_excel(file_path)

# Display unique values for inspection (optional)
print("Unique site types:", df["site_type"].unique())
print("Unique trajectory classes:", df["trajectory_class"].unique())

# Drop rows with missing values
subset_df = df[["site_type", "trajectory_class"]].dropna()

# Create contingency table
contingency = pd.crosstab(subset_df["site_type"], subset_df["trajectory_class"])

# Run chi-square test of independence
chi2, p, dof, expected = chi2_contingency(contingency)

# Print results
print("\nContingency Table:")
print(contingency)
print(f"\nChi-square Test Result:")
print(f"Chi² = {chi2:.2f}, p = {p:.4f}, df = {dof}")


# ### Statistical analysis restored sites and BII change

# In[32]:


import pandas as pd
from scipy.stats import chisquare

# Load the data
file_path = "/Users/michelle/Desktop/BA/restoration_analysis_table.xlsx"
df = pd.read_excel(file_path)

# Filter restored sites only
restored_sites = df[df["site_type"] == "Restoration"]

# Count how many restored sites had BII loss (1) vs. no change (0)
bii_loss_counts = restored_sites["bii_loss"].value_counts().sort_index()

# Print observed counts
print("BII loss value counts among restored sites:")
print(bii_loss_counts)

# Perform chi-square goodness-of-fit test against a uniform (50/50) expectation
expected = [len(restored_sites) / 2] * 2
chi2, p = chisquare(f_obs=bii_loss_counts, f_exp=expected)

# Show results
print(f"\nChi-square goodness-of-fit test:")
print(f"Chi² = {chi2:.2f}, p = {p:.4f}")


# In[33]:


import pandas as pd
from scipy.stats import chi2_contingency

# Load the dataset
file_path = "/Users/michelle/Desktop/BA/restoration_analysis_table.xlsx"
df = pd.read_excel(file_path)

# Keep only BII_loss and site_type columns
df_bii = df[df["bii_loss"].isin([0, 1])]

# Create contingency table
contingency = pd.crosstab(df_bii["site_type"], df_bii["bii_loss"])

# Run chi-square test
chi2, p, dof, expected = chi2_contingency(contingency)

# Output results
print("Contingency Table (Site Type × BII Loss):")
print(contingency)
print(f"\nChi-square Test:")
print(f"Chi² = {chi2:.2f}, p = {p:.4f}, df = {dof}")

