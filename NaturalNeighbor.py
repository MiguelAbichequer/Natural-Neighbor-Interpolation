#This Python script performs Natural Neighbor Interpolation on a set of spatial data points extracted from a shapefile. 
#It uses Delaunay triangulation and Voronoi-based methods to interpolate values over a regular grid. The final output is visualized using Matplotlib and exported as a GeoTIFF raster file.
import numpy as np
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
from metpy.interpolate.points import natural_neighbor_point
from metpy.interpolate import geometry
import geopandas as gpd
import rasterio
from rasterio.transform import from_origin

# Load the shapefile and extract coordinates
shapefile_path = r"Path to your file.shp"
points = gpd.read_file(shapefile_path)
x_coords = points.geometry.x.values
y_coords = points.geometry.y.values
z_values = points['target_column'].values  # Replace with your actual data column

# Create a matrix of points for Delaunay triangulation
all_points = np.vstack((x_coords, y_coords)).T

# Perform Delaunay triangulation
triangulation = Delaunay(all_points)

point_count = len(x_coords)  # Parameter used as the base for plot scale

# Create a regular grid of points where interpolation will be performed
resolution = 1 / 10
grid_x, grid_y = np.meshgrid(
    np.arange(min(x_coords), max(x_coords), resolution),
    np.arange(min(y_coords), max(y_coords), resolution)
)

# Apply Natural Neighbor interpolation over the grid
grid_z = np.full(grid_x.shape, np.nan)

for i in range(grid_x.shape[0]):
    for j in range(grid_x.shape[1]):
        grid_point = (grid_x[i, j], grid_y[i, j])
        neighbors, circumcenters = geometry.find_natural_neighbors(triangulation, [grid_point])

        if len(neighbors[0]) > 0 and len(circumcenters) > 0:
            grid_z[i, j] = natural_neighbor_point(
                x_coords, y_coords, z_values, grid_point,
                triangulation, neighbors[0], circumcenters
            )

# Define transformation to save the raster file
transform = from_origin(min(x_coords), max(y_coords), resolution, resolution)

# Save the interpolated grid as a TIFF file
tiff_path = r"C:\Users\User\Desktop\natural_neighbor_interpolation.tiff"
with rasterio.open(
    tiff_path,
    'w',
    driver='GTiff',
    height=grid_z.shape[0],
    width=grid_z.shape[1],
    count=1,
    dtype=grid_z.dtype,
    crs="EPSG:4326",
    transform=transform,
) as dst:
    dst.write(grid_z, 1)

# Plot the results
plt.figure(figsize=(8, 6))
plt.contourf(grid_x, grid_y, grid_z, levels=100, cmap='viridis')
plt.colorbar(label='Intensity (Z)')
plt.scatter(x_coords, y_coords, c=z_values, edgecolors='k', cmap='viridis')
plt.title('Natural Neighbor Interpolation')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()
