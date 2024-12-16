import os
import numpy as np
import scipy
from scipy.spatial.distance import cdist
import slicer
from DICOMLib import DICOMUtils
from collections import deque

# Define a threshold for grouping nearby points (in voxel space)
#distance_threshold = 4  # This can be adjusted based on your dataset

# Function to group points that are close to each other
def group_points(points, threshold):
    grouped_points = []
    while points:
        point = points.pop()  # Take one point from the list
        group = [point]  # Start a new group
        
        # Find all points close to this one
        distances = cdist([point], points)  # Calculate distances from this point to others
        close_points = [i for i, dist in enumerate(distances[0]) if dist < threshold]
        
        # Add the close points to the group
        group.extend([points[i] for i in close_points])
        
        # Remove the grouped points from the list
        points = [point for i, point in enumerate(points) if i not in close_points]
        
        # Add the group to the result
        grouped_points.append(group)
    
    return grouped_points


def region_growing(image_data, seed, intensity_threshold, max_distance):
    dimensions = image_data.GetDimensions()
    visited = set()
    region = []
    queue = deque([seed])

    while queue:
        x, y, z = queue.popleft()
        if (x, y, z) in visited:
            continue

        visited.add((x, y, z))
        voxel_value = image_data.GetScalarComponentAsDouble(x, y, z, 0)
        
        if voxel_value >= intensity_threshold:
            region.append((x, y, z))
            # Add neighbors within bounds
            for dx, dy, dz in [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]:
                nx, ny, nz = x + dx, y + dy, z + dz
                if 0 <= nx < dimensions[0] and 0 <= ny < dimensions[1] and 0 <= nz < dimensions[2]:
                    if (nx, ny, nz) not in visited:
                        queue.append((nx, ny, nz))

    return region


def detect_points_region_growing(volume_name, intensity_threshold=3000, x_min=90, x_max=380, y_min=190, y_max=380, z_min=80, z_max=120, max_distance=9, centroid_merge_threshold=5):
    volume_node = slicer.util.getNode(volume_name)
    if not volume_node:
        raise RuntimeError(f"Volume {volume_name} not found.")
    
    image_data = volume_node.GetImageData()
    matrix = vtk.vtkMatrix4x4()
    volume_node.GetIJKToRASMatrix(matrix)

    dimensions = image_data.GetDimensions()
    detected_regions = []

    # Check if it's CT or CBCT based on volume name (or other criteria)
    is_cbct = "cbct" in volume_name.lower()

    if is_cbct:
        # For CBCT, use actual dimensions
        valid_x_min = 0
        valid_x_max = dimensions[0] - 1
        valid_y_min = 0
        valid_y_max = dimensions[1] - 1
        valid_z_min = 0
        valid_z_max = dimensions[2] - 1
        print(f"CBCT Volume {volume_name}: Using volume dimensions: x({valid_x_min}-{valid_x_max}), y({valid_y_min}-{valid_y_max}), z({valid_z_min}-{valid_z_max})")
    else:
        # For CT, use user-provided bounds
        valid_x_min = max(x_min, 0)
        valid_x_max = min(x_max, dimensions[0] - 1)
        valid_y_min = max(y_min, 0)
        valid_y_max = min(y_max, dimensions[1] - 1)
        valid_z_min = max(z_min, 0)
        valid_z_max = min(z_max, dimensions[2] - 1)
        print(f"CT Volume {volume_name}: Using bounds: x({valid_x_min}-{valid_x_max}), y({valid_y_min}-{valid_y_max}), z({valid_z_min}-{valid_z_max})")

    # Create a set to track visited voxels and avoid redundant searches
    visited = set()

    # Define a helper function for region growing in a specific voxel
    def grow_region(x, y, z):
        # Skip if already visited
        if (x, y, z) in visited:
            return None

        voxel_value = image_data.GetScalarComponentAsDouble(x, y, z, 0)
        if voxel_value < intensity_threshold:
            return None

        # Perform region growing
        region = region_growing(image_data, (x, y, z), intensity_threshold, max_distance=max_distance)
        if region:
            for point in region:
                visited.add(tuple(point))  # Mark voxel as visited
            return region
        return None

    # Process voxels within the valid bounds
    regions = []
    for z in range(valid_z_min, valid_z_max + 1):  # Iterate over z bounds in voxel space
        for y in range(valid_y_min, valid_y_max + 1):  # Limit y values within bounds (voxel space)
            for x in range(valid_x_min, valid_x_max + 1):  # Limit x values within bounds (voxel space)
                region = grow_region(x, y, z)
                if region:
                    regions.append(region)

    # Collect centroids from regions
    centroids = []
    for region in regions:
        points = np.array([matrix.MultiplyPoint([*point, 1])[:3] for point in region])
        centroid = np.mean(points, axis=0)
        centroids.append(np.round(centroid, 2))  # Round to avoid floating point precision issues

    # Merging centroids that are too close to each other (within `centroid_merge_threshold` mm)
    unique_centroids = []
    for centroid in centroids:
        # Check if the centroid is close to any existing ones (merge if within threshold)
        if not any(np.linalg.norm(centroid - existing_centroid) < centroid_merge_threshold for existing_centroid in unique_centroids):
            unique_centroids.append(centroid)

    # Create markup points for each unique centroid
    markups_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLMarkupsFiducialNode", f"Markers_{volume_name}")
    for centroid in unique_centroids:
        markups_node.AddControlPoint(*centroid)
        print(f"Detected Centroid (RAS): {centroid}")

    return unique_centroids

# Initialize lists and dictionary
cbct_list = []
ct_list = []
volume_points_dict = {}

# Process loaded volumes
for volumeNode in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
    volumeName = volumeNode.GetName()
    shNode = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    imageItem = shNode.GetItemByDataNode(volumeNode)
    
    modality = shNode.GetItemAttribute(imageItem, 'DICOM.Modality')
    print(modality)
    
    # Check if the volume is loaded into the scene
    if not slicer.mrmlScene.IsNodePresent(volumeNode):
        print(f"Volume {volumeName} not present in the scene.")
        continue
    
    # Determine scan type
    if "cbct" in volumeName.lower():
        cbct_list.append(volumeName)
        scan_type = "CBCT"
    else:
        ct_list.append(volumeName)
        scan_type = "CT"
    
    # Detect points using region growing
    grouped_points = detect_points_region_growing(volumeName, intensity_threshold=3000)
    volume_points_dict[(scan_type, volumeName)] = grouped_points

# Print the results
print(f"\nCBCT Volumes: {cbct_list}")
print(f"CT Volumes: {ct_list}")
print("\nDetected Points by Volume:")
for (scan_type, vol_name), points in volume_points_dict.items():
    print(f"{scan_type} Volume '{vol_name}': {len(points)} points detected.")


# for volumeNode in slicer.util.getNodesByClass("vtkMRMLScalarVolumeNode"):
#     volumeName = volumeNode.GetName()
#     print(f"Volume: {volumeName}")
    
#     # Check if the volume is loaded into the scene
#     if not slicer.mrmlScene.IsNodePresent(volumeNode):
#         print(f"Volume {volumeName} not present in the scene.")
#         continue
    
#     # Retrieve the DICOM Series UID for the current volume
#     series_uid = volumeNode.GetAttribute("DICOM.seriesUID")
    
#     if not series_uid:
#         print(f"Series UID not found for volume: {volumeName}")
#         continue
    
#     # Get the files for this specific series only
#     db = slicer.dicomDatabase
#     fileList = db.filesForSeries(series_instance_uid)
    
#     if not fileList:
#         print(f"No files found for Series {series_instance_uid}.")
#         continue
    
#     # Extract manufacturer from the first file found
#     manufacturer = db.fileValue(fileList[0], "0008,0070") or "Unknown"
#     print(f"Manufacturer for Series {series_instance_uid}: {manufacturer}")
        
#     if manufacturer == "Varian Medical Systems":
#         print("This appears to be a CBCT scan.")
#     else:
#         print("This appears to be a CT scan.")

#detect_points_in_series("8: Unnamed Series", "8: Unnamed Series")