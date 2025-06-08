import numpy as np
import open3d as o3d
import trimesh
from sklearn.neighbors import NearestNeighbors
import scipy
from scipy.spatial import cKDTree

def sort_isthmus_pairs(isthmus_pairs):
    length = len(isthmus_pairs)
    if length<=1:
        return isthmus_pairs
    pivot = isthmus_pairs[length//2][2]
    left = [x for x in isthmus_pairs if x[2] < pivot]
    mid = [x for x in isthmus_pairs if x[2] == pivot]
    right = [x for x in isthmus_pairs if x[2] > pivot]
    
    return sort_isthmus_pairs(left) + mid + sort_isthmus_pairs(right) 


def find_local_maxima_along_axis(vertices, axis=2, k=10):
    """
    Find local maxima points along the specified axis using KNN search.
    """
    

    axis_vals = vertices[:, axis]
    nbrs = NearestNeighbors(n_neighbors=k).fit(vertices)
    _, indices = nbrs.kneighbors(vertices)

    local_max_mask = []
    for i, neighbors in enumerate(indices):
        val = axis_vals[i]
        neighborhood = axis_vals[neighbors]
        if val == np.max(neighborhood):
            local_max_mask.append(True)
        else:
            local_max_mask.append(False)
    return np.where(local_max_mask)[0]


def compute_isthmus_vectors(right_mesh, left_mesh, num_pairs=5):
    """
    Compute shortest distances between local maxima on right and left cavity walls.
    """
    right_vertices = np.asarray(right_mesh.vertices)
    left_vertices = np.asarray(left_mesh.vertices)

    # Step 1: Find local maxima points
    right_maxima_idx = find_local_maxima_along_axis(right_vertices, axis=2, k=10)
    left_maxima_idx = find_local_maxima_along_axis(left_vertices, axis=2, k=10)

    right_maxima = right_vertices[right_maxima_idx]
    left_maxima = left_vertices[left_maxima_idx]

    # Step 2: Build KDTree for fast distance lookup
    left_tree = cKDTree(left_maxima)

    # Step 3: For each right max point, find the closest left max point
    distances, indices = left_tree.query(right_maxima, k=1)
    
    # Step 4: Collect the closest pairs
    pairs = []
    for i, dist in enumerate(distances):
        pairs.append((right_maxima[i], left_maxima[indices[i]], dist))
    
    # Step 5: Sort by distance and select N shortest (isthmus candidates)
    pairs.sort(key=lambda x: x[2])
    top_pairs = pairs[:num_pairs]

    return top_pairs


def find_local_maxima(points, z_threshold=0.2, k=10):
    
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(points)
    _, indices = nbrs.kneighbors(points)

    maxima = []
    for i, neighbors in enumerate(indices):
        if all(points[i][2] >= points[n][2] for n in neighbors if n != i) and points[i][2] > z_threshold:
            maxima.append(points[i])
    return np.array(maxima)




def calculate_point_to_line_distance(point, line_start, line_end):
    point = np.array(point)
    a = np.array(line_start)
    b = np.array(line_end)

    ab = b - a
    t = np.dot(point - a, ab) / np.dot(ab, ab)
    t = np.clip(t, 0.0, 1.0)  # sadece doğru parçası için

    closest_point = a + t * ab
    distance = np.linalg.norm(point - closest_point)
    
    return distance, closest_point



def compute_n_closest_vectors(set_a, set_b, n=8):
    dists = scipy.spatial.distance.cdist(set_a, set_b)
    flat_indices = np.argpartition(dists.flatten(), n)[:n]
    closest_pairs = []

    for idx in flat_indices:
        i, j = np.unravel_index(idx, dists.shape)
        pt_a, pt_b = set_a[i], set_b[j]
        distance = np.linalg.norm(pt_a - pt_b)
        closest_pairs.append((pt_a, pt_b, distance))
    
    # Sort by distance
    return sorted(closest_pairs, key=lambda x: x[2])
 
def is_within_bounds(point, bounds_min, bounds_max):
    return np.all(point >= bounds_min) and np.all(point <= bounds_max)


def visualize_isthmus_filtered(left_points, right_points, n_vectors=12,
                                bounds_min=None, bounds_max=None):
    maxima_left = find_local_maxima(left_points)
    maxima_right = find_local_maxima(right_points)

    all_pairs = compute_n_closest_vectors(maxima_left, maxima_right, n=n_vectors)

    # Filter by midpoint inside bounding box
    filtered_pairs = []
    for start, end, dist in all_pairs:
        midpoint = (start + end) / 2.0
        if bounds_min is not None and bounds_max is not None:
            if not is_within_bounds(midpoint, bounds_min, bounds_max):
                continue
        filtered_pairs.append((start, end, dist))
        mesial_isthmus = filtered_pairs[0]
        distal_isthmus = filtered_pairs[1]
    return mesial_isthmus,distal_isthmus


def calculate_oklidian_length_point(point1 , point2 ): 
    x1 = np.asarray(point1.points)[0]
    x2 = np.asarray(point2.points)[0]

    # Öklidyen mesafe hesapla
    distance = np.linalg.norm(x1 - x2)
    return distance

def split_side_and_get_normal_means(side_mesh):
    """
    Splits a side mesh into right and left walls and returns their mean normal vectors.
    
    Args:
        side_mesh: Open3D triangle mesh representing side walls
        
    Returns:
        right_mesh: The right side wall mesh
        left_mesh: The left side wall mesh
        right_normal_mean: Mean normal vector of the right mesh
        left_normal_mean: Mean normal vector of the left mesh
    """
    # Ensure we have triangle normals
    side_mesh.compute_triangle_normals()
    triangle_normals = np.asarray(side_mesh.triangle_normals)
    
    # Project normals onto the x-y plane
    xy_normals = triangle_normals[:, :2]
    
    # Calculate angles in the x-y plane (in radians)
    angles = np.arctan2(xy_normals[:, 1], xy_normals[:, 0])
    
    # Convert to degrees and shift to 0-360 range
    angles_deg = np.degrees(angles) % 360
    
    # Define right (0-180) and left (180-360) indices
    right_indices = np.where((angles_deg >= 0) & (angles_deg < 180))[0]
    left_indices = np.where((angles_deg >= 180) & (angles_deg < 360))[0]
    
    # Calculate mean normal for right side
    right_normals = triangle_normals[right_indices]
    right_normal_mean = np.mean(right_normals, axis=0)
    # Normalize the mean normal vector
    right_normal_mean = right_normal_mean / np.linalg.norm(right_normal_mean)
    
    # Calculate mean normal for left side
    left_normals = triangle_normals[left_indices]
    left_normal_mean = np.mean(left_normals, axis=0)
    # Normalize the mean normal vector
    left_normal_mean = left_normal_mean / np.linalg.norm(left_normal_mean)
    
    # Create the right mesh
    vertices = np.asarray(side_mesh.vertices)
    triangles = np.asarray(side_mesh.triangles)
    
    right_triangles = triangles[right_indices]
    unique_right_vertices = np.unique(right_triangles.ravel())
    
    vertex_map_right = np.zeros(len(vertices), dtype=int)
    for i, idx in enumerate(unique_right_vertices):
        vertex_map_right[idx] = i
        
    remapped_right_triangles = np.zeros_like(right_triangles)
    for i in range(right_triangles.shape[0]):
        for j in range(3):
            remapped_right_triangles[i, j] = vertex_map_right[right_triangles[i, j]]
            
    right_mesh = o3d.geometry.TriangleMesh()
    right_mesh.vertices = o3d.utility.Vector3dVector(vertices[unique_right_vertices])
    right_mesh.triangles = o3d.utility.Vector3iVector(remapped_right_triangles)
    
    # Create the left mesh
    left_triangles = triangles[left_indices]
    unique_left_vertices = np.unique(left_triangles.ravel())
    
    vertex_map_left = np.zeros(len(vertices), dtype=int)
    for i, idx in enumerate(unique_left_vertices):
        vertex_map_left[idx] = i
        
    remapped_left_triangles = np.zeros_like(left_triangles)
    for i in range(left_triangles.shape[0]):
        for j in range(3):
            remapped_left_triangles[i, j] = vertex_map_left[left_triangles[i, j]]
            
    left_mesh = o3d.geometry.TriangleMesh()
    left_mesh.vertices = o3d.utility.Vector3dVector(vertices[unique_left_vertices])
    left_mesh.triangles = o3d.utility.Vector3iVector(remapped_left_triangles)
    
    return right_mesh, left_mesh, right_normal_mean, left_normal_mean

def calculate_roughness(mesh):
    mesh_vertices = np.asarray(mesh.vertices)
    mesh_faces = np.asarray(mesh.triangles)
    tri_mesh = trimesh.Trimesh(vertices=mesh_vertices, faces=mesh_faces, process=False)
    normals = tri_mesh.face_normals
    adj = tri_mesh.face_adjacency

    # Compute angle between adjacent face normals
    dot = np.einsum('ij,ij->i', normals[adj[:, 0]], normals[adj[:, 1]])
    angles = np.arccos(np.clip(dot, -1.0, 1.0)) # Mean angle
    angles_deg = np.degrees(angles) # roughness
    return angles_deg

def calculate_distal_mesial_marginal_ridge_width():
    pass