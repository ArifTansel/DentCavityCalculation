## Imports
import trimesh
import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import os
import argparse
import json
### Load and compute the mean curvature
# Load the tooth STL model using Trimesh
parser = argparse.ArgumentParser(description="Teeth path")
parser.add_argument("--path", type=str, help="Teeth path")

args = parser.parse_args()
mesh_trimesh = trimesh.load_mesh(f"StudentTeeth/{args.path}.stl")

# Get vertices, faces, and normals
vertices = np.array(mesh_trimesh.vertices)
faces = np.array(mesh_trimesh.faces)
normals = np.array(mesh_trimesh.vertex_normals)
tooth_o3d = o3d.geometry.TriangleMesh()
tooth_o3d.vertices = o3d.utility.Vector3dVector(vertices)
tooth_o3d.triangles = o3d.utility.Vector3iVector(faces)
tooth_o3d.compute_vertex_normals()# Convert full tooth to Open3D mesh

tooth_o3d.paint_uniform_color([0.8, 0.8, 0.8])  # light gray


# Compute Mean Curvature using Trimesh
mean_curvature = trimesh.curvature.discrete_mean_curvature_measure(mesh_trimesh, mesh_trimesh.vertices, radius=2)

### The function that extracts the cavity from the tooth model
def extract_largest_cavity(vertices, faces, cavity_indices):
    # Get unique cavity indices
    unique_cavity_indices = np.unique(cavity_indices)
    
    # Find faces that have all vertices in cavity_indices
    cavity_face_mask = np.isin(faces.ravel(), unique_cavity_indices).reshape(faces.shape)
    cavity_face_indices = np.where(np.all(cavity_face_mask, axis=1))[0]
    cavity_faces = faces[cavity_face_indices]
    
    # Create adjacency matrix for connected component analysis
    edges = set()
    for face in cavity_faces:
        edges.add((face[0], face[1]))
        edges.add((face[1], face[2]))
        edges.add((face[2], face[0]))
    
    # Create sparse adjacency matrix
    row, col = zip(*edges)
    row = np.array(row)
    col = np.array(col)
    data = np.ones_like(row)
    
    # Create sparse matrix with size equal to total vertices 
    # (will be pruned to cavity vertices later)
    adj_matrix = csr_matrix((data, (row, col)), shape=(vertices.shape[0], vertices.shape[0]))
    
    # Find connected components
    n_components, labels = connected_components(csgraph=adj_matrix, directed=False)
    
    # Count vertices in each component
    component_sizes = np.zeros(n_components, dtype=int)
    for label in labels[unique_cavity_indices]:
        component_sizes[label] += 1
    
    # Find the largest component
    largest_component = np.argmax(component_sizes)
    
    # Get vertices from the largest component
    largest_cavity_indices = np.where(labels == largest_component)[0]
    largest_cavity_indices = np.intersect1d(largest_cavity_indices, unique_cavity_indices)
    
    # Create index mapping for new mesh
    index_map = np.zeros(len(vertices), dtype=int)
    for i, idx in enumerate(largest_cavity_indices):
        index_map[idx] = i
    
    # Get faces for largest component
    largest_face_mask = np.isin(cavity_faces.ravel(), largest_cavity_indices).reshape(cavity_faces.shape)
    largest_face_indices = np.where(np.all(largest_face_mask, axis=1))[0]
    largest_cavity_faces = cavity_faces[largest_face_indices]
    
    # Remap face indices
    remapped_faces = np.zeros_like(largest_cavity_faces)
    for i in range(largest_cavity_faces.shape[0]):
        for j in range(3):
            remapped_faces[i, j] = index_map[largest_cavity_faces[i, j]]
    
    # Create and return the largest cavity mesh
    largest_cavity_mesh = o3d.geometry.TriangleMesh()
    largest_cavity_mesh.vertices = o3d.utility.Vector3dVector(vertices[largest_cavity_indices])
    largest_cavity_mesh.triangles = o3d.utility.Vector3iVector(remapped_faces)
    largest_cavity_mesh.compute_vertex_normals()
    
    # Set color for visualization
    cavity_colors = np.ones((len(largest_cavity_indices), 3)) * [0, 1, 0]  # Green
    largest_cavity_mesh.vertex_colors = o3d.utility.Vector3dVector(cavity_colors)
    
    return largest_cavity_mesh, largest_cavity_indices
### The function that extracts the cavity bottom 

def extract_cavity_parts(largest_cavity_mesh, bottom_threshold_percentage=0.1):
    # Get vertices and triangles from the mesh
    cavity_vertices = np.asarray(largest_cavity_mesh.vertices)
    cavity_triangles = np.asarray(largest_cavity_mesh.triangles)
    
    # Calculate z-range
    min_z = np.min(cavity_vertices[:, 2])
    max_z = np.max(cavity_vertices[:, 2])
    z_range = max_z - min_z
    
    # Define a threshold for what constitutes the "bottom"
    z_threshold = min_z + z_range * bottom_threshold_percentage
    
    # Find vertices that are in the bottom region
    bottom_vertex_mask = cavity_vertices[:, 2] <= z_threshold
    bottom_vertex_indices = np.where(bottom_vertex_mask)[0]
    
    # Find triangles where all three vertices are in the bottom region
    bottom_triangles_mask = np.isin(cavity_triangles.ravel(), bottom_vertex_indices).reshape(cavity_triangles.shape)
    bottom_triangle_indices = np.where(np.all(bottom_triangles_mask, axis=1))[0]
    
    # Find triangles that are NOT in the bottom (side triangles)
    total_triangles = len(cavity_triangles)
    side_triangle_indices = np.setdiff1d(np.arange(total_triangles), bottom_triangle_indices)
    
    result = {}
    
    # Process bottom mesh
    if len(bottom_triangle_indices) > 0:
        # Create a new mesh for the bottom surface
        bottom_triangles = cavity_triangles[bottom_triangle_indices]
        
        # Get unique vertices used in the bottom triangles
        unique_vertices = np.unique(bottom_triangles.ravel())
        
        # Create index mapping
        index_map = np.zeros(len(cavity_vertices), dtype=int)
        for i, idx in enumerate(unique_vertices):
            index_map[idx] = i
        
        # Remap triangle indices
        remapped_triangles = np.zeros_like(bottom_triangles)
        for i in range(bottom_triangles.shape[0]):
            for j in range(3):
                remapped_triangles[i, j] = index_map[bottom_triangles[i, j]]
        
        # Create bottom surface mesh
        bottom_mesh = o3d.geometry.TriangleMesh()
        bottom_mesh.vertices = o3d.utility.Vector3dVector(cavity_vertices[unique_vertices])
        bottom_mesh.triangles = o3d.utility.Vector3iVector(remapped_triangles)
        bottom_mesh.compute_vertex_normals()
        
        # Set color for visualization
        bottom_colors = np.ones((len(unique_vertices), 3)) * [0, 0, 1]  # Blue for bottom surface
        bottom_mesh.vertex_colors = o3d.utility.Vector3dVector(bottom_colors)
        
    
    # Process side mesh
    if len(side_triangle_indices) > 0:
        # Create a new mesh for the side surface
        side_triangles = cavity_triangles[side_triangle_indices]
        
        # Get unique vertices used in the side triangles
        unique_vertices = np.unique(side_triangles.ravel())
        
        # Create index mapping
        index_map = np.zeros(len(cavity_vertices), dtype=int)
        for i, idx in enumerate(unique_vertices):
            index_map[idx] = i
        
        # Remap triangle indices
        remapped_triangles = np.zeros_like(side_triangles)
        for i in range(side_triangles.shape[0]):
            for j in range(3):
                remapped_triangles[i, j] = index_map[side_triangles[i, j]]
        
        # Create side surface mesh
        side_mesh = o3d.geometry.TriangleMesh()
        side_mesh.vertices = o3d.utility.Vector3dVector(cavity_vertices[unique_vertices])
        side_mesh.triangles = o3d.utility.Vector3iVector(remapped_triangles)
        side_mesh.compute_vertex_normals()
        
        # Set color for visualization
        side_colors = np.ones((len(unique_vertices), 3)) * [1, 0, 0]  # Red for side surface
        side_mesh.vertex_colors = o3d.utility.Vector3dVector(side_colors)

    
    return side_mesh , bottom_mesh
### The function that calulates the cavity_bottom roughness
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

### kavitenin seçilmesi 
cavity_indices = np.where(mean_curvature < 0.4)[0]  # Select all vertices with negative curvature
largest_cavity_mesh, largest_cavity_indices = extract_largest_cavity(vertices, faces, cavity_indices)
cavity_vertices= np.asarray(largest_cavity_mesh.vertices)



def extract_top_percentage(cavity_mesh, percentage=28.0):
    """
    Z ekseninde yukarıdan belirtilen yüzde kadarını alarak yeni bir mesh döndürür.
    
    Args:
        cavity_mesh: İşlenecek Open3D mesh
        percentage: Yukarıdan alınacak yüzde (varsayılan %1)
        
    Returns:
        Yukarıdan belirtilen yüzde kadar kesilmiş yeni bir O3D mesh
    """
    # Mesh'i kopyala (orijinal mesh'i değiştirmemek için)
    clipped_mesh = o3d.geometry.TriangleMesh(cavity_mesh)
    
    # Vertexleri numpy array olarak al
    vertices = np.asarray(clipped_mesh.vertices)
    triangles = np.asarray(clipped_mesh.triangles)
    

    
    # Z eksenindeki min ve max değerleri bul
    min_z = np.min(vertices[:, 2])
    max_z = np.max(vertices[:, 2])
    
    # Z yüksekliğini hesapla
    height = max_z - min_z
    
    # Z ekseni boyunca kesim noktasını belirle (yukarıdan %1'lik dilim için)
    z_threshold = max_z - (height * percentage / 100.0)
    
    # Sadece z_threshold'dan büyük z değerlerine sahip vertexleri seç
    mask = vertices[:, 2] >= z_threshold
    

    
    # Yeni mesh oluştur
    result_mesh = o3d.geometry.TriangleMesh()
    
    # Yeni mesh için filtreleme yaklaşımı:
    # 1. Hangi üçgenlerin tamamen eşik değerinin üzerinde olduğunu belirle
    valid_triangles = []
    for triangle in triangles:
        v1, v2, v3 = triangle
        # Üçgenin tüm köşeleri eşik değerinden yukarıdaysa, bu üçgeni dahil et
        if vertices[v1, 2] >= z_threshold and vertices[v2, 2] >= z_threshold and vertices[v3, 2] >= z_threshold:
            valid_triangles.append(triangle)
    

    
    # Eşik üzerindeki vertexlerin indekslerini ve yeni indeks haritalamasını oluştur
    valid_vertices_indices = np.where(mask)[0]
    index_map = {old_idx: new_idx for new_idx, old_idx in enumerate(valid_vertices_indices)}
    
    # Yeni vertexleri ekle
    new_vertices = vertices[mask]
    result_mesh.vertices = o3d.utility.Vector3dVector(new_vertices)
    
    # Yeni üçgenleri ekle (indeks haritalamasını kullanarak)
    new_triangles = []
    for triangle in valid_triangles:
        try:
            new_triangle = [index_map[idx] for idx in triangle]
            new_triangles.append(new_triangle)
        except KeyError:
            # Eğer üçgen tamamen eşik üzerinde değilse, KeyError olabilir
            continue
    
    result_mesh.triangles = o3d.utility.Vector3iVector(new_triangles)
    
    # Vertex normallerini yeniden hesapla
    result_mesh.compute_vertex_normals()
    
    return result_mesh
top_tooth_mesh = extract_top_percentage(tooth_o3d) 
# o3d.visualization.draw_geometries([top_tooth_mesh]) 

### Experimental : Get height and width of the largest_cavity

def show_mesh_dimensions_with_cylinders(mesh):
    # Mesh'in vertex'lerini al
    vertices = np.asarray(mesh.vertices)
    
    # En düşük ve en yüksek noktaları bul (x, y, z eksenleri için)
    min_bound = np.min(vertices, axis=0)
    max_bound = np.max(vertices, axis=0)
    
    # Genişlik (x ekseni), uzunluk (y ekseni) ve yükseklik (z ekseni) hesapla
    width = max_bound[0] - min_bound[0]
    length = max_bound[1] - min_bound[1]
    height = max_bound[2] - min_bound[2]
    
    # Mesh'in merkez noktasını hesapla
    center = (min_bound + max_bound) / 2
    
    # Genişliği (x ekseni) temsil eden silindir oluştur
    width_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=height/30,  # Silindirin kalınlığı - görsellik için mesh yüksekliğine göre ayarlanabilir
        height=width       # Silindirin uzunluğu mesh genişliği kadar
    )
    
    # X ekseni boyunca hizalamak için silindiri döndür
    width_cylinder.rotate(
        R=np.array([
            [0, 0, 1],
            [0, 1, 0],
            [-1, 0, 0]
        ]),
        center=np.array([0, 0, 0])
    )
    
    # Silindiri merkeze hizala
    width_cylinder.translate(center - np.array([0, 0, 0]))
    
    # Genişlik silindirini kırmızı renkle boya (X ekseni için)
    width_cylinder.paint_uniform_color([1, 0, 0])  # Kırmızı
    
    # Uzunluğu (y ekseni) temsil eden silindir oluştur
    length_cylinder = o3d.geometry.TriangleMesh.create_cylinder(
        radius=height/30,  # Silindirin kalınlığı
        height=length      # Silindirin uzunluğu mesh uzunluğu kadar
    )
    
    # Y ekseni boyunca hizalamak için silindiri döndür
    length_cylinder.rotate(
        R=np.array([
            [1, 0, 0],
            [0, 0, -1],
            [0, 1, 0]
        ]),
        center=np.array([0, 0, 0])
    )
    
    # Silindiri merkeze hizala
    length_cylinder.translate(center - np.array([0, 0, 0]))
    
    # Uzunluk silindirini yeşil renkle boya (Y ekseni için)
    length_cylinder.paint_uniform_color([0, 1, 0])  # Yeşil
    
    
    # Genişlik ve uzunluk silindirlerini birleştirerek döndür
    result = width_cylinder + length_cylinder
    # o3d.visualization.draw_geometries([result,mesh])
    return result , width, length
    
tooth_dimension_cylinder_meshes, tooth_width, tooth_length = show_mesh_dimensions_with_cylinders(top_tooth_mesh)
cavity_dimension_cylinder_meshes, cavity_width, cavity_length = show_mesh_dimensions_with_cylinders(largest_cavity_mesh)


### Kavitenin alt kısmının seçilmesi ve kavite yüksekliğinin hesaplanması
side_bottom, cavity_bottom = extract_cavity_parts(largest_cavity_mesh, bottom_threshold_percentage=0.4)

# kavite altının z eksenindeki ortalamasını al
bottom_vertices = np.asarray(cavity_bottom.vertices)
bottom_z_values = bottom_vertices[:, 2]
min_z_mean = np.mean(bottom_z_values) 

# kavite alanının en üstü 
max_z = np.max(cavity_vertices[:, 2])
cavity_depth = max_z - min_z_mean  # Derinlik (Z eksenindeki fark)


### Roughness Görselleştirme
def visualize_roughness(cavity_bottom, full_tooth_mesh):
    # Cavity'den hesaplama yap
    cavity_vertices = np.asarray(cavity_bottom.vertices)
    z_values = cavity_vertices[:, 2]
    z_mean = np.mean(z_values)
    
    # Cavity için sapmaları hesapla
    deviations = np.abs(z_values - z_mean)
    max_deviation = np.max(deviations)
    
    # Cavity için normalize edilmiş sapmaları hesapla
    cavity_normalized_deviations = deviations / max_deviation if max_deviation > 0 else np.zeros_like(deviations)
    
    # Tüm diş modeli için renk arrayi oluştur
    full_vertices = np.asarray(full_tooth_mesh.vertices)
    colors = np.zeros((len(full_vertices), 3))
    
    # Önce tüm dişi mavi yap (düz alanlar)
    colors[:, :] = [0.5, 0.5, 0.5]  # Medium gray
    
    # Cavity vertex'lerinin tam diş modelindeki indekslerini bul ve bu pozisyonları renklendir
    # NOT: Bu kısım, cavity_bottom vertex'lerinin full_tooth_mesh içindeki aynı koordinatlara sahip vertex'leri bulma mantığına dayanır
    
    # Basit yaklaşım: Her cavity vertex'i için en yakın full_tooth vertex'ini bul
    from scipy.spatial import KDTree
    
    # KDTree oluştur tüm diş vertexleri için
    tree = KDTree(full_vertices)
    
    # Her cavity vertex'i için en yakın diş vertex'ini bul
    _, indices = tree.query(cavity_vertices)
    
    # Cavity vertex'lerinin renklerini tüm diş modelindeki karşılık gelen vertex'lere uygula
    colors[indices, 0] = cavity_normalized_deviations  # Kırmızı kanal
    colors[indices, 2] = 1.0 - cavity_normalized_deviations  # Mavi kanal
    
    # Renkleri tüm diş mesh'ine uygula
    colored_roughness_mesh = o3d.geometry.TriangleMesh()
    colored_roughness_mesh.vertices = full_tooth_mesh.vertices
    colored_roughness_mesh.triangles = full_tooth_mesh.triangles
    colored_roughness_mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    colored_roughness_mesh.compute_vertex_normals()
    
    return colored_roughness_mesh
### Roughness Hesaplanması

outline_indices = np.where((mean_curvature > 3.0))[0]
roughness = calculate_roughness(cavity_bottom)
colored_roughness = visualize_roughness(cavity_bottom,tooth_o3d)
def create_cylinder_between_points(point1, point2, radius=0.01, resolution=20, color=None):
    """
    Create a cylinder mesh between two points.
    
    Args:
        point1: Starting point as [x, y, z]
        point2: Ending point as [x, y, z]
        radius: Radius of the cylinder
        resolution: Number of segments for the cylinder
        color: RGB color as [r, g, b] where each value is between 0 and 1
    
    Returns:
        cylinder_mesh: An Open3D mesh representing the cylinder
    """
    # Convert points to numpy arrays
    point1 = np.asarray(point1)
    point2 = np.asarray(point2)
    
    # Calculate the direction vector from point1 to point2
    direction = point2 - point1
    length = np.linalg.norm(direction)
    
    # Create a cylinder along the Z-axis
    cylinder = o3d.geometry.TriangleMesh.create_cylinder(radius=radius, height=length, resolution=resolution)
    
    # Compute the rotation to align with the direction vector
    # First, we need to find the rotation axis and angle
    z_axis = np.array([0, 0, 1])
    direction_normalized = direction / length
    
    # Compute the rotation axis via cross product
    rotation_axis = np.cross(z_axis, direction_normalized)
    
    # If points are aligned along Z-axis, rotation axis will be zero
    if np.linalg.norm(rotation_axis) < 1e-6:
        # Check if direction is parallel or anti-parallel to z_axis
        if direction_normalized[2] > 0:
            # Parallel - no rotation needed
            rotation_matrix = np.eye(3)
        else:
            # Anti-parallel - rotate 180 degrees around X-axis
            rotation_matrix = np.array([
                [1, 0, 0],
                [0, -1, 0],
                [0, 0, -1]
            ])
    else:
        # Normalize rotation axis
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        
        # Compute rotation angle using dot product
        cos_angle = np.dot(z_axis, direction_normalized)
        angle = np.arccos(cos_angle)
        
        # Convert axis-angle to rotation matrix using Rodrigues' formula
        cross_matrix = np.array([
            [0, -rotation_axis[2], rotation_axis[1]],
            [rotation_axis[2], 0, -rotation_axis[0]],
            [-rotation_axis[1], rotation_axis[0], 0]
        ])
        rotation_matrix = np.eye(3) + np.sin(angle) * cross_matrix + (1 - np.cos(angle)) * (cross_matrix @ cross_matrix)
    
    # Rotate the cylinder to align with the direction
    cylinder.rotate(rotation_matrix, center=np.array([0, 0, 0]))
    
    # Translate the cylinder to start at point1
    cylinder.translate(point1 + direction_normalized * (length / 2))
    
    # Set the color if provided
    if color is not None:
        cylinder.paint_uniform_color(color)
    
    return cylinder
### Görüntüleme
# **Çizgiyi tanımlama**
cavity_centroid = np.mean(cavity_vertices, axis=0)
min_z_point = [cavity_centroid[0], cavity_centroid[1], min_z_mean]
max_z_point = [cavity_centroid[0], cavity_centroid[1], max_z]
cavity_depth_mesh = create_cylinder_between_points(min_z_point, max_z_point)

# print("cavity_depth : ",cavity_depth)
o3d.io.write_triangle_mesh("output/colored_roughness.ply", colored_roughness , write_vertex_colors=True)

#colored_roughness, cavity_bottom, line_set, largest_cavity_mesh, tooth_o3d
# o3d.visualization.draw_geometries([colored_roughness,],mesh_show_back_face=True)
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
right_mesh, left_mesh, right_normal_mean, left_normal_mean = split_side_and_get_normal_means(side_bottom)

cavity_bottom.compute_vertex_normals()
bottom_normal_mean = np.mean(np.asarray(cavity_bottom.vertex_normals),axis=0)

right_angle = np.dot(right_normal_mean, bottom_normal_mean)
left_angle = np.dot(left_normal_mean, bottom_normal_mean)


##print("right_angle : " ,right_angle)
##print("left_angle : " ,left_angle)

# export meshes to stl files
#ALL MESHES : colored_roughness, cavity_bottom, line_set, largest_cavity_mesh, tooth_o3d  ,cylinder_mesh
mkdir = args.path.split(".")[0]
os.makedirs(f"output/{mkdir}",exist_ok=True)
o3d.io.write_triangle_mesh(f"output/{mkdir}/colored_roughness.ply", colored_roughness , write_vertex_colors=True)
o3d.io.write_triangle_mesh(f"output/{mkdir}/cavity_bottom.ply", cavity_bottom)
o3d.io.write_triangle_mesh(f"output/{mkdir}/largest_cavity_mesh.ply", largest_cavity_mesh)
o3d.io.write_triangle_mesh(f"output/{mkdir}/tooth_o3d.ply", tooth_o3d)
o3d.io.write_triangle_mesh(f"output/{mkdir}/cavity_depth_mesh.ply", cavity_depth_mesh)
o3d.io.write_triangle_mesh(f"output/{mkdir}/tooth_dimension_cylinder_meshes.ply", tooth_dimension_cylinder_meshes)
o3d.io.write_triangle_mesh(f"output/{mkdir}/cavity_dimension_cylinder_meshes.ply", cavity_dimension_cylinder_meshes)

b_l_length_ratio = (tooth_width - cavity_width) / tooth_width 
m_d_length_ratio = (tooth_length - cavity_length) / tooth_length 
#Stdout to return
#tooth_dimension_cylinder_meshes
# JSON içerisindeki veriler (örnek değerler kullanılmaktadır)
data = {
    "right_angle": right_angle,
    "left_angle": left_angle,
    "cavity_depth": cavity_depth,
    "roughness":  np.std(roughness),
    "m_d_length_ratio" : m_d_length_ratio ,
    "b_l_length_ratio" : b_l_length_ratio

}

# JSON verisini string'e dönüştürüp yazdırma
print(json.dumps(data))


# o3d.visualization.draw_geometries([right_mesh])

