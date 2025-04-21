import numpy as np
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
import open3d as o3d

### The function that extracts the cavity from the tooth model
### The function that extracts the cavity bottom 
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