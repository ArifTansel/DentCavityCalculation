
import open3d as o3d
import numpy as np

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
    colors[:, :] = [0.8, 0.8, 0.8]  # Medium gray
    
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