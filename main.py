## Imports
import trimesh
import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
import os
import argparse
import json

from utils import extract_largest_cavity, extract_cavity_parts ,extract_top_percentage
from utils import show_mesh_dimensions_with_cylinders, visualize_roughness, create_cylinder_between_points
from utils import split_side_and_get_normal_means, calculate_roughness, discrete_mean_curvature_measure_gpu


BOTTOM_THRESHOLD_PERCENTAGE=0.4
MEAN_CURVATURE_RADİUS=2

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
mean_curvature = trimesh.curvature.discrete_mean_curvature_measure(mesh_trimesh, mesh_trimesh.vertices, MEAN_CURVATURE_RADİUS)

### kavitenin seçilmesi 
cavity_indices = np.where(mean_curvature < 0.4)[0]  # Select all vertices with negative curvature
largest_cavity_mesh, largest_cavity_indices = extract_largest_cavity(vertices, faces, cavity_indices)
cavity_vertices= np.asarray(largest_cavity_mesh.vertices)

top_tooth_mesh = extract_top_percentage(tooth_o3d) 
    
tooth_dimension_cylinder_meshes, tooth_width, tooth_length = show_mesh_dimensions_with_cylinders(top_tooth_mesh)
cavity_dimension_cylinder_meshes, cavity_width, cavity_length = show_mesh_dimensions_with_cylinders(largest_cavity_mesh)


### Kavitenin alt kısmının seçilmesi ve kavite yüksekliğinin hesaplanması
side_bottom, cavity_bottom = extract_cavity_parts(largest_cavity_mesh, BOTTOM_THRESHOLD_PERCENTAGE)

# kavite altının z eksenindeki ortalamasını al
bottom_vertices = np.asarray(cavity_bottom.vertices)
bottom_z_values = bottom_vertices[:, 2]
min_z_mean = np.mean(bottom_z_values) 

# kavite alanının en üstü 
max_z = np.max(cavity_vertices[:, 2])
cavity_depth = max_z - min_z_mean  # Derinlik (Z eksenindeki fark)

outline_indices = np.where((mean_curvature > 3.0))[0]
roughness = calculate_roughness(cavity_bottom)
colored_roughness = visualize_roughness(cavity_bottom,tooth_o3d)

cavity_centroid = np.mean(cavity_vertices, axis=0)
min_z_point = [cavity_centroid[0], cavity_centroid[1], min_z_mean]
max_z_point = [cavity_centroid[0], cavity_centroid[1], max_z]
cavity_depth_mesh = create_cylinder_between_points(min_z_point, max_z_point)

right_mesh, left_mesh, right_normal_mean, left_normal_mean = split_side_and_get_normal_means(side_bottom)

cavity_bottom.compute_vertex_normals()
bottom_normal_mean = np.mean(np.asarray(cavity_bottom.vertex_normals),axis=0)

right_angle = np.dot(right_normal_mean, bottom_normal_mean)
left_angle = np.dot(left_normal_mean, bottom_normal_mean)


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

# Calculate score
score = 0
if cavity_width>=2.7 and cavity_width<=3.3:
     score +=10
elif (cavity_width<=2.69 and cavity_width>=2.5) or (cavity_width>=3.31 and cavity_width<=3.5):
    score += 5
elif cavity_width<2.5 or cavity_width>3.5:
    pass

if cavity_length>=7.1 and cavity_length<=8.29:
     score +=10
elif cavity_length>=6.6 and cavity_length<=7.00:
    score += 5
elif cavity_length>8.3:
    pass

if cavity_depth>=2.5 and cavity_depth<=3.0:
     score +=10
elif (cavity_depth<=2.49 and cavity_depth>=2.0) or (cavity_depth>=3.01 and cavity_depth<=3.39):
    score += 5
elif cavity_depth<2.0 or cavity_depth>3.5:
    pass

std_roughness = np.std(roughness)

if std_roughness>=0 and std_roughness<=10.0:
     score +=10
elif std_roughness>=10.01 and std_roughness<=40.00:
    score += 5
elif std_roughness>40.00:
    pass


#Stdout to return
#tooth_dimension_cylinder_meshes
# JSON içerisindeki veriler (örnek değerler kullanılmaktadır)
data = {
    
    "right_angle": right_angle,
    "left_angle": left_angle,
    "cavity_depth": cavity_depth,
    "roughness":  np.std(roughness),
    "m_d_length_ratio" : m_d_length_ratio ,
    "b_l_length_ratio" : b_l_length_ratio,
    "score" : score
    
}
# JSON verisini string'e dönüştürüp yazdırma api stdout olarak alacak
print(json.dumps(data))

# JSON dosyasına yazma
with open(f"output/{mkdir}/data.json", "w") as f:
    json.dump(data, f, indent=4)