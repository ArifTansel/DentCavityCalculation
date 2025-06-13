## Imports
import trimesh
import open3d as o3d
import numpy as np
from sklearn.cluster import KMeans
import os
import argparse
import json
from utils import discrete_mean_curvature_measure_gpu
from utils import extract_largest_cavity, extract_cavity_parts ,extract_top_percentage
from utils import show_mesh_dimensions_with_cylinders, visualize_roughness, create_cylinder_between_points
from utils import split_side_and_get_normal_means, calculate_roughness ,calculate_point_to_line_distance, get_top_right_edge_midpoint_pcd
from utils import   trim_mesh_by_percent, compute_isthmus_vectors ,sort_isthmus_pairs, visualize_mesial_distal_isthmuses
import mysql.connector
from sklearn.decomposition import PCA

BOTTOM_THRESHOLD_PERCENTAGE=0.4
MEAN_CURVATURE_RADİUS=2
BASE_DIR = "output"
### database infos
HOST= "localhost"
USER = "root"
PASSWORD = "patatoes"
DATABASE = "cavity_analysis_db"
#TODO Z up a göre çalııyor düzelt Y-up yap onu 


### Load and compute the mean curvature
# Load the tooth STL model using Trimesh
parser = argparse.ArgumentParser(description="studentId for path")
parser.add_argument("--studentId", type=str, help="studentId")
args = parser.parse_args()
mesh_trimesh = trimesh.load_mesh(f"StudentTeeth/{args.studentId}.stl")

# X ekseni etrafında -90 derece (radyan cinsinden -π/2) döndürme matrisi
angle_rad = np.pi / 2
rotation_matrix = trimesh.transformations.rotation_matrix(
    angle_rad, [1, 0, 0], point=mesh_trimesh.centroid  # mesh merkezinden döndür
)

# Mesh'i döndür
mesh_trimesh.apply_transform(rotation_matrix)
#################TODO utils içerisinde bir FONKSİYONA AL
    
mesh_source = o3d.io.read_triangle_mesh("input/Master.stl")
mesh_aligned = o3d.io.read_triangle_mesh(f"StudentTeeth/{args.studentId}.stl")

pcd_source = mesh_source.sample_points_uniformly(50000)
pcd_target = mesh_aligned.sample_points_uniformly(50000)


pcd_source.estimate_normals()
pcd_target.estimate_normals()

# Align using ICP (Point-to-Plane)
reg = o3d.pipelines.registration.registration_icp(
    pcd_source, pcd_target, 1.5, np.eye(4),
    o3d.pipelines.registration.TransformationEstimationPointToPlane()
)

# Apply transformation to source mesh
mesh_source.transform(reg.transformation)

# Color code the source mesh based on distance
target_pcd = mesh_aligned.sample_points_uniformly(100000)
kdtree = o3d.geometry.KDTreeFlann(target_pcd)
distances = [np.sqrt(kdtree.search_knn_vector_3d(v, 1)[2][0]) for v in mesh_source.vertices]
distances = np.array(distances)

# Binary red/green coloring
threshold = 0.12
colors = np.zeros((len(distances), 3))
colors[distances < threshold] = [0, 1, 0]
colors[distances >= threshold] = [1, 0, 0]
mesh_source.vertex_colors = o3d.utility.Vector3dVector(colors)

# Critical: Compute normals to enable shading
mesh_source.compute_vertex_normals()
mesh_aligned.paint_uniform_color([0.7, 0.7, 0.7])
mesh_aligned.compute_vertex_normals()
################# FONKSİYONA AL

# # Get vertices, faces, and normals
# old_vertices = np.array(mesh_trimesh.vertices)
# old_faces = np.array(mesh_trimesh.faces)
# old_normals = np.array(mesh_trimesh.vertex_normals)

# old_tooth_o3d = o3d.geometry.TriangleMesh()
# old_tooth_o3d.vertices = o3d.utility.Vector3dVector(old_vertices)
# old_tooth_o3d.triangles = o3d.utility.Vector3iVector(old_faces)
# old_tooth_o3d.compute_vertex_normals()# Convert full tooth to Open3D mesh

# old_tooth_o3d.paint_uniform_color([0.8, 0.8, 0.8])  # light gray


# ### Dişi üst kısmının OBB sine göre rotate et 
# num_vertices = old_vertices.shape[0]
# num_samples = int(num_vertices * 0.2)

# # Sort vertices by Z-coordinate (descending order)
# sorted_indices = np.argsort(old_vertices[:, 2])[::-1]  # Use `1` for Y-axis or `0` for X-axis
# top_indices = sorted_indices[:num_samples]  # Select top 10%

# # Extract the top points
# top_points = old_vertices[top_indices]

# # Create a point cloud object
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(top_points)

# # Oriented Bounding Box (OBB) - if you want tighter fit but possibly rotated
# obb = pcd.get_oriented_bounding_box()
# obb.color = (0, 1, 0)  # Green

# R = obb.R              # Rotation matrix (3x3)
# center = obb.center    # Center of OBB
# T_translate_to_origin = np.eye(4)
# T_translate_to_origin[:3, 3] = -center

# T_translate_back = np.eye(4)
# T_translate_back[:3, 3] = center
# # Create rotation matrix in 4x4
# T_rotate = np.eye(4)
# T_rotate[:3, :3] = R  # your rotation matrix
# T_final = T_translate_back @ T_rotate @ T_translate_to_origin

# mesh_trimesh.apply_transform(T_final)
# #rotate tooth_o3d 


vertices = np.array(mesh_trimesh.vertices)
faces = np.array(mesh_trimesh.faces)
normals = np.array(mesh_trimesh.vertex_normals)

tooth_o3d = o3d.geometry.TriangleMesh()
tooth_o3d.vertices = o3d.utility.Vector3dVector(vertices)
tooth_o3d.triangles = o3d.utility.Vector3iVector(faces)
tooth_o3d.compute_vertex_normals()# Convert full tooth to Open3D mesh

tooth_o3d.paint_uniform_color([0.8, 0.8, 0.8])  # light gray

##################

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
min_z = np.min(bottom_z_values) 

# kavite alanının en üstü 
max_z = np.max(cavity_vertices[:, 2])
cavity_depth = max_z - min_z  # Derinlik (Z eksenindeki fark)

outline_indices = np.where((mean_curvature > 3.0))[0]
roughness = calculate_roughness(cavity_bottom)
colored_roughness = visualize_roughness(roughness,cavity_bottom,tooth_o3d)

cavity_centroid = np.mean(cavity_vertices, axis=0)
min_z_point = [cavity_centroid[0], cavity_centroid[1], min_z]
max_z_point = [cavity_centroid[0], cavity_centroid[1], max_z]
cavity_depth_mesh = create_cylinder_between_points(min_z_point, max_z_point)

right_mesh, left_mesh, right_normal_mean, left_normal_mean = split_side_and_get_normal_means(side_bottom)


trimmed_right_mesh = trim_mesh_by_percent(right_mesh,trim_percent=0.20)
trimmed_left_mesh = trim_mesh_by_percent(left_mesh,trim_percent=0.20)

isthmus_pairs = compute_isthmus_vectors(trimmed_right_mesh, trimmed_left_mesh,num_pairs=10)
sorted_isthmus_pairs = sort_isthmus_pairs(isthmus_pairs=isthmus_pairs)

points, _ = trimesh.sample.sample_surface(mesh_trimesh, 3000)

# Apply PCA to find mesial-distal orientation
pca = PCA(n_components=3)
pca.fit(points)
pc1 = pca.components_[0]  # Principal axis (first component)

distal_isthmus, mesial_isthmus = visualize_mesial_distal_isthmuses(sorted_isthmus_pairs, pc1, tooth_o3d)

distal_isthmus_width = distal_isthmus[2]
mesial_isthmus_width = mesial_isthmus[2]

import copy

my_list = []
my_secondary_list = []
for i in range(2):
    for j in range(3):
        my_secondary_list.append(mesial_isthmus[i][j])
    my_list.append(my_secondary_list)
    my_secondary_list = []
mesial_isthmus_points = copy.deepcopy(my_list)
my_list = []
for i in range(2):
    for j in range(3):
        my_secondary_list.append(distal_isthmus[i][j])
    my_list.append(my_secondary_list)
    my_secondary_list = []
distal_isthmus_points = copy.deepcopy(my_list)
    


distal_isthmus_cylinder = create_cylinder_between_points(distal_isthmus_points[0],distal_isthmus_points[1]) # color değişkeni ile renk verebilirsin 
mesial_isthmus_cylinder = create_cylinder_between_points(mesial_isthmus_points[0],mesial_isthmus_points[1])


cavity_bottom.compute_vertex_normals()
bottom_normal_mean = np.mean(np.asarray(cavity_bottom.vertex_normals),axis=0)

right_angle = np.dot(right_normal_mean, bottom_normal_mean)
left_angle = np.dot(left_normal_mean, bottom_normal_mean)


## calculate marginal ridge widths 
# outer_mesial_point = get_highest_point_near_mid_y(tooth_o3d , 0 , mesial=1) 
# cavity_mesial_point = get_highest_point_near_mid_y(largest_cavity_mesh , 0 , mesial=1) 
# outer_distal_point = get_highest_point_near_mid_y(tooth_o3d , 0 , mesial=-1) 
# cavity_distal_point = get_highest_point_near_mid_y(largest_cavity_mesh , 0 , mesial=-1) 

# mesial_ridge_distance = calculate_oklidian_length_point(outer_mesial_point, cavity_mesial_point )
# distal_ridge_distance = calculate_oklidian_length_point(outer_distal_point, cavity_distal_point )




num_vertices = vertices.shape[0]
start_idx  = int(num_vertices * 0.3)
end_idx  = int(num_vertices * 0.5)

# Sort vertices by Z-coordinate (descending order)
sorted_indices = np.argsort(vertices[:, 2])[::-1]  # Use `1` for Y-axis or `0` for X-axis
interval_indices = sorted_indices[start_idx:end_idx]  # Select top 10%

# Extract the top points
interval_points = vertices[interval_indices]

# Create a point cloud object
pcd = o3d.geometry.PointCloud()
pcd.points = o3d.utility.Vector3dVector(interval_points)

interval_obb = pcd.get_oriented_bounding_box()
interval_obb.color = (1,0,0)

cavity_obb = largest_cavity_mesh.get_oriented_bounding_box()
cavity_obb.color = (0,0,1)
box_corners = np.asarray(interval_obb.get_box_points()) #3,5 4,6

#mesial ridge_width calculation
mesial_line_start = box_corners[4]
mesial_line_end = box_corners[6]
mesial_middle_point = get_top_right_edge_midpoint_pcd(cavity_obb, 4 ,6)
# other_middle = get_top_right_edge_midpoint_pcd(interval_obb)
mesial_ridge_distance , mesial_ridge_closest_point= calculate_point_to_line_distance(mesial_middle_point , mesial_line_start , mesial_line_end)
mesial_ridge_width_mesh = create_cylinder_between_points(mesial_ridge_closest_point,mesial_middle_point)
mesial_ridge_width_mesh.paint_uniform_color([1.0, 0.0, 0.0])  

#distal ridge_width calculation
distal_line_start = box_corners[3]
distal_line_end = box_corners[5]
distal_middle_point = get_top_right_edge_midpoint_pcd(cavity_obb, 3 ,5)
distal_ridge_distance , distal_ridge_closest_point= calculate_point_to_line_distance(distal_middle_point , distal_line_start , distal_line_end)
distal_ridge_width_mesh = create_cylinder_between_points(distal_ridge_closest_point,distal_middle_point)
distal_ridge_width_mesh.paint_uniform_color([0.0, 1.0, 0.0])  


# export meshes to stl files
#ALL MESHES : colored_roughness, cavity_bottom, line_set, largest_cavity_mesh, tooth_o3d  ,cylinder_mesh ,distal_ridge_width_mesh ,mesial_ridge_width_mesh
mkdir = args.studentId
os.makedirs(f"output/{mkdir}",exist_ok=True)
o3d.io.write_triangle_mesh(f"{BASE_DIR}/{mkdir}/colored_roughness.ply", colored_roughness , write_vertex_colors=True)
o3d.io.write_triangle_mesh(f"{BASE_DIR}/{mkdir}/cavity_bottom.ply", cavity_bottom ,  write_vertex_colors=True)
o3d.io.write_triangle_mesh(f"{BASE_DIR}/{mkdir}/largest_cavity_mesh.ply",  largest_cavity_mesh ,  write_vertex_colors=True)
o3d.io.write_triangle_mesh(f"{BASE_DIR}/{mkdir}/tooth_o3d.ply", tooth_o3d , write_vertex_colors=True)
o3d.io.write_triangle_mesh(f"{BASE_DIR}/{mkdir}/cavity_depth_mesh.ply", cavity_depth_mesh , write_vertex_colors=True)
o3d.io.write_triangle_mesh(f"{BASE_DIR}/{mkdir}/tooth_dimension_cylinder_meshes.ply", tooth_dimension_cylinder_meshes , write_vertex_colors=True)
o3d.io.write_triangle_mesh(f"{BASE_DIR}/{mkdir}/cavity_dimension_cylinder_meshes.ply", cavity_dimension_cylinder_meshes,  write_vertex_colors=True)
o3d.io.write_triangle_mesh(f"{BASE_DIR}/{mkdir}/distal_ridge_width_mesh.ply", distal_ridge_width_mesh , write_vertex_colors=True)
o3d.io.write_triangle_mesh(f"{BASE_DIR}/{mkdir}/mesial_ridge_width_mesh.ply", mesial_ridge_width_mesh,  write_vertex_colors=True)
o3d.io.write_triangle_mesh(f"{BASE_DIR}/{mkdir}/distal_isthmus_mesh.ply", distal_isthmus_cylinder, write_vertex_colors=True)
o3d.io.write_triangle_mesh(f"{BASE_DIR}/{mkdir}/mesial_isthmus_mesh.ply", mesial_isthmus_cylinder, write_vertex_colors=True)
o3d.io.write_triangle_mesh(f"{BASE_DIR}/{mkdir}/mesh_aligned.ply", mesh_source , write_vertex_colors = True )


b_l_length_ratio = (tooth_width - cavity_width) / tooth_width
m_d_length_ratio = (tooth_length - cavity_length) / tooth_length
# Calculate score,
is_cavity_length = 0
is_right_angle = 0
is_left_angle = 0
is_cavity_depth = 0
is_roughness = 0
is_m_d_length_ratio = 0
is_b_l_length_ratio = 0
is_mesial_ridge_distance_true = 0
is_distal_ridge_distance_true = 0 
is_cavity_width = 0
is_mesial_isthmus_width_true = 0
is_distal_isthmus_width_true = 0
is_critical_limits_exceeded = 0
score = 0

#degree ye çevir
right_angle = np.degrees(np.arccos(right_angle))
left_angle = np.degrees(np.arccos(left_angle))

if mesial_isthmus_width >= 1.5 and mesial_isthmus_width <= 1.99: 
    score += 10
    is_mesial_isthmus_width_true = 1 
elif (1.0 <= mesial_isthmus_width < 1.5) or (2.01 <= mesial_isthmus_width <= 2.5):
    score += 5
    is_mesial_isthmus_width_true = 0.5 
elif mesial_isthmus_width < 1.0 or mesial_isthmus_width > 2.5:
    is_mesial_isthmus_width_true = 0


if distal_isthmus_width >= 1.5 and distal_isthmus_width <= 1.99: 
    score += 10
    is_distal_isthmus_width_true = 1 
elif (1.0 <= distal_isthmus_width < 1.5) or (2.01 <= distal_isthmus_width <= 2.5):
    score += 5
    is_distal_isthmus_width_true = 0.5 
elif distal_isthmus_width < 1.0 or distal_isthmus_width > 2.5:
    is_distal_isthmus_width_true = 0

## mesial marginal
if mesial_ridge_distance>=1.2 and mesial_ridge_distance<=1.6:
    score +=10
    is_mesial_ridge_distance_true = 1
elif (mesial_ridge_distance<1.2 and mesial_ridge_distance>=1.0) or (mesial_ridge_distance>1.6 and mesial_ridge_distance<=2.0):
    score += 5
    is_mesial_ridge_distance_true = 0.5
elif mesial_ridge_distance<1.0 or mesial_ridge_distance>2:
    is_mesial_ridge_distance_true = 0
## distal marginal
if distal_ridge_distance>=1.2 and distal_ridge_distance<=1.6:
    score +=10
    is_distal_ridge_distance_true = 1
elif (distal_ridge_distance<1.2 and distal_ridge_distance>=1.0) or (distal_ridge_distance>1.6 and distal_ridge_distance<=2.0):
    score += 5
    is_distal_ridge_distance_true = 0.5
elif distal_ridge_distance<1.0 or distal_ridge_distance>2:
    is_distal_ridge_distance_true = 0


# Bucco-lingual length ratio grading
if b_l_length_ratio >= 0.35 and b_l_length_ratio <= 0.45:
    score += 10
    is_b_l_length_ratio = 1
elif (b_l_length_ratio >= 0.29 and b_l_length_ratio < 0.34):
    score += 5
    is_b_l_length_ratio = 0.5


# Mesio-distal length ratio grading
if m_d_length_ratio >= 0.65 and m_d_length_ratio <= 0.75:
    score += 10
    is_m_d_length_ratio = 1
elif m_d_length_ratio<0.65 and m_d_length_ratio>=0.6:
    score += 5
    is_m_d_length_ratio = 0.5


is_right_angle = 0
if right_angle >80 :
    score += 10
    is_right_angle = 1
elif right_angle > 70 :
    score +=5
    is_right_angle = 0.5

is_left_angle = 0
if left_angle > 80 :
    score +=10
    is_left_angle = 1
elif left_angle > 70 :
    score +=5
    is_left_angle = 0.5

if cavity_length>=2.7 and cavity_length<=3.3:
    is_cavity_length = 1
elif (cavity_length<=2.69 and cavity_length>=2.5) or (cavity_length>=3.31 and cavity_length<=3.5):
    is_cavity_length = 0.5
elif cavity_length<2.5 or cavity_length>3.5:
    is_cavity_length = 0


if cavity_width>=7.1 and cavity_width<=8.29:
    is_cavity_width = 1
elif cavity_width>=6.6 and cavity_width<=7.00:
    is_cavity_width = 0.5
    

if cavity_depth>=2.5 and cavity_depth<=3.0:
    score +=10
    is_cavity_depth = 1
elif (cavity_depth<=2.49 and cavity_depth>=2.0) or (cavity_depth>=3.01 and cavity_depth<=3.49):
    is_cavity_depth = 0.5
elif cavity_depth<2.0 or cavity_depth>3.5:
    is_cavity_depth = 0

std_roughness = np.std(roughness)

if std_roughness>=0 and std_roughness<=10.0:
    score +=10
    is_roughness = 1
elif std_roughness>=10.01 and std_roughness<=40.00:
    score += 5
    is_roughness = 0.5
elif std_roughness>40.00:
    is_roughness = 0


####critical 
if cavity_length > 3.5 or cavity_width > 8.3 or cavity_depth > 3.5 :
    score = 0
    is_critical_limits_exceeded = 1
    


#Stdout to return
#tooth_dimension_cylinder_meshes
# JSON içerisindeki veriler (örnek değerler kullanılmaktadır)
data = {
    "studentId" : args.studentId , 
    "mesial_isthmus_width": round(mesial_isthmus_width, 3),
    "is_mesial_isthmus_length_true": is_mesial_isthmus_width_true,

    "distal_isthmus_width": round(distal_isthmus_width, 3),
    "is_distal_isthmus_length_true": is_distal_isthmus_width_true,

    "right_angle": round(right_angle,3),
    "is_right_angle_true" : is_right_angle, 

    "left_angle": round(left_angle,3),
    "is_left_angle_true" : is_left_angle , 
    
    "cavity_depth": round(cavity_depth,3),
    "is_cavity_depth_true" : is_cavity_depth ,  
    
    "roughness":  round(np.std(roughness),3),
    "is_roughness_true" : is_roughness , 
    
    "m_d_length_ratio" : round(m_d_length_ratio,3) ,
    "is_m_d_length_ratio_true" : is_m_d_length_ratio ,
    
    "m_d_length" : round(cavity_width,3) ,
    "is_m_d_length_true" : is_cavity_width ,

    "b_l_length_ratio" : round(b_l_length_ratio,3),
    "is_b_l_length_ratio_true" : is_b_l_length_ratio ,
    
    "b_l_length" : round(cavity_length,3),
    "is_b_l_length_true" : is_cavity_length , 

    "distal_ridge_distance" : round(distal_ridge_distance,3) ,
    "is_distal_ridge_distance_true" : is_distal_ridge_distance_true,
    
    "mesial_ridge_distance" : round(mesial_ridge_distance,3) ,
    "is_mesial_ridge_distance_true" : is_mesial_ridge_distance_true ,

    "is_critical_limits_exceeded" : is_critical_limits_exceeded , 

    "score" : score
}

print(json.dumps(data))

# JSON dosyasına yazma
with open(f"output/{mkdir}/data.json", "w") as f:
    json.dump(data, f, indent=4)
### database yükleme bağlanma işlemleri :


def insert_ply_paths(studentId,base_dir = BASE_DIR):
    """
    student_ply_paths tablosuna bir kayıt ekler.
    
    :param studentID:    Öğrenci numarası (aynı zamanda mkdir adı olarak da kullanılır).
    :param base_dir:     Ana klasör (varsayılan "output").
    """
    # Dosya yollarını hazırla
    mkdir = studentId
    colored_roughness_path              = f"{base_dir}/{mkdir}/colored_roughness.ply"
    cavity_bottom_path                  = f"{base_dir}/{mkdir}/cavity_bottom.ply"
    largest_cavity_mesh_path            = f"{base_dir}/{mkdir}/largest_cavity_mesh.ply"
    tooth_o3d_path                      = f"{base_dir}/{mkdir}/tooth_o3d.ply"
    cavity_depth_mesh_path              = f"{base_dir}/{mkdir}/cavity_depth_mesh.ply"
    tooth_dimension_cylinder_meshes_path= f"{base_dir}/{mkdir}/tooth_dimension_cylinder_meshes.ply"
    cavity_dimension_cylinder_meshes_path = f"{base_dir}/{mkdir}/cavity_dimension_cylinder_meshes.ply"
    distal_ridge_width_mesh_path = f"{base_dir}/{mkdir}/distal_ridge_width_mesh.ply"
    mesial_ridge_width_mesh_path = f"{base_dir}/{mkdir}/mesial_ridge_width_mesh.ply"
    distal_isthmus_width_mesh_path = f"{base_dir}/{mkdir}/distal_isthmus_mesh.ply"
    mesial_isthmus_width_mesh_path = f"{base_dir}/{mkdir}/mesial_isthmus_mesh.ply"
    mesh_aligned_path = f"{BASE_DIR}/{mkdir}/mesh_aligned.ply"

    # Veritabanı bağlantısını aç
    conn = mysql.connector.connect(
        host=HOST,
        user=USER,
        password=PASSWORD,
        database=DATABASE
    )
    cur = conn.cursor()

    # Eğer studentID için bir UNIQUE kısıtlaman yoksa, tekrar eklemeleri önlemek için
    # öğrenci numarasını UNIQUE yapman veya ON DUPLICATE KEY UPDATE kullanman gerekebilir.
    insert_sql = """
        INSERT INTO student_ply_paths (
            studentID,
            colored_roughness_path,
            cavity_bottom_path,
            largest_cavity_mesh_path,
            tooth_o3d_path,
            cavity_depth_mesh_path,
            tooth_dimension_cylinder_meshes_path,
            cavity_dimension_cylinder_meshes_path,
            distal_ridge_width_mesh_path,
            mesial_ridge_width_mesh_path,
            distal_isthmus_width_mesh_path,
            mesial_isthmus_width_mesh_path,
            mesh_aligned_path
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s , %s , %s , %s)
        ON DUPLICATE KEY UPDATE
            colored_roughness_path               = VALUES(colored_roughness_path),
            cavity_bottom_path                   = VALUES(cavity_bottom_path),
            largest_cavity_mesh_path             = VALUES(largest_cavity_mesh_path),
            tooth_o3d_path                       = VALUES(tooth_o3d_path),
            cavity_depth_mesh_path               = VALUES(cavity_depth_mesh_path),
            tooth_dimension_cylinder_meshes_path = VALUES(tooth_dimension_cylinder_meshes_path),
            cavity_dimension_cylinder_meshes_path= VALUES(cavity_dimension_cylinder_meshes_path),
            distal_ridge_width_mesh_path         = VALUES(distal_ridge_width_mesh_path),
            mesial_ridge_width_mesh_path         = VALUES(mesial_ridge_width_mesh_path),
            distal_isthmus_width_mesh_path       = VALUES(mesial_ridge_width_mesh_path),
            mesial_ridge_width_mesh_path         = VALUES(mesial_ridge_width_mesh_path),
            mesh_aligned_path                    = VALUES(mesh_aligned_path)
    """

    values = (
        studentId,
        colored_roughness_path,
        cavity_bottom_path,
        largest_cavity_mesh_path,
        tooth_o3d_path,
        cavity_depth_mesh_path,
        tooth_dimension_cylinder_meshes_path,
        cavity_dimension_cylinder_meshes_path,
        distal_ridge_width_mesh_path,
        mesial_ridge_width_mesh_path,
        distal_isthmus_width_mesh_path,
        mesial_isthmus_width_mesh_path,
        mesh_aligned_path
    )

    cur.execute(insert_sql, values)
    conn.commit()
    cur.close()
    conn.close()

def insert_score_data(data):
    connection = mysql.connector.connect(
        host=HOST,
        user=USER,
        password=PASSWORD,
        database=DATABASE
    )
    cursor = connection.cursor()
    insert_query = """
    INSERT INTO cavity_scores (
        studentId,
        right_angle, is_right_angle_true,
        left_angle, is_left_angle_true,
        cavity_depth, is_cavity_depth_true,
        roughness, is_roughness_true,
        m_d_length_ratio, is_m_d_length_ratio_true,
        m_d_length, is_m_d_length_true,
        b_l_length_ratio, is_b_l_length_ratio_true,
        b_l_length, is_b_l_length_true,
        distal_ridge_distance, is_distal_ridge_distance_true,
        mesial_ridge_distance, is_mesial_ridge_distance_true,
        mesial_isthmus_width , is_mesial_isthmus_width_true, 
        distal_isthmus_width , is_distal_isthmus_width_true,
        is_critical_limits_exceeded,
        score
    ) VALUES ( %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s , %s ,%s , %s , %s , %s)
    ON DUPLICATE KEY UPDATE
        right_angle = VALUES(right_angle),
        is_right_angle_true = VALUES(is_right_angle_true),
        left_angle = VALUES(left_angle),
        is_left_angle_true = VALUES(is_left_angle_true),
        cavity_depth = VALUES(cavity_depth),
        is_cavity_depth_true = VALUES(is_cavity_depth_true),
        roughness = VALUES(roughness),
        is_roughness_true = VALUES(is_roughness_true),
        m_d_length_ratio = VALUES(m_d_length_ratio),
        is_m_d_length_ratio_true = VALUES(is_m_d_length_ratio_true),
        m_d_length = VALUES(m_d_length),
        is_m_d_length_true = VALUES(is_m_d_length_true),
        b_l_length_ratio = VALUES(b_l_length_ratio),
        is_b_l_length_ratio_true = VALUES(is_b_l_length_ratio_true),
        b_l_length = VALUES(b_l_length),
        is_b_l_length_true = VALUES(is_b_l_length_true),
        distal_ridge_distance = VALUES(distal_ridge_distance),
        is_distal_ridge_distance_true = VALUES(is_distal_ridge_distance_true),
        mesial_ridge_distance = VALUES(mesial_ridge_distance),
        is_mesial_ridge_distance_true = VALUES(is_mesial_ridge_distance_true),
        mesial_isthmus_width = VALUES(mesial_isthmus_width),
        is_mesial_isthmus_width_true = VALUES(is_mesial_isthmus_width_true),
        distal_isthmus_width = VALUES(distal_isthmus_width),
        is_distal_isthmus_width_true = VALUES(is_distal_isthmus_width_true),
        is_critical_limits_exceeded = VALUES(is_critical_limits_exceeded),
        score = VALUES(score);
    """
    # bazı değerler numpy olarak kaldıği için database a atarken sıkıntı yaşatabiliyor.
    clean_data = {k: float(v) if isinstance(v, np.floating) else int(v) if isinstance(v, np.integer) else v for k, v in data.items()}

    values = (
        clean_data["studentId"],

        round(clean_data["right_angle"], 3),
        clean_data["is_right_angle_true"],

        round(clean_data["left_angle"], 3),
        clean_data["is_left_angle_true"],

        round(clean_data["cavity_depth"], 3),
        clean_data["is_cavity_depth_true"],

        round(clean_data["roughness"], 3),
        clean_data["is_roughness_true"],

        round(clean_data["m_d_length_ratio"], 3),
        clean_data["is_m_d_length_ratio_true"],

        round(clean_data["m_d_length"], 3),
        clean_data["is_m_d_length_true"],

        round(clean_data["b_l_length_ratio"], 3),
        clean_data["is_b_l_length_ratio_true"],

        round(clean_data["b_l_length"], 3),
        clean_data["is_b_l_length_true"],

        round(clean_data["distal_ridge_distance"], 3),
        clean_data["is_distal_ridge_distance_true"],

        round(clean_data["mesial_ridge_distance"], 3),
        clean_data["is_mesial_ridge_distance_true"],

        round(clean_data["mesial_isthmus_width"], 3),
        clean_data["is_mesial_isthmus_length_true"] , 

        round(clean_data["distal_isthmus_width"], 3),
        clean_data["is_distal_isthmus_length_true"] , 

        clean_data["is_critical_limits_exceeded"],
        clean_data["score"]
    )

    cursor.execute(insert_query, values)
    connection.commit()
    cursor.close()
    connection.close()

insert_ply_paths(args.studentId)
insert_score_data(data=data)

#TODO cuvature fonksiyonun GPU ya taşınması

#TODO farklı dişler için de yap yaklaşık 7-8 tane yap 
#TODO elenecek 3 şey critical | database DONE | önyüz  

# info butonu ekle hesaplama şekli 
#TODO PDF 
#TODO dişin alignment ı 

#TODO isthmus ve ridge gösterimini ekle arayüze (done)
