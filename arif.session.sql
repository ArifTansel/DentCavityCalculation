DROP TABLE IF EXISTS cavity_scores;

CREATE TABLE cavity_scores ( 
    studentID VARCHAR(20) PRIMARY KEY,
    
    right_angle FLOAT,
    is_right_angle_true FLOAT,

    left_angle FLOAT,
    is_left_angle_true FLOAT,

    cavity_depth FLOAT,
    is_cavity_depth_true FLOAT,

    roughness FLOAT,
    is_roughness_true FLOAT,

    m_d_length_ratio FLOAT,
    is_m_d_length_ratio_true FLOAT,

    m_d_length FLOAT,
    is_m_d_length_true FLOAT,

    b_l_length_ratio FLOAT,
    is_b_l_length_ratio_true FLOAT,

    b_l_length FLOAT,
    is_b_l_length_true FLOAT,

    distal_ridge_distance FLOAT,
    is_distal_ridge_distance_true FLOAT,

    mesial_ridge_distance FLOAT,
    is_mesial_ridge_distance_true FLOAT,

    distal_isthmus_width FLOAT , 
    is_distal_isthmus_width_true FLOAT ,

    mesial_isthmus_width FLOAT , 
    is_mesial_isthmus_width_true FLOAT,

    is_critical_limits_exceeded FLOAT, 
    score FLOAT
);


--@block
CREATE TABLE student_list (
    studentID VARCHAR(20) PRIMARY KEY,
    studentName VARCHAR(50) NOT NULL,
    studentLastname VARCHAR(50) NOT NULL,
    stlFile BOOLEAN NOT NULL 
);

--@block
DROP TABLE IF EXISTS student_ply_paths;
CREATE TABLE student_ply_paths (
    studentID VARCHAR(20) PRIMARY KEY,
    colored_roughness_path VARCHAR(255),
    cavity_bottom_path VARCHAR(255),
    largest_cavity_mesh_path VARCHAR(255),
    tooth_o3d_path VARCHAR(255),
    cavity_depth_mesh_path VARCHAR(255),
    tooth_dimension_cylinder_meshes_path VARCHAR(255),
    cavity_dimension_cylinder_meshes_path VARCHAR(255),
    distal_ridge_width_mesh_path VARCHAR(255),
    mesial_ridge_width_mesh_path VARCHAR(255),
    distal_isthmus_width_mesh_path VARCHAR(255),
    mesial_isthmus_width_mesh_path VARCHAR(255),
    mesh_aligned_path VARCHAR(255)
);

--@block
DELETE FROM cavity_scores

--@block
DELETE  FROM student_list
--@block
SELECT s.stlFile, c.* FROM student_list s LEFT JOIN cavity_scores c ON s.studentID = c.studentID WHERE s.studentID = 1111;
--@block
INSERT INTO student_list (studentID, studentName, studentLastname)
VALUES (
    '124',
    'veli',
    'mehmed'
  );