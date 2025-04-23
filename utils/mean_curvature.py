import torch
import numpy as np

def discrete_mean_curvature_measure_gpu(mesh, points, radius):
    """
    Return the discrete mean curvature measure of a sphere
    centered at a point as detailed in 'Restricted Delaunay
    triangulations and normal cycle'- Cohen-Steiner and Morvan.
    
    GPU accelerated version using PyTorch.

    Parameters
    ----------
    points : (n, 3) float or torch.Tensor
      Points in space
    radius : float
      Sphere radius which should typically be greater than zero

    Returns
    --------
    mean_curvature : (n,) float or torch.Tensor
      Discrete mean curvature measure.
    """
    # Convert inputs to torch tensors if they aren't already
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        points = points.to('cuda' if torch.cuda.is_available() else 'cpu')
    
    if points.shape[1] != 3:
        raise ValueError("points must be (n,3)!")
    
    # Convert mesh data to torch tensors on the same device as points
    device = points.device
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32).to(device)
    face_adjacency_edges = torch.tensor(mesh.face_adjacency_edges, dtype=torch.int64).to(device)
    face_adjacency_angles = torch.tensor(mesh.face_adjacency_angles, dtype=torch.float32).to(device)
    face_adjacency_convex = torch.tensor(mesh.face_adjacency_convex, dtype=torch.bool).to(device)
    
    # Define bounds for each point's sphere
    # Since we can't use mesh.face_adjacency_tree directly with GPU, 
    # we'll process all edges for each point and filter by distance
    
    # Get all edge endpoints
    edge_start = vertices[face_adjacency_edges[:, 0]]
    edge_end = vertices[face_adjacency_edges[:, 1]]
    
    # Initialize mean curvature result
    mean_curv = torch.zeros(len(points), dtype=torch.float32, device=device)
    
    # Process each point
    for i, point in enumerate(points):
        # Calculate the length of intersection between each edge and the sphere
        lengths = line_ball_intersection_gpu(
            edge_start, 
            edge_end, 
            center=point, 
            radius=radius
        )
        
        # Filter to only include edges that intersect with the sphere
        mask = lengths > 0
        if torch.any(mask):
            angles = face_adjacency_angles[mask]
            signs = torch.where(face_adjacency_convex[mask], 
                            torch.tensor(1.0, device=device), 
                            torch.tensor(-1.0, device=device))
            
            # Calculate contribution to mean curvature
            mean_curv[i] = torch.sum(lengths[mask] * angles * signs) / 2
    
    # Return result, converting back to numpy if input was numpy
    if isinstance(points, np.ndarray):
        return mean_curv.cpu().numpy()
    return mean_curv


def line_ball_intersection_gpu(start_points, end_points, center, radius):
    """
    Compute the length of the intersection of line segments with a ball.
    GPU accelerated version using PyTorch.

    Parameters
    ----------
    start_points : (n,3) torch.Tensor, list of points in space
    end_points   : (n,3) torch.Tensor, list of points in space
    center       : (3,) torch.Tensor, the sphere center
    radius       : float, the sphere radius

    Returns
    --------
    lengths: (n,) torch.Tensor, the intersection lengths.
    """
    device = start_points.device
    
    # Convert to torch tensor if it's not already
    if not isinstance(radius, torch.Tensor):
        radius = torch.tensor(radius, dtype=torch.float32).to(device)
    
    # Vector along each line segment
    L = end_points - start_points
    
    # Vector from center to start point
    oc = start_points - center
    
    # Calculate dot products efficiently using torch operations
    ldotl = torch.sum(L * L, dim=1)  # l.l
    ldotoc = torch.sum(L * oc, dim=1)  # l.(o-c)
    ocdotoc = torch.sum(oc * oc, dim=1)  # (o-c).(o-c)
    
    # Calculate discriminant
    discrims = ldotoc**2 - ldotl * (ocdotoc - radius**2)
    
    # Initialize lengths to zeros
    lengths = torch.zeros_like(ldotl)
    
    # Create mask for positive discriminants
    mask = discrims > 0
    
    if torch.any(mask):
        # Calculate intersection parameters
        d1 = (-ldotoc[mask] - torch.sqrt(discrims[mask])) / ldotl[mask]
        d2 = (-ldotoc[mask] + torch.sqrt(discrims[mask])) / ldotl[mask]
        
        # Clip to line segment bounds [0,1]
        d1 = torch.clamp(d1, 0, 1)
        d2 = torch.clamp(d2, 0, 1)
        
        # Calculate lengths
        lengths[mask] = (d2 - d1) * torch.sqrt(ldotl[mask])
    
    return lengths


# Example usage:
# If mesh is your trimesh object and you have points and radius
# mean_curvature = discrete_mean_curvature_measure_gpu(mesh, points, radius)

# For batch processing (much more efficient):
def batch_discrete_mean_curvature_gpu(mesh, points, radius, batch_size=1000):
    """
    Process mean curvature calculation in batches for very large point sets.
    
    Parameters
    ----------
    mesh : trimesh.Trimesh
        The mesh object
    points : (n, 3) array-like
        Points in space
    radius : float
        Sphere radius
    batch_size : int
        Number of points to process in each batch
        
    Returns
    -------
    mean_curvature : (n,) array
        Discrete mean curvature measure for each point
    """
    if not isinstance(points, torch.Tensor):
        points = torch.tensor(points, dtype=torch.float32)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    points = points.to(device)
    
    num_points = len(points)
    results = []
    
    for i in range(0, num_points, batch_size):
        batch_points = points[i:i+batch_size]
        batch_result = discrete_mean_curvature_measure_gpu(mesh, batch_points, radius)
        results.append(batch_result)
    
    # Combine results
    if isinstance(results[0], torch.Tensor):
        return torch.cat(results)
    else:
        return np.concatenate(results)