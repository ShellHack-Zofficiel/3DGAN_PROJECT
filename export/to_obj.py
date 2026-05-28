import numpy as np
import os
from pathlib import Path


class PointCloudToOBJ:
    """Convert point clouds to OBJ format"""
    
    @staticmethod
    def save_point_cloud_as_obj(point_cloud, obj_path, scale=1.0):
        """
        Save point cloud as OBJ file
        
        Args:
            point_cloud: numpy array of shape (N, 3) or torch tensor
            obj_path: path to save OBJ file
            scale: scaling factor
        """
        # Convert to numpy if needed
        if hasattr(point_cloud, 'cpu'):
            point_cloud = point_cloud.cpu().detach().numpy()
        
        # Create directory if needed
        os.makedirs(os.path.dirname(obj_path), exist_ok=True)
        
        # Scale points
        point_cloud = point_cloud * scale
        
        # Write OBJ file
        with open(obj_path, 'w') as f:
            f.write("# Generated point cloud OBJ file\n")
            f.write(f"# Total points: {len(point_cloud)}\n\n")
            
            # Write vertices
            for i, point in enumerate(point_cloud):
                f.write(f"v {point[0]:.6f} {point[1]:.6f} {point[2]:.6f}\n")
            
            # Write points as faces (for visualization in some viewers)
            for i in range(1, len(point_cloud) + 1):
                f.write(f"p {i}\n")


class PointCloudToMesh:
    """Convert point clouds to mesh format"""
    
    @staticmethod
    def poisson_reconstruction(point_cloud, depth=9):
        """
        Poisson surface reconstruction
        Requires: pip install open3d
        
        Args:
            point_cloud: numpy array of shape (N, 3)
            depth: octree depth for reconstruction
            
        Returns:
            mesh: open3d TriangleMesh
        """
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Please install open3d: pip install open3d")
        
        # Convert to numpy if needed
        if hasattr(point_cloud, 'cpu'):
            point_cloud = point_cloud.cpu().detach().numpy()
        
        # Create point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        
        # Estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_neighbor=30))
        
        # Poisson reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=depth)
        
        return mesh
    
    @staticmethod
    def ball_pivoting(point_cloud, radii=None):
        """
        Ball pivoting algorithm for mesh reconstruction
        Requires: pip install open3d
        
        Args:
            point_cloud: numpy array of shape (N, 3)
            radii: list of radii for ball pivoting
            
        Returns:
            mesh: open3d TriangleMesh
        """
        try:
            import open3d as o3d
        except ImportError:
            raise ImportError("Please install open3d: pip install open3d")
        
        # Convert to numpy if needed
        if hasattr(point_cloud, 'cpu'):
            point_cloud = point_cloud.cpu().detach().numpy()
        
        if radii is None:
            radii = [0.005, 0.01, 0.02, 0.04]
        
        # Create point cloud object
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(point_cloud)
        
        # Estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=0.1, max_neighbor=30))
        
        # Ball pivoting
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii))
        
        return mesh


class OBJExporter:
    """Export mesh to OBJ format"""
    
    @staticmethod
    def save_mesh_as_obj(mesh, obj_path, scale=1.0):
        """
        Save mesh as OBJ file
        
        Args:
            mesh: trimesh object or open3d TriangleMesh
            obj_path: path to save OBJ file
            scale: scaling factor
        """
        import os
        os.makedirs(os.path.dirname(obj_path), exist_ok=True)
        
        # Handle different mesh types
        if hasattr(mesh, 'export'):  # trimesh
            mesh.export(obj_path)
        elif hasattr(mesh, 'write_triangle_mesh'):  # open3d
            import open3d as o3d
            o3d.io.write_triangle_mesh(obj_path, mesh)
        else:
            raise TypeError("Unsupported mesh type")
    
    @staticmethod
    def export_batch(point_clouds, output_dir, mesh_type='poisson'):
        """
        Export batch of point clouds as OBJ files
        
        Args:
            point_clouds: list of point clouds or batch tensor
            output_dir: directory to save OBJ files
            mesh_type: 'point_cloud', 'poisson', or 'ball_pivoting'
        """
        os.makedirs(output_dir, exist_ok=True)
        
        for i, pc in enumerate(point_clouds):
            output_path = os.path.join(output_dir, f"model_{i:04d}.obj")
            
            if mesh_type == 'point_cloud':
                PointCloudToOBJ.save_point_cloud_as_obj(pc, output_path)
            elif mesh_type == 'poisson':
                mesh = PointCloudToMesh.poisson_reconstruction(pc)
                OBJExporter.save_mesh_as_obj(mesh, output_path)
            elif mesh_type == 'ball_pivoting':
                mesh = PointCloudToMesh.ball_pivoting(pc)
                OBJExporter.save_mesh_as_obj(mesh, output_path)
            
            print(f"Exported: {output_path}")
