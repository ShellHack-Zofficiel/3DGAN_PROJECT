import os
import subprocess
import numpy as np
from pathlib import Path


class FBXExporter:
    """Export 3D models to FBX format"""
    
    @staticmethod
    def obj_to_fbx(obj_path, fbx_path, blender_path=None):
        """
        Convert OBJ to FBX using Blender
        
        Args:
            obj_path: path to OBJ file
            fbx_path: path to save FBX file
            blender_path: path to Blender executable (auto-detect if None)
        """
        if blender_path is None:
            blender_path = FBXExporter.find_blender()
        
        if not os.path.exists(blender_path):
            raise FileNotFoundError(f"Blender not found at {blender_path}")
        
        # Créer le script Python pour Blender
        obj_path_escaped = obj_path.replace('\\', '\\\\')
        fbx_path_escaped = fbx_path.replace('\\', '\\\\')
        script = f"""
import bpy
import os

# Clear default scene
bpy.ops.object.select_all(action='SELECT')
bpy.ops.object.delete()

# Import OBJ
bpy.ops.import_scene.obj(filepath='{obj_path_escaped}')

# Export FBX
bpy.ops.export_scene.fbx(filepath='{fbx_path_escaped}')

print(f"Successfully converted {obj_path_escaped} to {fbx_path_escaped}")
"""
        
        # Enregistrer le script temporaire de conversion
        import tempfile
        with tempfile.NamedTemporaryFile(delete=False, suffix='.py', mode='w', encoding='utf-8') as temp_file:
            temp_file.write(script)
            script_path = temp_file.name
        
        # Exécuter Blender
        try:
            subprocess.run(
                [blender_path, '--background', '--python', script_path],
                check=True
            )
            print(f"Conversion successful: {fbx_path}")
        except subprocess.CalledProcessError as e:
            print(f"Blender conversion failed: {e}")
            raise
        finally:
            # Nettoyer le script temporaire
            if os.path.exists(script_path):
                os.remove(script_path)
    
    @staticmethod
    def find_blender():
        """Find Blender executable"""
        import platform
        import shutil
        
        system = platform.system()
        
        if system == 'Windows':
            # Try common installation paths
            possible_paths = [
                "C:\\Program Files\\Blender Foundation\\Blender 3.6\\blender.exe",
                "C:\\Program Files\\Blender Foundation\\Blender 4.0\\blender.exe",
                "C:\\Program Files (x86)\\Blender Foundation\\Blender\\blender.exe",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    return path
        elif system == 'Darwin':  # macOS
            possible_paths = [
                "/Applications/Blender.app/Contents/MacOS/Blender",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    return path
        elif system == 'Linux':
            # Try system blender
            blender = shutil.which('blender')
            if blender:
                return blender
        
        # Try PATH
        blender = shutil.which('blender')
        if blender:
            return blender
        
        raise FileNotFoundError("Blender not found. Please install Blender with Python support.")
    
    @staticmethod
    def batch_convert(obj_dir, fbx_dir, blender_path=None):
        """
        Convert all OBJ files in directory to FBX
        
        Args:
            obj_dir: directory containing OBJ files
            fbx_dir: directory to save FBX files
            blender_path: path to Blender executable
        """
        os.makedirs(fbx_dir, exist_ok=True)
        
        for obj_file in os.listdir(obj_dir):
            if obj_file.endswith('.obj'):
                obj_path = os.path.join(obj_dir, obj_file)
                fbx_filename = obj_file.replace('.obj', '.fbx')
                fbx_path = os.path.join(fbx_dir, fbx_filename)
                
                try:
                    FBXExporter.obj_to_fbx(obj_path, fbx_path, blender_path)
                except Exception as e:
                    print(f"Failed to convert {obj_file}: {e}")


class UniversalExporter:
    """Universal exporter for multiple formats"""
    
    @staticmethod
    def export_to_format(mesh, output_path, format='obj'):
        """
        Export mesh to various formats
        
        Args:
            mesh: trimesh object
            output_path: path to save file
            format: 'obj', 'fbx', 'gltf', 'ply', 'stl', etc.
        """
        try:
            import trimesh
        except ImportError:
            raise ImportError("Please install trimesh: pip install trimesh")
        
        # Ensure directory exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Export
        if format == 'fbx':
            # For FBX, first export as OBJ then convert
            obj_path = output_path.replace('.fbx', '.obj')
            mesh.export(obj_path)
            FBXExporter.obj_to_fbx(obj_path, output_path)
        else:
            mesh.export(output_path, file_type=format)
        
        print(f"Exported to: {output_path}")
