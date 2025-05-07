import os
import bpy
import numpy as np
import sys
from PIL import Image
from mathutils import Vector
import argparse
import json
import math 


def set_env_map(env_path: str):
    world = bpy.context.scene.world
    if not world:
        world = bpy.data.worlds.new("World")
        bpy.context.scene.world = world
    else:
        world.use_nodes = True
        world.node_tree.nodes.clear()

    env_texture_node = world.node_tree.nodes.new(type="ShaderNodeTexEnvironment")
    env_texture_node.location = (0, 0)

    bg_node = world.node_tree.nodes.new(type="ShaderNodeBackground")
    bg_node.location = (400, 0)

    output_node = world.node_tree.nodes.new(type="ShaderNodeOutputWorld")
    output_node.location = (800, 0)

    links = world.node_tree.links
    links.new(env_texture_node.outputs["Color"], bg_node.inputs["Color"])
    links.new(bg_node.outputs["Background"], output_node.inputs["Surface"])

    bpy.ops.image.open(filepath=env_path)
    env_texture_node.image = bpy.data.images.get(os.path.basename(env_path))


def look_at(camera, point):
    # Makes the camera point at a specific location
    direction = point - camera.location
    rot_quat = direction.to_track_quat('-Z', 'Y')
    camera.rotation_euler = rot_quat.to_euler()


def init_orth_camera( 
    center=Vector((float(0), float(0), float(0))), 
    radius: float = 1.5,
    elevation: float = 0,
    ortho_scale: float = 0.5, 
    num_views: int = 8,
    out_folder: str = '', 
    azimuth_deg = [-180, -135, 135, 0]
):  
    if 'OrthoCamera' in bpy.data.objects:
        camera = bpy.data.objects['OrthoCamera']
        camera.data.ortho_scale = ortho_scale  # Update scale if needed
    else:
        # Create new camera
        camera_data = bpy.data.cameras.new(name='OrthoCamera')
        camera_data.type = 'ORTHO'  # Set orthographic projection
        camera_data.ortho_scale = ortho_scale  # Adjust scale based on scene size
        
        camera = bpy.data.objects.new('OrthoCamera', camera_data)
        bpy.context.collection.objects.link(camera)
        
    bpy.context.scene.camera = camera  
    
    elevation = np.deg2rad(elevation)
    phi = 0.5 * math.pi - elevation  
    # azimuth_deg = np.linspace(-180, 0, num_views + 1)[:-1] 
    azimuth = np.deg2rad(azimuth_deg)
    
    for i, theta in enumerate(azimuth):   
        x = radius * math.sin(phi) * math.sin(theta)
        y = radius * math.sin(phi) * math.cos(theta)
        z = radius * math.cos(phi) 
        camera.location = Vector(center) + Vector((x, y, z))
        
        look_at(camera, Vector(center))  

        frame = bpy.context.scene.frame_end 
        bpy.context.scene.frame_end = frame + 1 
        camera.keyframe_insert(data_path="location", frame=frame)
        camera.keyframe_insert(data_path="rotation_euler", frame=frame) 
    return camera


def rendering(mesh_folder, idx, sequence_file, out_folder):
    # Main function that:
    # 1. Imports the mesh
    # 2. Updates vertex positions
    # 3. Handles materials and textures
    # 4. Sets up camera
    # 5. Renders the scene
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = 0

    imported_object = bpy.ops.import_scene.obj(filepath=mesh_folder+'/mesh.obj', use_split_objects=False) 
    new_vertices = np.load(sequence_file)[idx] 
    
    bpy.ops.object.mode_set(mode='OBJECT')
    # assign material 
    mat = bpy.data.materials.get('defaultMat.001') 
    obj = bpy.data.objects.get('mesh')
    obj.data.materials.append(mat)
    obj.active_material_index = 0 
    
    # assign new vertices
    mesh = obj.data
    for i, vert in enumerate(mesh.vertices):
        vert.co.x, vert.co.y, vert.co.z = new_vertices[i][0], new_vertices[i][1], new_vertices[i][2] 
    mesh.update()

    cond = obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.material_slot_remove()

    #change image of the material
    albedo_name = 'mesh_albedo_alpha.png'
    albedo_apha_path = os.path.join(mesh_folder, albedo_name) 

    bpy.data.images.load(albedo_apha_path)
    mat.node_tree.nodes["Image Texture"].image = bpy.data.images[albedo_name]

    # fix tooth normal
    bpy.ops.object.editmode_toggle() #enter edit mode
    bpy.ops.mesh.select_all(action='SELECT') #select all objects elements
    bpy.ops.mesh.normals_make_consistent(inside=False) #recalc normals
    bpy.ops.mesh.set_normals_from_faces() #set from faces
    bpy.ops.object.editmode_toggle() #exit edit mode

    init_orth_camera( 
        center=Vector((float(0), float(0), float(0))), 
        radius=1.0, 
        ortho_scale=2, 
        num_views=3, 
        out_folder=out_folder,  
    ) 
    # set_env_map('./resource/brown_photostudio_02_1k.exr')
 
    scene = bpy.context.scene
    # out_folder = mesh_folder.replace('recon-teeth','animation') 
    os.makedirs(out_folder , exist_ok=True)
    scene.render.filepath = f'{out_folder}/{idx:04d}_' 

    #render image
    bpy.context.scene.use_nodes = True
    if bpy.context.scene.frame_end != bpy.context.scene.frame_start: 
        bpy.context.scene.frame_end -= 1
        bpy.ops.render.render(animation=True, write_still=True)
        bpy.context.scene.frame_end += 1

    bpy.ops.render.render(write_still=True)

    #remove obj after rendering
    # bpy.data.objects.remove(obj, do_unlink=True)

def build_render(mesh_folder, fid):
    scene = bpy.context.scene
    # scene.cycles.device = 'GPU'
    # prefs = bpy.context.preferences
    # prefs.addons['cycles'].preferences.get_devices()
    # cprefs = prefs.addons['cycles'].preferences
    # cprefs.compute_device_type = 'CUDA'
    # for device in cprefs.devices:
    #     if device.type == 'CUDA':
    #         device.use = True

    rendering(mesh_folder, fid)

if __name__ == "__main__":
    sys.argv = [sys.argv[0]] + sys.argv[6:]
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh_folder", type=str)
    parser.add_argument("--frame_id", type=int)
    parser.add_argument("--sequence", type=str)
    parser.add_argument("--out_dir", type=str) 
    args = parser.parse_args()
    print(sys.argv)

    rendering(args.mesh_folder, args.frame_id, args.sequence, args.out_dir)
