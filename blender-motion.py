import bpy
import os
import json
import math
import mathutils

# --- CONFIG ---
FOLDER = r"C:/Users/brany/Videos/object-mimic/objectmimic-output-001"
FOLDER = r"C:/Users/brany/Videos/object-mimic/objectmimic-output-002"
FOLDER = r"C:/Users/brany/Videos/object-mimic/taza-right-up-down"
ANIMATE_SCALE = False  # False: apply average scale to all frames; True: keyframe per-frame scale
POINT_RADIUS = 0.005  # sphere radius (metres) instanced on each point cloud vertex

# Coordinate remapping: src X→Blender -Y, src Y→Blender Z, src Z→Blender X
M = mathutils.Matrix((
    (0,  0, 1),
    (-1, 0, 0),
    (0,  1, 0),
))
M_inv = M.transposed()  # M is orthogonal

# Rest-pose correction: -90° around Blender X, then -90° around Blender Y
R_offset = (
    mathutils.Matrix.Rotation(math.radians(-90), 3, 'Y')
    @ mathutils.Matrix.Rotation(math.radians(-90), 3, 'X')
)

ply_files = sorted([g for g in os.listdir(FOLDER) if g.endswith(".ply")])
ply_path = os.path.join(FOLDER, ply_files[0])


def make_renderable(obj, radius=POINT_RADIUS):
    """Instance a small sphere on every point so the object is renderable."""
    mod = obj.modifiers.new(name="PointsToSpheres", type='NODES')
    ng = bpy.data.node_groups.new("PointsToSpheres", 'GeometryNodeTree')
    mod.node_group = ng

    ng.interface.new_socket(
        "Geometry", in_out='INPUT', socket_type='NodeSocketGeometry'
    )
    ng.interface.new_socket(
        "Geometry", in_out='OUTPUT', socket_type='NodeSocketGeometry'
    )

    nodes = ng.nodes
    links = ng.links
    nodes.clear()

    in_node = nodes.new('NodeGroupInput')
    out_node = nodes.new('NodeGroupOutput')
    sphere = nodes.new('GeometryNodeMeshIcoSphere')
    sphere.inputs['Radius'].default_value = radius
    sphere.inputs['Subdivisions'].default_value = 2
    inst = nodes.new('GeometryNodeInstanceOnPoints')
    realize = nodes.new('GeometryNodeRealizeInstances')

    links.new(in_node.outputs[0], inst.inputs['Points'])
    links.new(sphere.outputs['Mesh'], inst.inputs['Instance'])
    links.new(inst.outputs['Instances'], realize.inputs['Geometry'])
    links.new(realize.outputs['Geometry'], out_node.inputs[0])


def rotation_6d_to_matrix(r6d):
    """Convert 6D rotation representation to a 3x3 mathutils.Matrix.

    The 6D format encodes the first two columns of a rotation matrix.
    The third column is recovered via Gram-Schmidt + cross product.
    """
    a1 = mathutils.Vector(r6d[0:3])
    a2 = mathutils.Vector(r6d[3:6])
    b1 = a1.normalized()
    b2 = (a2 - a2.dot(b1) * b1).normalized()
    b3 = b1.cross(b2)
    return mathutils.Matrix((
        (b1.x, b2.x, b3.x),
        (b1.y, b2.y, b3.y),
        (b1.z, b2.z, b3.z),
    ))


def apply_motion(json_filename, name_suffix=""):
    json_path = os.path.join(FOLDER, json_filename)
    if not os.path.exists(json_path):
        return

    with open(json_path) as f:
        motion = json.load(f)

    fps = motion["video_info"]["fps"]
    frames_data = motion["frames"]

    bpy.ops.wm.ply_import(filepath=ply_path)
    obj = bpy.context.selected_objects[0]
    if name_suffix:
        obj.name = obj.name + name_suffix
    obj.rotation_mode = 'QUATERNION'
    make_renderable(obj)

    # Apply scale: either average across all frames, or per-frame keyframes
    if not ANIMATE_SCALE:
        avg_scale = sum(f["scale"][0] for f in frames_data) / len(frames_data)
        obj.scale = (avg_scale, avg_scale, avg_scale)

    # Match scene FPS to source video (set once; last file wins if both exist)
    bpy.context.scene.render.fps = round(fps)

    for frame_data in frames_data:
        blender_frame = round(frame_data["timestamp"] * fps)

        # Location
        t = mathutils.Vector(frame_data["translation"])
        obj.location = M @ t
        obj.keyframe_insert(data_path="location", frame=blender_frame)

        # Rotation: convert 6D → matrix, remap coordinate system, apply rest-pose correction
        R_src = rotation_6d_to_matrix(frame_data["rotation_6d"])
        R_bl = R_offset @ M @ R_src @ M_inv
        obj.rotation_quaternion = R_bl.to_quaternion()
        obj.keyframe_insert(data_path="rotation_quaternion", frame=blender_frame)

        # Scale
        if ANIMATE_SCALE:
            s = frame_data["scale"][0]
            obj.scale = (s, s, s)
            obj.keyframe_insert(data_path="scale", frame=blender_frame)

    # Frame range
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = round(frames_data[-1]["timestamp"] * fps)

    # Linear interpolation — more faithful for motion-capture data
    if obj.animation_data and obj.animation_data.action:
        for fcurve in obj.animation_data.action.fcurves:
            for kp in fcurve.keyframe_points:
                kp.interpolation = 'LINEAR'


apply_motion("motion.json")
apply_motion("motion-smooth.json", name_suffix="-smooth")

