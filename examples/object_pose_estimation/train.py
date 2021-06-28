import os
import numpy as np
from glob import glob

from scenes import SingleView
from pipelines import RandomKeypointsRender, DrawNormalizedKeypoints
from paz.backend.image import show_image
from trimesh import load_mesh
import pyrender

# mesh_path = '.keras/paz/datasets/ycb_models/035_power_drill/textured.obj'
mesh_path = 'Repositories/solar_panels/solar_panel_02/meshes/obj/base_link.obj'
image_path = 'Pictures/JPEGImages/*.jpg'

mesh_path = os.path.join(os.path.expanduser('~'), mesh_path)
mesh = load_mesh(mesh_path)
myscene = pyrender.Scene.from_trimesh_scene(mesh)
nodes = list(myscene.nodes)
meshes = list(myscene.meshes)
myscene.ambient_light = [0.3, 0.3, 0.3, 1.0]

for node in nodes:
    if node.mesh.bounds[0, 0] > 0:
        a = myscene.remove_node(node)
        print(a)

# pyrender.Viewer(scene, use_raymond_lighting=True)
# node = scene.nodes
# mesh = mesh.dump(concatenate=True)

image_shape = (512, 512)
y_fov = 3.14159 / 4.0
distance = [0.5, 1.5]
light = [30, 30]
top_only = False
roll = None
shift = None
occlusions = 3
image_path = os.path.join(os.path.expanduser('~'), image_path)
image_paths = glob(image_path)
x_offset = y_offset = z_offset = 0.05
num_keypoints = 6
keypoints = np.zeros((num_keypoints, 4))
radius = 0.25
angles = np.linspace(0, 2 * np.pi, num_keypoints, endpoint=False)
for keypoint_arg, angle in enumerate(angles):
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    keypoints[keypoint_arg] = x, y, 0.0, 1.0

"""
keypoints = np.array([[0.25, 0.0, 0.0, 1.0],
                      [0.0, 0.25, 0.0, 1.0],
                      [0.0, 0.0, 0.25, 1.0]])
"""

args = (myscene, image_shape, y_fov, distance, light, top_only, roll, shift)
scene = SingleView(*args)
scene.scene.ambient_light = [0.3, 0.3, 0.3, 2.0]
image, alpha_channel, world_to_camera = scene.render()
processor = RandomKeypointsRender(scene, keypoints, image_paths, occlusions)
draw_normalized_keypoints = DrawNormalizedKeypoints(len(keypoints), 10, True)

for arg in range(100):
    sample = processor()
    image = sample['inputs']['image']
    keypoints = sample['labels']['keypoints']
    image = draw_normalized_keypoints(image, keypoints)
    image = (255.0 * image).astype('uint8')
    show_image(image)
