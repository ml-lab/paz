import os
import numpy as np
from glob import glob

from scenes import SingleView2
from pipelines import RandomKeypointsRender, DrawNormalizedKeypoints
from paz.backend.image import show_image
from paz.abstract import GeneratingSequence
from models import Poseur2D
from trimesh import load_mesh
import pyrender
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger
from tensorflow.keras.callbacks import LearningRateScheduler
from tensorflow.keras.optimizers import Adam

# mesh_path = '.keras/paz/datasets/ycb_models/035_power_drill/textured.obj'
mesh_path = 'Repositories/solar_panels/solar_panel_02/meshes/obj/base_link.obj'
image_path = '.keras/paz/datasets/voc-backgrounds/*.png'

mesh_path = os.path.join(os.path.expanduser('~'), mesh_path)
old_mesh = load_mesh(mesh_path)
save_path = 'trained_models/'
# mesh_path = 'Repositories/solar_panels/solar_panel_02/meshes/obj/base_link.obj'
# mesh_path = os.path.join(os.path.expanduser('~'), mesh_path)
# mesh = load_mesh(mesh_path)
for key in old_mesh.geometry:
    tri_mesh = old_mesh.geometry[key]
    if len(tri_mesh.vertices) == 1849:
        mesh = tri_mesh

# myscene = pyrender.Scene.from_trimesh_scene(mesh)
pymesh = pyrender.Mesh.from_trimesh(mesh)
myscene = pyrender.Scene()
myscene.add(pymesh)
# nodes = list(myscene.nodes)
# meshes = list(myscene.meshes)
myscene.ambient_light = [0.5, 0.5, 0.5, 1.0]

"""
for node in nodes:
    print(node.mesh.bounds[0, 0])
    # if node.mesh.bounds[0, 0] > 0.1:
    if node.mesh.bounds[0, 0] > 0:
        a = myscene.remove_node(node)
        print(a)
"""
# pyrender.Viewer(scene, use_raymond_lighting=True)
# node = scene.nodes
# mesh = mesh.dump(concatenate=True)


image_shape = (128, 128, 3)
y_fov = 3.14159 / 4.0
distance = [0.7, 0.7]
light = [30, 30]
top_only = True
roll = np.pi
shift = None
occlusions = 0
batch_size = 16
num_epochs = 100
multiprocessing = False
workers = 0
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

path = '/home/octavio/solar_panel.obj'
cs = np.cos(np.deg2rad(90))
ss = np.sin(np.deg2rad(90))
initial_pose = np.array([[1.0, 0.0, 0.0, 0.0],
                         [0.0, +cs, -ss, 0.0],
                         [0.0, +ss, +cs, 0.0],
                         [0.0, 0.0, 0.0, 1.0]])
keypoints = np.matmul(initial_pose, keypoints.T).T
args = (path, initial_pose, image_shape[:2], y_fov,
        distance, light, top_only, roll, shift)
scene = SingleView2(*args)
scene.scene.ambient_light = [1.0, 1.0, 1.0, 2.0]
image, alpha_channel, world_to_camera = scene.render()
# image, alpha_channel = scene.render()
processor = RandomKeypointsRender(scene, keypoints, image_paths, occlusions)
draw_normalized_keypoints = DrawNormalizedKeypoints(len(keypoints), 10, True)
show_image(image)
show_image(alpha_channel)
sequence = GeneratingSequence(processor, batch_size, 1000)
batch = sequence.__getitem__(0)
"""
for sample_arg in range(batch_size):
    image = batch[0]['image'][sample_arg]
    keypoints = batch[1]['keypoints'][sample_arg]
    mask = batch[1]['mask'][sample_arg]
    image = draw_normalized_keypoints(image, keypoints)
    image = (255.0 * image).astype('uint8')
    show_image(image)
    mask = (255.0 * mask).astype('uint8')
    show_image(mask)
"""
model = Poseur2D(image_shape, num_keypoints, True, 32)

# setting callbacks
model_path = os.path.join(save_path, model.name)
if not os.path.exists(model_path):
    os.makedirs(model_path)
log = CSVLogger(os.path.join(model_path, model.name + '-optimization.log'))
save_path = os.path.join(model_path, 'weights.{epoch:02d}-{loss:.2f}.hdf5')
checkpoint = ModelCheckpoint(
    save_path, 'loss', verbose=1, save_weights_only=True)

optimizer = Adam()
loss = {'keypoints': 'mean_squared_error',
        'mask': 'binary_crossentropy'}
model.compile(optimizer, loss)

model.fit(
    sequence,
    epochs=num_epochs,
    verbose=1,
    callbacks=[checkpoint, log],
    use_multiprocessing=multiprocessing,
    workers=workers)
