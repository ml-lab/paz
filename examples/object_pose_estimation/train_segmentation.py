import os
import numpy as np
from glob import glob

from scenes import SingleView2
from pipelines import RandomKeypointsRender, DrawNormalizedKeypoints
from pipelines import RandomSegmentationRender
from paz.backend.image import show_image
from paz.abstract import GeneratingSequence
from paz.optimization import DiceLoss, FocalLoss, JaccardLoss
from models import PoseurSegmentation
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

image_shape = (128, 128, 3)
y_fov = 3.14159 / 4.0
distance = [0.7, 0.7]
light = [30, 30]
top_only = False
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
# args = (path, image_shape[:2], y_fov, distance, light, top_only, roll, shift)
scene = SingleView2(*args)
scene.scene.ambient_light = [1.0, 1.0, 1.0, 2.0]
image, alpha_channel, world_to_camera = scene.render()
# image, alpha_channel = scene.render()
processor = RandomSegmentationRender(scene, image_paths, occlusions)
sequence = GeneratingSequence(processor, batch_size, 1000)

model = PoseurSegmentation(image_shape, num_keypoints, 32)
optimizer = Adam()
# loss = DiceLoss() + JaccardLoss() + FocalLoss()
loss = {'mask': [DiceLoss(), JaccardLoss(), FocalLoss()]}
model.compile(optimizer, loss, metrics='mean_squared_error')

# setting callbacks
model_path = os.path.join(save_path, model.name)
if not os.path.exists(model_path):
    os.makedirs(model_path)
log = CSVLogger(os.path.join(model_path, model.name + '-optimization.log'))
save_path = os.path.join(model_path, 'weights.{epoch:02d}-{loss:.2f}.hdf5')
checkpoint = ModelCheckpoint(
    save_path, 'loss', verbose=1, save_weights_only=True)

model.fit(
    sequence,
    epochs=num_epochs,
    verbose=1,
    callbacks=[checkpoint, log],
    use_multiprocessing=multiprocessing,
    workers=workers)
