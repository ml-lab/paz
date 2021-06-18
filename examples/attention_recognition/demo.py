import argparse
import numpy as np
import cv2
import math
from pipelines import AttentionKeypointNet2D32

import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


from paz.backend.camera import Camera
from paz.backend.camera import VideoPlayer
from paz.backend.image import load_image
# from paz.pipelines import HeadPoseKeypointNet2D32
from pose import HeadPoseKeypointNet2D32
from paz.applications import DetectMiniXceptionFER
from backend import draw_axis

description = 'Demo script for estimating 6D pose-heads from face-keypoints'
parser = argparse.ArgumentParser(description=description)
parser.add_argument('-c', '--camera_id', type=int, default=4,
                    help='Camera device ID')
parser.add_argument('-fl', '--focal_length', type=float, default=None,
                    help="Focal length in pixels. If ''None'' it's"
                    "approximated using the image width")
parser.add_argument('-ic', '--image_center', nargs='+', type=float,
                    default=None, help="Image center in pixels for internal"
                    "camera matrix. If ''None'' it's approximated using the"
                    "image center from an extracted frame.")
args = parser.parse_args()

# obtaining a frame to perform focal-length and camera center approximation
camera = Camera(args.camera_id)
camera.start()
image_size = camera.read().shape[0:2]
camera.stop()

# loading focal length or approximating it
focal_length = args.focal_length
if focal_length is None:
    focal_length = image_size[1]

# loading image/sensor center or approximating it
image_center = args.image_center
if args.image_center is None:
    image_center = (image_size[1] / 2.0, image_size[0] / 2.0)

# building camera parameters
camera.distortion = np.zeros((4, 1))
camera.intrinsics = np.array([[focal_length, 0, image_center[0]],
                              [0, focal_length, image_center[1]],
                              [0, 0, 1]])

pipeline = AttentionKeypointNet2D32(camera)
player = VideoPlayer((640, 480), pipeline, camera)
player.run()
"""
predict_pose = HeadPoseKeypointNet2D32(camera)
image = load_image('test_image.jpg')
results = predict_pose(image)
pose6D = results['poses6D'][0]
rotation = pose6D.rotation_vector
translation = pose6D.translation
image, axis_points2D = draw_axis(image, rotation, translation, camera)
from paz.backend.image import show_image
rvec_matrix = cv2.Rodrigues(rotation)[0]
proj_matrix = np.hstack((rvec_matrix, translation))
eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]

pitch = math.degrees(math.asin(math.sin(pitch)))
roll = -math.degrees(math.asin(math.sin(roll)))
yaw = math.degrees(math.asin(math.sin(yaw)))
print(roll, pitch, yaw)
show_image(image)

predict_emotion = DetectMiniXceptionFER()
image = load_image('test_image.jpg')
results = predict_emotion(image)
show_image(results['image'])
"""
