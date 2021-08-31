import os
import glob
import numpy as np

from paz.backend.image import draw_circle


def list_filepaths(path, extensions):
    if not isinstance(extensions, list):
        raise ValueError('Extensions must be a list')
    wild_card = os.path.join(path, *extensions)
    filepaths = glob.glob(wild_card)
    return filepaths


def extend_path_by_split(path, split, split_to_name):
    name = split_to_name[split]
    return os.path.join(path, name)


def project_to_image_space(xyz, camera_matrix):
    """ Projects 3D coordinates into image space. """
    xyz = np.array(xyz)
    camera_matrix = np.array(camera_matrix)
    uvz = np.matmul(camera_matrix, xyz.T).T
    uv, z = uvz[:, :2], uvz[:, 2:3]
    return uv / z


def draw_keypoints2D(image, keypoints2D, colors, radius):
    for keypoint_arg, keypoint2D in enumerate(keypoints2D):
        color = colors[keypoint_arg]
        draw_circle(image, keypoint2D, color, radius)
    return image


def merge_masks(masks):
    mask = np.all(masks, axis=2, keepdims=True)
    return mask


def denormalize_image(image):
    return image * 255.0
