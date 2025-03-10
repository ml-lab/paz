import numpy as np
import tensorflow as tf
import cv2


def get_dims_x64(dims, multiple=64):
    dims = dims + (multiple - 1)
    floor_value = dims // multiple
    dims = floor_value * multiple
    return dims


def get_transformation_size(input_size, dims1, dims2):
    '''
    Resize the short side of the input image to 512 and keep
    the aspect ratio.
    '''
    dims1_resized = int(input_size)
    dims2_resized = input_size / dims1
    dims2_resized = dims2_resized * dims2
    dims2_resized = int(get_dims_x64(dims2_resized, 64))
    return dims1_resized, dims2_resized


def get_transformation_scale(dims1, dims1_resized, dims2_resized,
                             scaling_factor):
    scale_dims1 = dims1 / scaling_factor
    scale_dims2 = dims2_resized / dims1_resized
    scale_dims2 = scale_dims2 * dims1
    scale_dims2 = scale_dims2 / scaling_factor
    return scale_dims1, scale_dims2


def calculate_image_center(image):
    H, W = image.shape[:2]
    center_W = W / 2.0
    center_H = H / 2.0
    return center_W, center_H


def add_offset(x, offset):
    return (x + offset)


def rotate_point(point2D, rotation_angle):
    rotation_angle = np.pi * rotation_angle / 180
    sn, cs = np.sin(rotation_angle), np.cos(rotation_angle)
    x_rotated = (point2D[0] * cs) - (point2D[1] * sn)
    y_rotated = (point2D[0] * sn) + (point2D[1] * cs)
    return [x_rotated, y_rotated]


def calculate_third_point(point2D_a, point2D_b):
    diff = point2D_a - point2D_b
    return point2D_a + np.array([-diff[1], diff[0]], dtype=np.float32)


def get_input_image_points(scale, center, shift=np.array([0., 0.])):
    scale = scale * 200
    image_W = scale[0]
    image_dir = rotate_point([0, image_W * -0.5], 0)
    image = np.zeros((3, 2), dtype=np.float32)
    image[0, :] = center + scale * shift
    image[1, :] = center + image_dir + scale * shift
    image[2:, :] = calculate_third_point(image[0, :], image[1, :])
    return image


def get_output_image_points(output_size):
    W = output_size[0]
    H = output_size[1]
    image_dir = np.array([0, W * -0.5], np.float32)
    image = np.zeros((3, 2), dtype=np.float32)
    image[0, :] = [W * 0.5, H * 0.5]
    image[1, :] = np.array([W * 0.5, H * 0.5]) + image_dir
    image[2:, :] = calculate_third_point(image[0, :], image[1, :])
    return image


def imagenet_preprocess_input(image, data_format=None, mode='torch'):
    image = tf.keras.applications.imagenet_utils.preprocess_input(image,
                                                                  data_format,
                                                                  mode)
    return image


def resize_output(output, size):
    resized_output = []
    for image_arg, image in enumerate(output):
        resized_images = []
        for joint_arg in range(len(image)):
            resized = cv2.resize(output[image_arg][joint_arg], size)
            resized_images.append(resized)
        resized_images = np.stack(resized_images, axis=0)
    resized_output.append(resized_images)
    resized_output = np.stack(resized_output, axis=0)
    return resized_output
