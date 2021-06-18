from paz.backend.image.draw import GREEN
from paz.backend.image.draw import draw_line, draw_dot
from paz.backend.keypoints import project_points3D
import numpy as np
import cv2


def draw_cube(image, points, color=GREEN, thickness=2, radius=5):
    """ Draws a cube in image.

    # Arguments
        image: Numpy array of shape ``[H, W, 3]``.
        points: List of length 8  having each element a list
            of length two indicating ``(y, x)`` openCV coordinates.
        color: List of length three indicating RGB color of point.
        thickness: Integer indicating the thickness of the line to be drawn.
        radius: Integer indicating the radius of corner points to be drawn.

    # Returns
        Numpy array with shape ``[H, W, 3]``. Image with cube.
    """
    # color = color[::-1]  # transform to BGR for openCV

    # draw bottom
    draw_line(image, points[0][0], points[1][0], color, thickness)
    draw_line(image, points[1][0], points[2][0], color, thickness)
    draw_line(image, points[3][0], points[2][0], color, thickness)
    draw_line(image, points[3][0], points[0][0], color, thickness)

    # draw top
    draw_line(image, points[4][0], points[5][0], color, thickness)
    draw_line(image, points[6][0], points[5][0], color, thickness)
    draw_line(image, points[6][0], points[7][0], color, thickness)
    draw_line(image, points[4][0], points[7][0], color, thickness)

    # draw sides
    draw_line(image, points[0][0], points[4][0], color, thickness)
    draw_line(image, points[7][0], points[3][0], color, thickness)
    draw_line(image, points[5][0], points[1][0], color, thickness)
    draw_line(image, points[2][0], points[6][0], color, thickness)

    # draw X mark on top
    # draw_line(image, points[0][0], points[2][0], color, thickness)
    # draw_line(image, points[1][0], points[3][0], color, thickness)
    draw_line(image, points[4][0], points[6][0], color, thickness)
    draw_line(image, points[5][0], points[7][0], color, thickness)

    # draw dots
    [draw_dot(image, np.squeeze(point), color, radius) for point in points]
    return image


def build_cube_points3D(height, width, translation=None):
    """ Build the 3D points of a cube using openCV coordinate system.
                   2--------3
                  /|       /|
                 / |      / |
                1--------4  |
                |  6_____|__7
                | /      | /
                |/       |/
                5--------8

    # Arguments
        height: float, height of the 3D box.
        width: float,  width of the 3D box.
        translation: Numpy array of flots of shape ``(3)'' corresponding
            to a translation in ``[x, y, z]'' using the openCV coordinate
            system.

    # Returns
        Numpy array of shape ``(8, 3)'' corresponding to 3D keypoints of a cube
    """
    half_height, half_width = height / 2.0, width / 2.0
    point_1 = [+half_width, -half_height, +half_width]
    point_2 = [+half_width, -half_height, -half_width]
    point_3 = [-half_width, -half_height, -half_width]
    point_4 = [-half_width, -half_height, +half_width]
    point_5 = [+half_width, +half_height, +half_width]
    point_6 = [+half_width, +half_height, -half_width]
    point_7 = [-half_width, +half_height, -half_width]
    point_8 = [-half_width, +half_height, +half_width]
    cube_points3D = [point_1, point_2, point_3, point_4,
                     point_5, point_6, point_7, point_8]
    cube_points3D = np.array(cube_points3D)
    if translation is not None:
        cube_points3D = cube_points3D + translation
    return cube_points3D


def draw_axis(image, rotation, translation, camera, thickness=3):
    # print('inside', rotation, translation)
    axis_points3D = np.float32([[1000, 0, 0], [0, 1000, 0], [0, 0, 1000], [0, 0, 0]])
    axis_points2D, jacobian = cv2.projectPoints(
        axis_points3D, rotation, translation,
        camera.intrinsics, camera.distortion)
    axis_points2D = np.squeeze(axis_points2D)
    origin_2D = axis_points2D[3]
    axis_x_2D = axis_points2D[0]
    axis_y_2D = axis_points2D[1]
    axis_z_2D = axis_points2D[2]
    image = draw_line(image, origin_2D, axis_x_2D, (255, 0, 0), thickness)
    image = draw_line(image, origin_2D, axis_y_2D, (0, 255, 0), thickness)
    image = draw_line(image, origin_2D, axis_z_2D, (0, 0, 255), thickness)
    # it would be a good idea to have a function that returns the axis too
    # return image, axis_points2D
    return image
