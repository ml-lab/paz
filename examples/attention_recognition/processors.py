from paz.processors import Processor
from paz.backend.keypoints import project_points3D
from backend import draw_cube
from backend import draw_axis
from backend import build_cube_points3D
import numpy as np


class DrawBoxes3D(Processor):
    def __init__(self, camera, class_to_dimensions, thickness=1):
        """Draw boxes 3D of multiple objects

        # Arguments
            camera: Instance of ``paz.backend.camera.Camera''.
            class_to_dimensions: Dictionary that has as keys the
                class names and as value a list [model_height, model_width]
            thickness: Int. Thickness of 3D box
        """
        super(DrawBoxes3D, self).__init__()
        self.camera = camera
        self.class_to_dimensions = class_to_dimensions
        self.class_to_points3D = self._build_class_to_points3D(
            self.class_to_dimensions)
        self.thickness = thickness

    def _build_class_to_points3D(self, class_to_dimensions):
        class_to_points3D = {}
        for class_name, dimensions in self.class_to_dimensions.items():
            height, width = dimensions
            class_to_points3D[class_name] = build_cube_points3D(
                height, width, np.array([0.0, 0.0, -1000.0]))
        return class_to_points3D

    def call(self, image, pose6D):
        points3D = self.class_to_points3D[pose6D.class_name]
        points2D = project_points3D(points3D, pose6D, self.camera)
        points2D = points2D.astype(np.int32)
        image = draw_cube(image, points2D, thickness=self.thickness)
        return image


class DrawAxis(Processor):
    def __init__(self, camera, thickness=1):
        """Draws 3D axis

        # Arguments
            camera: Instance of ``paz.backend.camera.Camera''.
            thickness: Int. Thickness of 3D box
        """
        super(DrawAxis, self).__init__()
        self.camera = camera
        self.thickness = thickness

    def call(self, image, pose6D):
        rotation, translation = pose6D.rotation_vector, pose6D.translation
        image = draw_axis(image, rotation, translation,
                          self.camera, self.thickness)
        return image
