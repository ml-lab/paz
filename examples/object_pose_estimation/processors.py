import numpy as np

from paz.abstract import Processor, SequentialProcessor
import paz.processors as pr
from backend import apply_speckle_noise
from paz.models import Projector
from paz.pipelines import RandomizeRenderedImage


class _RandomKeypointsRender(pr.Processor):
    def __init__(self, scene, keypoints, image_paths, num_occlusions):
        super(_RandomKeypointsRender, self).__init__()
        self.render = pr.Render(scene)
        projector = self._build_projector(scene)
        self.project = pr.ProjectKeypoints(projector, keypoints)
        self.augment = RandomizeRenderedImage(image_paths, num_occlusions)
        self.remove_depth = pr.RemoveKeypointsDepth()
        self.augment.add(pr.NormalizeImage())

    def _build_projector(self, scene):
        focal_length = scene.camera.camera.get_projection_matrix()[0, 0]
        return Projector(focal_length, use_numpy=True)

    def call(self):
        image, alpha_mask, world_to_camera = self.render()
        input_image = self.augment(image, alpha_mask)
        keypoints = self.project(world_to_camera)
        keypoints = self.remove_depth(keypoints)
        return input_image, keypoints, alpha_mask / 255.0


class ApplySpeckleNoise(Processor):
    def __init__(self, probability=0.5, mean=0, variance=0.1):
        super(ApplySpeckleNoise, self).__init__()
        self.probability = probability
        self.mean = mean
        self.variance = variance

    def call(self, image):
        if self.probability >= np.random.rand():
            apply_speckle_noise(image, self.mean, self.variance)
        return image


class Keypoints2DEstimator(SequentialProcessor):
    def __init__(self, model):
        super(Keypoints2DEstimator, self).__init__()
        self.num_keypoints = model.output_shape[1]
        self.add(pr.ConvertColorSpace(pr.RGB2BGR))
        self.add(pr.ResizeImage(model.input_shape[1:3]))
        self.add(pr.NormalizeImage())
        self.add(pr.ExpandDims(0))
        self.add(pr.Predict(model))
        self.add(pr.Squeeze(0))


class PoseEstimator(Processor):
    def __init__(self, keypoint_predictor, keypoints3D, camera, dimensions):
        super(PoseEstimator, self).__init__()
        self.keypoint_predictor = keypoint_predictor
        self.dimensions = dimensions
        self.camera = camera
        self.solve_PNP = pr.SolvePNP(keypoints3D, camera)
        self.denormalize_keypoints = pr.DenormalizeKeypoints()
        self.change_coordinates = pr.ChangeKeypointsCoordinateSystem()

    def call(self, image, box2D):
        keypoints2D = self.keypoint_predictor(image)
        keypoints2D = self.denormalize_keypoints(keypoints2D, image)
        keypoints2D = self.change_coordinates(keypoints2D, box2D)
        pose6D = self.solve_PNP(keypoints2D)
        return keypoints2D, pose6D


class Pose6DPredictor(Processor):
    def __init__(self, detector, pose_estimator, valid_classes, offset_scales):
        super(Pose6DPredictor, self).__init__()
        self.detector, self.pose_estimator = detector, pose_estimator
        self.process_boxes = SequentialProcessor(
            [pr.FilterClassBoxes2D(valid_classes), pr.SquareBoxes2D()])
        self.clip_boxes = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()
        self.draw_boxes2D = pr.DrawBoxes2D(self.detector.class_names)
        self.draw_box3D = pr.DrawBoxes3D(
            self.pose_estimator.camera, self.pose_estimator.dimensions)
        num_keypoints = self.pose_estimator.keypoint_predictor.num_keypoints
        self.draw_keypoints = pr.DrawKeypoints2D(num_keypoints, 10, False)
        self.wrap = pr.WrapOutput(['image', 'boxes2D', 'poses6D'])

    def call(self, image):
        detector_results = self.detector(image)
        image, boxes2D = detector_results['image'], detector_results['boxes2D']
        if len(boxes2D) == 0:
            return self.wrap(image, None, None)
        valid_boxes2D = self.process_boxes(boxes2D)
        valid_boxes2D = self.clip_boxes(image, valid_boxes2D)
        cropped_images = self.crop(image, valid_boxes2D)
        poses6D = []
        for box2D, cropped_image in zip(valid_boxes2D, cropped_images):
            keypoints2D, pose6D = self.pose_estimator(cropped_image, box2D)
            image = self.draw_keypoints(image, keypoints2D)
            image = self.draw_box3D(image, pose6D)
            poses6D.append(pose6D)
        image = self.draw_boxes2D(image, valid_boxes2D)
        return self.wrap(image, boxes2D, poses6D)
