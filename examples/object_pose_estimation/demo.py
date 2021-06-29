from paz.abstract import Processor, SequentialProcessor
from paz import processors as pr
from models import Poseur2D
from paz.backend.image import load_image, show_image, resize_image
import glob
import numpy as np
from paz.backend.camera import Camera
from paz.pipelines import DetectSingleShot
from paz.models.detection import SSD300


class SSD300SolarPanel(DetectSingleShot):
    def __init__(self, weights_path, score_thresh=0.50,
                 nms_thresh=0.45, draw=True):
        class_names = ['background', 'solar_panel']
        model = SSD300(len(class_names), None, None)
        model.load_weights(weights_path)
        super(SSD300SolarPanel, self).__init__(
            model, class_names, score_thresh, nms_thresh, draw=draw)


class EstimatePoseKeypoints(Processor):
    def __init__(self, detect, estimate_keypoints, camera, offsets,
                 model_points, class_to_dimensions, radius=3, thickness=1):
        """Pose estimation pipeline using keypoints.

        # Arguments
            detect: Function that outputs a dictionary with a key
                ``boxes2D`` having a list of ``Box2D`` messages.
            estimate_keypoints: Function that outputs a dictionary
                with a key ``keypoints`` with numpy array as value
            camera: Instance of ``paz.backend.camera.Camera`` with
                camera intrinsics.
            offsets: List of floats indicating the scaled offset to
                be added to the ``Box2D`` coordinates.
            model_points: Numpy array of shape ``(num_keypoints, 3)``
                indicating the 3D coordinates of the predicted keypoints
                from the ``esimate_keypoints`` function.
            class_to_dimensions: Dictionary with keys being the class labels
                of the predicted ``Box2D`` messages and the values a list of
                two integers indicating the height and width of the object.
                e.g. {'PowerDrill': [30, 20]}.
            radius: Int. radius of keypoint to be drawn.
            thickness: Int. thickness of 3D box.

        # Returns
            A function that takes an RGB image and outputs the following
            inferences as keys of a dictionary:
                ``image``, ``boxes2D``, ``keypoints`` and ``poses6D``.
        """
        super(EstimatePoseKeypoints, self).__init__()
        self.num_keypoints = estimate_keypoints.num_keypoints
        self.detect = detect
        self.estimate_keypoints = estimate_keypoints
        self.square = SequentialProcessor()
        self.square.add(pr.SquareBoxes2D())
        self.square.add(pr.OffsetBoxes2D(offsets))
        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()
        self.change_coordinates = pr.ChangeKeypointsCoordinateSystem()
        self.solve_PNP = pr.SolvePNP(model_points, camera)
        self.draw_keypoints = pr.DrawKeypoints2D(self.num_keypoints, radius)
        self.draw_box = pr.DrawBoxes3D(camera, class_to_dimensions, thickness)
        outputs = ['image', 'boxes2D', 'keypoints', 'poses6D', 'mask']
        self.wrap = pr.WrapOutput(outputs)
        self.change_mask = ChangeMaskCoordinateSystem()

    def call(self, image):
        boxes2D = self.detect(image)['boxes2D']
        boxes2D = self.square(boxes2D)
        boxes2D = self.clip(image, boxes2D)
        cropped_images = self.crop(image, boxes2D)
        poses6D, keypoints2D, masks = [], [], []
        for cropped_image, box2D in zip(cropped_images, boxes2D):
            crop_results = self.estimate_keypoints(cropped_image)
            keypoints, mask = crop_results['keypoints'], crop_results['mask']
            keypoints = self.change_coordinates(keypoints, box2D)
            pose6D = self.solve_PNP(keypoints)
            pose6D.class_name = box2D.class_name
            image_with_results = self.draw_keypoints(image, keypoints)
            image_with_results = self.draw_box(image_with_results, pose6D)
            keypoints2D.append(keypoints)
            poses6D.append(pose6D)
            mask = self.change_mask(mask, box2D, image_with_results)
            masks.append(mask)
        return self.wrap(image, boxes2D, keypoints2D, poses6D, masks)


class ChangeMaskCoordinateSystem(Processor):
    """Changes ``keypoints`` 2D coordinate system using ``box2D`` coordinates
        to locate the new origin at the openCV image origin (top-left).
    """
    def __init__(self):
        super(ChangeMaskCoordinateSystem, self).__init__()

    def call(self, mask, box2D, image):
        x_min, y_min, x_max, y_max = box2D.coordinates
        H = y_max - y_min
        W = x_max - x_min
        image = np.zeros(image.shape[:2])
        mask = resize_image(mask, (W, H))
        print('mask', mask.shape)
        print('image', image.shape)
        image[y_min:y_max, x_min:x_max] = mask
        return image


class EstimateKeypoints2D(Processor):
    """ THIS DIFFERS from PAZ because it predicts masks
    Basic 2D keypoint prediction pipeline.

    # Arguments
        model: Keras model for predicting keypoints.
        num_keypoints: Int or None. If None ``num_keypoints`` is
            tried to be inferred from ``model.output_shape``
        draw: Boolean indicating if inferences should be drawn.
        radius: Int. used for drawing the predicted keypoints.
    """
    def __init__(self, model, num_keypoints, draw=True, radius=3,
                 color=pr.RGB2BGR):
        self.model = model
        self.num_keypoints = num_keypoints
        self.draw, self.radius, self.color = draw, radius, color
        self.preprocess = SequentialProcessor()
        self.preprocess.add(pr.ResizeImage(self.model.input_shape[1:3]))
        # self.preprocess.add(pr.ConvertColorSpace(self.color))
        self.preprocess.add(pr.NormalizeImage())
        self.preprocess.add(pr.ExpandDims(0))
        self.preprocess.add(pr.ExpandDims(-1))
        # self.predict = pr.Predict(model, self.preprocess, pr.Squeeze(0))
        self.predict = pr.Predict(model, self.preprocess)
        self.denormalize = pr.DenormalizeKeypoints()
        self.draw = pr.DrawKeypoints2D(self.num_keypoints, self.radius, False)
        self.wrap = pr.WrapOutput(['image', 'keypoints', 'mask'])

    def call(self, image):
        keypoints, mask = self.predict(image)
        keypoints = np.squeeze(keypoints)
        mask = np.squeeze(mask)

        keypoints = self.denormalize(keypoints, image)
        if self.draw:
            image = self.draw(image, keypoints)
        return self.wrap(image, keypoints, mask)


model = Poseur2D((128, 128, 3), 6, True, 32)
model.load_weights('trained_models/Poseur2D/weights.12-0.20.hdf5')
estimate_keypoints = EstimateKeypoints2D(model, 6)
weights_path = 'trained_models/SSD300/weights.141-2.66.hdf5'
detect = SSD300SolarPanel(weights_path)

camera = Camera(0)
image_size = [720, 1280]
# image_size = [1280, 720]
focal_length = image_size[1]
image_center = (image_size[1] / 2.0, image_size[0] / 2.0)

# building camera parameters
camera.distortion = np.zeros((4, 1))
camera.intrinsics = np.array([[focal_length, 0, image_center[0]],
                              [0, focal_length, image_center[1]],
                              [0, 0, 1]])
offsets = [0.3, 0.3]
class_to_dimensions = {'solar_panel': [0.25, 0.25]}
model_points = np.zeros((6, 3))
radius = 0.25
angles = np.linspace(0, 2 * np.pi, 6, endpoint=False)
for keypoint_arg, angle in enumerate(angles):
    x = radius * np.cos(angle)
    y = radius * np.sin(angle)
    model_points[keypoint_arg] = x, y, 0.0

estimate_pose = EstimatePoseKeypoints(
    detect, estimate_keypoints, camera, offsets,
    model_points, class_to_dimensions,
    radius=3, thickness=1)

image_paths = glob.glob('datasets/test_solar_panel/*.jpg')
for image_arg, image_path in enumerate(image_paths):
    image = load_image(image_path)
    results = estimate_pose(image)
    show_image(results['image'])
    masks = results['mask']
    if len(masks) != 0:
        mask = masks[0]
        mask = (255 * mask).astype('uint8')
        show_image(mask)
    # write_image('results/image_%s.png' % image_arg, results['image'])
