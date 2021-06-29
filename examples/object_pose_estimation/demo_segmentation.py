from paz.abstract import Processor, SequentialProcessor
from paz import processors as pr
from models import Poseur2D
from paz.backend.image import load_image, show_image, resize_image
import glob
import numpy as np
from paz.backend.camera import Camera
from paz.pipelines import DetectSingleShot
from paz.models.detection import SSD300
from models import PoseurSegmentation
from processors import DrawMask


class SSD300SolarPanel(DetectSingleShot):
    def __init__(self, weights_path, score_thresh=0.50,
                 nms_thresh=0.45, draw=True):
        class_names = ['background', 'solar_panel']
        model = SSD300(len(class_names), None, None)
        model.load_weights(weights_path)
        super(SSD300SolarPanel, self).__init__(
            model, class_names, score_thresh, nms_thresh, draw=draw)


class EstimateMask(Processor):
    def __init__(self, detect, segment, offsets):
        """Pose estimation pipeline using keypoints.

        # Arguments
            detect: Function that outputs a dictionary with a key
                ``boxes2D`` having a list of ``Box2D`` messages.
            segment: Function that outputs a dictionary
                with a key ``mask`` with numpy array as value
            camera: Instance of ``paz.backend.camera.Camera`` with
                camera intrinsics.
            offsets: List of floats indicating the scaled offset to
                be added to the ``Box2D`` coordinates.

        # Returns
            A function that takes an RGB image and outputs the following
            inferences as keys of a dictionary:
                ``image``, ``boxes2D``, ``keypoints`` and ``poses6D``.
        """
        super(EstimateMask, self).__init__()
        self.detect = detect
        self.segment = segment
        self.square = SequentialProcessor()
        self.square.add(pr.SquareBoxes2D())
        self.square.add(pr.OffsetBoxes2D(offsets))
        self.clip = pr.ClipBoxes2D()
        self.crop = pr.CropBoxes2D()
        self.wrap = pr.WrapOutput(['image', 'boxes2D', 'mask'])
        self.change_mask = ChangeMaskCoordinateSystem()
        self.draw_masks = DrawMask(1)

    def call(self, image):
        boxes2D = self.detect(image)['boxes2D']
        boxes2D = self.square(boxes2D)
        boxes2D = self.clip(image, boxes2D)
        cropped_images = self.crop(image, boxes2D)
        masks = []
        for cropped_image, box2D in zip(cropped_images, boxes2D):
            mask = self.segment(cropped_image)['mask']
            mask = self.change_mask(mask, box2D, image)
            image = self.draw_masks(image, mask)
            masks.append(mask)
        return self.wrap(image, boxes2D, masks)


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


class Segment(Processor):
    """ THIS DIFFERS from PAZ because it predicts masks
    Basic 2D keypoint prediction pipeline.

    # Arguments
        model: Keras model for predicting keypoints.
        num_keypoints: Int or None. If None ``num_keypoints`` is
            tried to be inferred from ``model.output_shape``
        draw: Boolean indicating if inferences should be drawn.
        radius: Int. used for drawing the predicted keypoints.
    """
    def __init__(self, model, color=pr.RGB2BGR):
        self.model = model
        self.color = color
        self.preprocess = SequentialProcessor()
        self.preprocess.add(pr.ResizeImage(self.model.input_shape[1:3]))
        self.preprocess.add(pr.ConvertColorSpace(self.color))
        self.preprocess.add(pr.NormalizeImage())
        self.preprocess.add(pr.ExpandDims(0))
        self.predict = pr.Predict(model, self.preprocess, pr.Squeeze(0))
        self.wrap = pr.WrapOutput(['image', 'mask'])

    def call(self, image):
        mask = self.predict(image)
        return self.wrap(image, mask)


model = PoseurSegmentation((128, 128, 3), 6, 32)
model.load_weights('trained_models/PoseurSegmentation/weights.07-0.04.hdf5')
# estimate_keypoints = EstimateKeypoints2D(model, 6)
weights_path = 'trained_models/SSD300/weights.141-2.66.hdf5'
detect = SSD300SolarPanel(weights_path)
segment = Segment(model)

offsets = [0.3, 0.3]
estimate_mask = EstimateMask(detect, segment, offsets)

image_paths = glob.glob('datasets/test_solar_panel/*.jpg')
for image_arg, image_path in enumerate(image_paths):
    image = load_image(image_path)
    results = estimate_mask(image)
    show_image(results['image'])
    masks = results['mask']
    if len(masks) != 0:
        mask = masks[0]
        mask = (255 * mask).astype('uint8')
        # show_image(mask)
    # write_image('results/image_%s.png' % image_arg, results['image'])
