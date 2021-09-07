import processors as pe

from paz import processors as pr
from tensorflow.keras.utils import get_file
import cv2
import numpy as np
import os


class PreprocessHandPose(pr.SequentialProcessor):
    def __init__(self, image_size, split, num_keypoints=21):
        super(PreprocessHandPose, self).__init__()
        H, W = image_size
        topics = ['image', 'masks', 'keypoints3D', 'camera_matrix']
        self.add(pr.UnpackDictionary(topics))
        self.add(pr.ControlMap(PreprocessImage((H, W), split), [0], [0]))
        self.add(pr.ControlMap(PreprocessMasks((H, W)), [1], [1]))
        self.add(pr.ControlMap(PreprocessKeypoints((224, 224)), [2, 3], [2]))
        """
        self.add(pr.SequenceWrapper(
            {0: {'image': [H, W, 3]}},
            {1: {'masks': [H, W, 1]}, 2: {'keypoints2D': [num_keypoints, 2]}}))
        """
        self.add(pr.SequenceWrapper(
            {0: {'image': [H, W, 3]}},
            {2: {'keypoints2D': [num_keypoints, 2]}}))


class PreprocessMasks(pr.SequentialProcessor):
    def __init__(self, image_size):
        super(PreprocessMasks, self).__init__()
        self.add(pr.LoadImage())
        self.add(pr.ResizeImage(image_size))
        self.add(pr.NormalizeImage())
        self.add(pr.CastImage(bool))
        self.add(pe.MergeMasks())
        self.add(pr.CastImage(float))


class PreprocessImage(pr.SequentialProcessor):
    def __init__(self, image_size, split):
        super(PreprocessImage, self).__init__()
        self.add(pr.LoadImage())
        if split == pr.TRAIN:
            self.add(pr.RandomContrast())
            self.add(pr.RandomBrightness())
            self.add(pr.RandomSaturation())
            self.add(pr.RandomHue())
        self.add(pr.ResizeImage(image_size))
        self.add(pr.NormalizeImage())


class PreprocessKeypoints(pr.SequentialProcessor):
    def __init__(self, original_image_size):
        super(PreprocessKeypoints, self).__init__()
        self.add(pe.ProjectToImageSpace())
        self.add(pr.NormalizeKeypoints(original_image_size))


class HaarCascadeDetector(object):
    """Haar cascade face detector.

    # Arguments
        path: String. Postfix to default openCV haarcascades XML files, see [1]
            e.g. `eye`, `frontalface_alt2`, `fullbody`
        class_arg: Int. Class label argument.
        scale = Float. Scale for image reduction
        neighbors: Int. Minimum neighbors

    # Reference
        - [Haar
            Cascades](https://github.com/opencv/opencv/tree/master/data/haarcascades)
    """

    def __init__(self, weights='haarcascade_frontalface_default.xml',
                 class_arg=None, scale=1.3, neighbors=5,
                 URL=('https://raw.githubusercontent.com/opencv/opencv/'
                      'master/data/haarcascades/')):
        self.weights = weights
        self.URL = URL
        # self.url = os.path.join(URL, self.weights)
        self.path = get_file(self.weights, self.URL, cache_subdir='paz/models')
        self.model = cv2.CascadeClassifier(self.path)
        self.class_arg = class_arg
        self.scale = scale
        self.neighbors = neighbors

    def predict(self, gray_image):
        """ Detects faces from gray images.

        # Arguments
            gray_image: Numpy array of shape ``(H, W, 2)``.

        # Returns
            Numpy array of shape ``(num_boxes, 4)``.
        """
        if len(gray_image.shape) != 2:
            raise ValueError('Invalid gray image shape:', gray_image.shape)
        args = (gray_image, self.scale, self.neighbors)
        boxes = self.model.detectMultiScale(*args)
        boxes_point_form = np.zeros_like(boxes)
        if len(boxes) != 0:
            boxes_point_form[:, 0] = boxes[:, 0]
            boxes_point_form[:, 1] = boxes[:, 1]
            boxes_point_form[:, 2] = boxes[:, 0] + boxes[:, 2]
            boxes_point_form[:, 3] = boxes[:, 1] + boxes[:, 3]
            if self.class_arg is not None:
                class_args = np.ones((len(boxes_point_form), 1))
                class_args = class_args * self.class_arg
                boxes_point_form = np.hstack((boxes_point_form, class_args))
        return boxes_point_form.astype('int')
