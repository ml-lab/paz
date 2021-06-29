from paz import processors as pr

from processors import _RandomKeypointsRender
from processors import _RandomSegmentationRender


class RandomKeypointsRender(pr.SequentialProcessor):
    def __init__(self, scene, keypoints, image_paths, num_occlusions):
        super(RandomKeypointsRender, self).__init__()
        args = [scene, keypoints, image_paths, num_occlusions]
        H, W = scene.viewport_size
        self.add(_RandomKeypointsRender(*args))
        self.add(pr.SequenceWrapper({0: {'image': [H, W, 3]}},
                                    {1: {'keypoints': [len(keypoints), 2]},
                                     2: {'mask': [H, W, 1]}}))


class RandomSegmentationRender(pr.SequentialProcessor):
    def __init__(self, scene, image_paths, num_occlusions):
        super(RandomSegmentationRender, self).__init__()
        args = [scene, image_paths, num_occlusions]
        H, W = scene.viewport_size
        self.add(_RandomSegmentationRender(*args))
        self.add(pr.SequenceWrapper({0: {'image': [H, W, 3]}},
                                    {1: {'mask': [H, W, 1]}}))


class DrawNormalizedKeypoints(pr.Processor):
    def __init__(self, num_keypoints, radius=3, image_normalized=False):
        super(DrawNormalizedKeypoints, self).__init__()
        self.denormalize = pr.DenormalizeKeypoints()
        self.remove_depth = pr.RemoveKeypointsDepth()
        self.draw = pr.DrawKeypoints2D(num_keypoints, radius, image_normalized)

    def call(self, image, keypoints):
        keypoints = self.denormalize(keypoints, image)
        keypoints = self.remove_depth(keypoints)
        image = self.draw(image, keypoints)
        return image
