import processors as pe

from paz import processors as pr


class PreprocessHandPose(pr.SequentialProcessor):
    def __init__(self, image_size, num_keypoints=21):
        super(PreprocessHandPose, self).__init__()
        H, W = image_size
        topics = ['image', 'masks', 'keypoints3D', 'camera_matrix']
        self.add(pr.UnpackDictionary(topics))
        self.add(pr.ControlMap(PreprocessImage((H, W)), [0], [0]))
        self.add(pr.ControlMap(PreprocessMasks((H, W)), [1], [1]))
        self.add(pr.ControlMap(PreprocessKeypoints((224, 224)), [2, 3], [2]))
        self.add(pr.SequenceWrapper(
            {0: {'image': [H, W, 3]}},
            {1: {'masks': [H, W, 1]}, 2: {'keypoints2D': [num_keypoints, 2]}}))


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
    def __init__(self, image_size):
        super(PreprocessImage, self).__init__()
        self.add(pr.LoadImage())
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
