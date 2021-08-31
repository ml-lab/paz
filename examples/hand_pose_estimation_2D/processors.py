from paz import processors as pr
from backend import merge_masks
from backend import project_to_image_space


class MergeMasks(pr.Processor):
    def __init__(self):
        super(MergeMasks, self).__init__()

    def call(self, masks):
        return merge_masks(masks)


class ProjectToImageSpace(pr.Processor):
    def __init__(self):
        super(ProjectToImageSpace, self).__init__()

    def call(self, keypoints3D, camera_matrix):
        return project_to_image_space(keypoints3D, camera_matrix)
