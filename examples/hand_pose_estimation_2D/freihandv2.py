import os
import json
import numpy as np

from paz.abstract import Loader
import paz.processors as pr

from backend import extend_path_by_split, list_filepaths


class FreiHANDV2(Loader):
    def __init__(self, path, split):
        self._split_to_name = {pr.TRAIN: 'training', pr.TEST: 'evaluation'}
        self._topics = ['image', 'masks', 'keypoints3D', 'camera_matrix']
        super(FreiHANDV2, self).__init__(path, split, None, 'FreiHANDV2')

    def load_data(self):
        camera_matrices = self._extract_camera_matrices()
        keypoints = self._extract_keypoints()
        images = self._extract_image_paths()
        masks = self._extract_masks_paths()
        data = []
        data_iterator = zip(images, masks, keypoints, camera_matrices)
        for sample_data in data_iterator:
            sample = dict(zip(self._topics, sample_data))
            data.append(sample)
        return data

    def _extract_filepaths(self, extensions):
        path = extend_path_by_split(self.path, self.split, self._split_to_name)
        paths = list_filepaths(path, extensions)
        paths = sorted(paths)
        return paths

    def _extract_image_paths(self):
        return self._extract_filepaths(['rgb', '*jpg'])

    def _extract_masks_paths(self):
        mask_paths = self._extract_filepaths(['mask', '*jpg'])
        # masks are repeated four times
        return mask_paths + mask_paths + mask_paths + mask_paths

    def _extract_json(self, extension):
        internal_name = self._split_to_name[self.split]
        filename = os.path.join(self.path, internal_name + extension)
        filedata = open(filename, 'r')
        data = json.load(filedata)
        # labels are repeated four times
        data = data + data + data + data
        return np.array(data)

    def _extract_keypoints(self):
        return self._extract_json('_xyz.json')

    def _extract_camera_matrices(self):
        return self._extract_json('_K.json')


if __name__ == '__main__':
    from backend import project_to_image_space
    from backend import draw_keypoints2D
    from pipelines import PreprocessHandPose
    from paz.backend.image import lincolor, load_image, show_image
    from paz.backend.keypoints import denormalize_keypoints
    user = os.path.expanduser('~')
    path = os.path.join(user, 'paz/examples/hand_pose_estimation_2D/dataset/')
    split = pr.TRAIN
    data_manager = FreiHANDV2(path, split)
    data = data_manager.load_data()
    preprocess = PreprocessHandPose((128, 128))
    for sample in data:
        preprocessed_sample = preprocess(sample)
        image = preprocessed_sample['inputs']['image']
        keypoints2D = preprocessed_sample['labels']['keypoints2D']
        image = (image * 255.0).astype('uint8')
        H, W = image.shape[:2]
        keypoints2D = denormalize_keypoints(keypoints2D, H, W)
        keypoints2D = keypoints2D.astype(int)
        draw_keypoints2D(image, keypoints2D, lincolor(21), 5)
        show_image(image)
