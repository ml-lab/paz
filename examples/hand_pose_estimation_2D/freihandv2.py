import os
import glob
import json
import numpy as np

from paz.abstract import Loader
import paz.processors as pr


def list_filepaths(path, extensions):
    if not isinstance(extensions, list):
        raise ValueError('Extensions must be a list')
    wild_card = os.path.join(path, *extensions)
    filepaths = glob.glob(wild_card)
    return filepaths


def extend_path_by_split(path, split, split_to_name):
    name = split_to_name[split]
    return os.path.join(path, name)


class FreiHANDV2(Loader):
    def __init__(self, path, split):
        self._split_to_name = {pr.TRAIN: 'training', pr.TEST: 'evaluation'}
        super(FreiHANDV2, self).__init__(path, split, None, 'FreiHANDV2')

    def load_data(self):
        images = self._extract_image_paths()
        masks = self._extract_mask_paths()
        keypoints = self._extract_keypoints()
        data = []
        for image, mask, keypointset in zip(images, masks, keypoints):
            sample = {'image': image, 'mask': mask, 'keypoints': keypointset}
            data.append(sample)
        return data

    def _extract_image_paths(self):
        path = extend_path_by_split(self.path, self.split, self._split_to_name)
        extensions = ['rgb', '*jpg']
        image_paths = list_filepaths(path, extensions)
        image_paths = sorted(image_paths)
        return image_paths

    def _extract_mask_paths(self):
        path = extend_path_by_split(self.path, self.split, self._split_to_name)
        extensions = ['mask', '*jpg']
        mask_paths = list_filepaths(path, extensions)
        mask_paths = sorted(mask_paths)
        # masks are repeated at every 32560 image sample
        return mask_paths + mask_paths + mask_paths + mask_paths

    def _extract_keypoints(self):
        internal_name = self._split_to_name[self.split]
        filename = os.path.join(self.path, internal_name + '_xyz.json')
        filedata = open(filename, 'r')
        keypoints = json.load(filedata)
        # labels are repeated at every 32560 image sample
        keypoints = keypoints + keypoints + keypoints + keypoints
        keypoints = np.array(keypoints)
        return keypoints


if __name__ == '__main__':
    user = os.path.expanduser('~')
    path = os.path.join(user, 'paz/examples/hand_pose_estimation_2D/dataset/')
    split = pr.TRAIN
    data_manager = FreiHANDV2(path, split)
    data = data_manager.load_data()
