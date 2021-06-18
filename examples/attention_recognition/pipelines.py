from paz.abstract import Processor
# from paz.pipelines import HeadPoseKeypointNet2D32
from pose import HeadPoseKeypointNet2D32
import paz.processors as pr
from processors import DrawAxis
import math
from paz.backend.image import put_text
import cv2
import numpy as np


class AttentionKeypointNet2D32(Processor):
    def __init__(self, camera):
        super(AttentionKeypointNet2D32, self).__init__()
        self.predict_pose = HeadPoseKeypointNet2D32(camera)
        self.draw_axis = DrawAxis(camera)

    def call(self, image):
        pose_results = self.predict_pose(image)
        poses6D = pose_results['poses6D']
        if len(poses6D) == 0:
            return {'image': image}
        image = pose_results['image']
        pose6D = pose_results['poses6D'][0]
        rotation = pose6D.rotation_vector
        translation = pose6D.translation
        image = self.draw_axis(image, pose6D)

        rvec_matrix = cv2.Rodrigues(rotation)[0]
        proj_matrix = np.hstack((rvec_matrix, translation))
        eulerAngles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        pitch, yaw, roll = [math.radians(_) for _ in eulerAngles]
        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
        """
        rotation_matrix = cv2.Rodrigues(rotation)[0]
        proj_matrix = np.hstack((rotation_matrix, translation))
        euler_angles = cv2.decomposeProjectionMatrix(proj_matrix)[6]
        pitch, yaw, roll = [math.radians(angle) for angle in euler_angles]
        pitch = math.degrees(math.asin(math.sin(pitch)))
        roll = -math.degrees(math.asin(math.sin(roll)))
        yaw = math.degrees(math.asin(math.sin(yaw)))
        """
        roll = roll - 7.0
        pitch = pitch + 7.0
        yaw = yaw + 2.0
        put_text(image, 'R: %.2f' % roll, (20, 20), 0.8, pr.GREEN, 2)
        put_text(image, 'P: %.2f' % pitch, (20, 50), 0.8, pr.GREEN, 2)
        put_text(image, 'Y: %.2f' % yaw, (20, 80), 0.8, pr.GREEN, 2)
        # beta_1, beta_2, beta_3 = 0.33, 0.33, 0.33
        beta_1 = 1 / (2 * np.pi)
        beta_2 = 1 / (2 * np.pi)
        beta_3 = 1 / (2 * np.pi)
        attention = (beta_1 * roll) + (beta_2 * pitch) + (beta_3 * yaw)
        attention = np.linalg.norm(attention)
        attention = 1 / (1 + np.exp(attention - 1.0))
        # attention = 1 - attention
        put_text(image, 'Attn: %.2f' % attention, (200, 20), 0.8, pr.GREEN, 2)
        return {'image': image, 'euler_angles': [roll, pitch, yaw]}
