from tensorflow.keras.losses import Loss
import tensorflow as tf


class SymmetricPose(Loss):
    """Computes a symmetric pose loss.

    # Arguments
        invariant_transforms: List of numpy arrays. Each numpy array is a
            transformation that leads the pose invariant.
    """
    def __init__(self, compute_loss, invariant_transforms):
        super(SymmetricPose, self).__init__()
        self.compute_loss = compute_loss
        self.invariant_transforms = invariant_transforms

    def call(self, y_true, y_pred):
        valid_poses = []
        for transform in self.invariant_transforms:
            valid_pose = tf.matmul(transform, y_true)
            valid_poses.append(valid_pose)

        losses = []
        for valid_pose in valid_poses:
            loss = self.compute_loss(valid_pose, y_pred)
            losses.append(loss)

        return tf.min(losses)
