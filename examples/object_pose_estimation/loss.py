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


class InvariantMSE(Loss):
    """Computes a symmetric pose loss.

    # Arguments
        invariant_transforms: List of numpy arrays. Each numpy array is a
            transformation that leads the pose invariant.
    """
    def __init__(self):
        super(InvariantMSE, self).__init__()

    def call(self, y_true, y_pred):
        """ Computes invariant mean squared error.
        # Arguments
            y_true: tensor of shape
                ``[batch_size, num_invariants, num_keypoints, 4]``.
            y_pred: tensor of shape ``[batch_size, 4]``.
        """
        y_pred = tf.expand_dims(y_pred, 1)
        keypoints_loss_all = tf.keras.losses.mean_squared_error(y_true, y_pred)
        keypoints_loss_closest = tf.reduce_min(keypoints_loss_all, axis=1)
        keypoints_loss_closest = tf.reduce_mean(keypoints_loss_closest, axis=1)
        return keypoints_loss_closest
