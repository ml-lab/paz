from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Permute
from tensorflow.keras.layers import Multiply
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import Activation
from tensorflow.keras.models import Model
from paz.models.layers import ExpectedValue2D


def block(x, num_filters, dilation_rate, alpha, name, kernel_size=(3, 3)):
    kwargs = {'dilation_rate': dilation_rate, 'padding': 'same', 'name': name}
    x = Conv2D(num_filters, kernel_size, **kwargs)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(alpha)(x)
    return x


def Poseur2D(input_shape, num_keypoints, mask, filters=64, alpha=0.1):
    """Model for discovering keypoint locations in 2D space, modified from

    # Arguments
        input_shape: List of integers indicating ``[H, W, num_channels]``.
        num_keypoints: Int. Number of keypoints to discover.
        filters: Int. Number of filters used in convolutional layers.
        alpha: Float. Alpha parameter of leaky relu.

    # Returns
        Keras/tensorflow model

    # References
        - [Discovery of Latent 3D Keypoints via End-to-end
            Geometric Reasoning](https://arxiv.org/abs/1807.03146)
    """
    width, height = input_shape[:2]
    base = input_tensor = Input(input_shape, name='image')
    for base_arg, rate in enumerate([1, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1]):
        name = 'conv2D_base-%s' % base_arg
        base = block(base, filters, (rate, rate), alpha, name)
    if mask:
        x = Conv2D(1, (3, 3), padding='same', name='conv2D_mas')(base)
        mask = Activation('sigmoid', name='mask')(x)
    name = 'uv_volume_features-%s'
    uv_volume = Conv2D(num_keypoints, (3, 3),
                       padding='same', name=name % 0)(base)
    uv_volume = Permute([3, 1, 2], name=name % 1)(uv_volume)
    volume_shape = [num_keypoints, width * height]
    uv_volume = Reshape(volume_shape, name=name % 2)(uv_volume)
    uv_volume = Activation('softmax', name=name % 3)(uv_volume)
    volume_shape = [num_keypoints, width, height]
    uv_volume = Reshape(volume_shape, name='uv_volume')(uv_volume)
    keypoints = ExpectedValue2D(name='keypoints')(uv_volume)
    outputs = [keypoints]
    if mask:
        outputs.append([mask])
    model = Model(input_tensor, outputs, name='Poseur2D')
    return model


def Poseur2DX(input_shape, num_keypoints, mask, filters=64, alpha=0.1):
    """Model for discovering keypoint locations in 2D space, modified from

    # Arguments
        input_shape: List of integers indicating ``[H, W, num_channels]``.
        num_keypoints: Int. Number of keypoints to discover.
        filters: Int. Number of filters used in convolutional layers.
        alpha: Float. Alpha parameter of leaky relu.

    # Returns
        Keras/tensorflow model

    # References
        - [Discovery of Latent 3D Keypoints via End-to-end
            Geometric Reasoning](https://arxiv.org/abs/1807.03146)
    """
    width, height = input_shape[:2]
    base = input_tensor = Input(input_shape, name='image')
    for base_arg, rate in enumerate([1, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1]):
        name = 'conv2D_base-%s' % base_arg
        base = block(base, filters, (rate, rate), alpha, name)
    if mask:
        x = Conv2D(1, (3, 3), padding='same', name='conv2D_seg')(base)
        mask = Activation('sigmoid', name='mask')(x)
    x = Multiply()([mask, input_tensor])
    for base_arg, rate in enumerate([1, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1]):
        name = 'conv2D_base-2-%s' % base_arg
        x = block(x, filters, (rate, rate), alpha, name)
    name = 'uv_volume_features-%s'
    uv_volume = Conv2D(num_keypoints, (3, 3),
                       padding='same', name=name % 0)(x)
    uv_volume = Permute([3, 1, 2], name=name % 1)(uv_volume)
    volume_shape = [num_keypoints, width * height]
    uv_volume = Reshape(volume_shape, name=name % 2)(uv_volume)
    uv_volume = Activation('softmax', name=name % 3)(uv_volume)
    volume_shape = [num_keypoints, width, height]
    uv_volume = Reshape(volume_shape, name='uv_volume')(uv_volume)
    keypoints = ExpectedValue2D(name='keypoints')(uv_volume)
    outputs = [keypoints, mask]
    model = Model(input_tensor, outputs, name='Poseur2DX')
    return model


def PoseurSegmentation(input_shape, num_keypoints, filters=32, alpha=0.1):
    """Model for discovering keypoint locations in 2D space, modified from

    # Arguments
        input_shape: List of integers indicating ``[H, W, num_channels]``.
        num_keypoints: Int. Number of keypoints to discover.
        filters: Int. Number of filters used in convolutional layers.
        alpha: Float. Alpha parameter of leaky relu.

    # Returns
        Keras/tensorflow model

    # References
        - [Discovery of Latent 3D Keypoints via End-to-end
            Geometric Reasoning](https://arxiv.org/abs/1807.03146)
    """
    width, height = input_shape[:2]
    base = input_tensor = Input(input_shape, name='image')
    for base_arg, rate in enumerate([1, 1, 2, 4, 8, 16, 1, 2, 4, 8, 16, 1]):
        name = 'conv2D_base-%s' % base_arg
        base = block(base, filters, (rate, rate), alpha, name)
    x = Conv2D(1, (3, 3), padding='same', name='conv2D_mask')(base)
    mask = Activation('sigmoid', name='mask')(x)
    model = Model(input_tensor, mask, name='PoseurSegmentation')
    return model


if __name__ == "__main__":
    import numpy as np
    poseur = Poseur2D((128, 128, 3), 10, True, 38)
    poseur.summary()

    poseurx = Poseur2DX((128, 128, 3), 10, True, 38)
    results = poseurx.predict(np.ones((3, 128, 128, 3)))
    poseurx.summary()
