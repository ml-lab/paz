import argparse
from paz.backend.camera import VideoPlayer
from paz.backend.camera import Camera
from paz.pipelines import DetectKeypoints2D
from pipelines import HaarCascadeDetector
from paz.pipelines import EstimateKeypoints2D
from paz.models import KeypointNet2D
from paz.pipelines import DetectHaarCascade


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Real-time face classifier')
    parser.add_argument('-c', '--camera_id', type=int, default=0,
                        help='Camera device ID')
    parser.add_argument('-o', '--offset', type=float, default=0.1,
                        help='Scaled offset to be added to bounding boxes')
    args = parser.parse_args()

    # URL = 'https://raw.githubusercontent.com/Balaje/OpenCV/master/haarcascades/hand.xml'
    URL = 'https://raw.githubusercontent.com/heppu/Hand-recognition/master/xml/hand.xml'
    detector = HaarCascadeDetector(weights='hand.xml', URL=URL, class_arg=1)
    detect = DetectHaarCascade(detector, class_names=['background', 'hand'])
    model = KeypointNet2D((128, 128, 3), 21, 32)
    model.load_weights('HandKP_keypointnet2D_32_21_weights.hdf5')
    estimate_keypoints = EstimateKeypoints2D(model, 21, True)
    for processor in estimate_keypoints.preprocess.processors:
        print(processor.name)

    estimate_keypoints.preprocess.remove('ConvertColorSpace')
    """
    pipeline = DetectKeypoints2D(
        detect, estimate_keypoints, [args.offset, args.offset])
    print(detector.path)
    player = VideoPlayer((640, 480), pipeline, camera)
    """
    camera = Camera(args.camera_id)
    player = VideoPlayer((640, 480), estimate_keypoints, camera)
    player.run()
