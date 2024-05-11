import sys
from argparse import Namespace
import tensorflow as tf
import tensorflow_hub as hub
import sqlite3
from pathlib import Path
import datetime
import torch
import posenet
import os
import cv2
import numpy as np
import alphapose_api
from alphapose.utils.config import update_config


UNIVERSAL_POINTS_NUMBER = 17


class Library:
    ID = -1

    def __init__(self, confidence=0.3):
        self.confidence = confidence

    def get_skeleton_data(self, image):
        return 0

    def convert(self, points):
        return points


class Movenet(Library):
    ID = 0
    POINTS_NUMBER = 17
    HIP_COEFFICIENT = 1.01
    SHOULDER_COEFFICIENT_Y = 1.05
    SHOULDER_COEFFICIENT_X = 1.01
    CONNECTIONS = [[0, 1], [1, 3], [0, 2], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                   [6, 12], [12, 14], [14, 16], [5, 11], [11, 13], [13, 15], [11, 12]]

    def __init__(self, confidence=0.3,
                 model_url="https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/multipose-lightning/versions/1"):
        super().__init__(confidence=confidence)
        model = hub.load(model_url)
        self.movenet = model.signatures['serving_default']

    def get_skeleton_data(self, image):
        image_width, image_height = image.shape[1], image.shape[0]

        input_image = cv2.resize(image, dsize=(256, 256))
        input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
        input_image = input_image.reshape(-1, 256, 256, 3)
        input_image = tf.cast(input_image, dtype=tf.int32)

        outputs = self.movenet(input_image)

        key_points_with_scores = outputs['output_0'].numpy()
        key_points_with_scores = np.squeeze(key_points_with_scores)

        key_points_list = []
        for key_points_with_score in key_points_with_scores:
            key_points = np.ndarray([self.POINTS_NUMBER, 3])
            for index in range(self.POINTS_NUMBER):
                key_points[index][0] = round(image_width *
                                             key_points_with_score[(index * 3) + 1])
                key_points[index][1] = round(image_height *
                                             key_points_with_score[(index * 3) + 0])
                key_points[index][2] = key_points_with_score[(index * 3) + 2]
            key_points_list.append(key_points)
        return key_points_list

    def convert(self, points):
        """ Преобразование скелетной модели posenet (movenet) к универсальной модели """
        new_points = np.ndarray([UNIVERSAL_POINTS_NUMBER, 3])
        new_points[3] = get_middle_point([points[0], points[1], points[2], points[3], points[4]])
        new_points[16] = get_middle_point([points[5], points[6]])
        new_points[4] = points[6]
        new_points[4][0] = points[6][0] * self.SHOULDER_COEFFICIENT_X
        new_points[4][1] = points[6][1] * self.SHOULDER_COEFFICIENT_Y
        new_points[5] = points[8]
        new_points[6] = points[10]
        new_points[10] = points[12]
        new_points[10][0] = points[12][0] * self.HIP_COEFFICIENT
        new_points[11] = points[14]
        new_points[12] = points[16]
        new_points[7] = points[5]
        new_points[7][0] = points[5][0] / self.SHOULDER_COEFFICIENT_X
        new_points[7][1] = points[5][1] * self.SHOULDER_COEFFICIENT_Y
        new_points[8] = points[7]
        new_points[9] = points[9]
        new_points[13] = points[11]
        new_points[13][0] = points[11][0] / self.HIP_COEFFICIENT
        new_points[14] = points[13]
        new_points[15] = points[15]
        new_points[0] = get_middle_point([points[11], points[12]])
        new_points[1] = get_middle_point([new_points[16], new_points[0]])
        new_points[2] = get_middle_point([new_points[3], new_points[16]])
        return new_points


class Posenet(Library):
    ID = 1
    POINTS_NUMBER = 17
    HIP_COEFFICIENT = 1.01
    SHOULDER_COEFFICIENT_Y = 1.05
    SHOULDER_COEFFICIENT_X = 1.01
    CONNECTIONS = [[0, 1], [1, 3], [0, 2], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                   [6, 12], [12, 14], [14, 16], [5, 11], [11, 13], [13, 15], [11, 12]]

    def __init__(self, confidence=0.05, model_id=101, model_dir='/posenet/_models',
                 max_pose_detections=10, min_pose_score=0.25):
        super().__init__(confidence=confidence)
        self.sess = tf.compat.v1.Session()
        self.model_cfg, self.model_outputs = posenet.load_model(model_id, self.sess, model_dir)
        self.output_stride = self.model_cfg['output_stride']
        self.max_pose_detections = max_pose_detections
        self.min_pose_score = min_pose_score

    def get_skeleton_data(self, image):
        with tf.compat.v1.Session() as sess:
            input_image, draw_image, output_scale = posenet.read_img(
                image, output_stride=self.output_stride)
            # image, scale_factor=args.scale_factor, output_stride=output_stride)

            heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
                self.model_outputs,
                feed_dict={'image:0': input_image}
            )

            pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
                heatmaps_result.squeeze(axis=0),
                offsets_result.squeeze(axis=0),
                displacement_fwd_result.squeeze(axis=0),
                displacement_bwd_result.squeeze(axis=0),
                output_stride=self.output_stride,
                max_pose_detections=self.max_pose_detections,
                min_pose_score=self.min_pose_score)

            keypoint_coords *= output_scale
            keypoint_with_scores = np.ndarray([len(keypoint_coords), self.POINTS_NUMBER, 3])
            for i in range(len(keypoint_coords)):
                for j in range(self.POINTS_NUMBER):
                    keypoint_with_scores[i][j][0] = keypoint_coords[i][j][1]
                    keypoint_with_scores[i][j][1] = keypoint_coords[i][j][0]
                    keypoint_with_scores[i][j][2] = keypoint_scores[i][j]

            return keypoint_with_scores

    def convert(self, points):
        """ Преобразование скелетной модели posenet (movenet) к универсальной модели """
        new_points = np.ndarray([UNIVERSAL_POINTS_NUMBER, 3])
        new_points[3] = get_middle_point([points[0], points[1], points[2], points[3], points[4]])
        new_points[16] = get_middle_point([points[5], points[6]])
        new_points[4] = points[6]
        new_points[4][0] = points[6][0] * self.SHOULDER_COEFFICIENT_X
        new_points[4][1] = points[6][1] * self.SHOULDER_COEFFICIENT_Y
        new_points[5] = points[8]
        new_points[6] = points[10]
        new_points[10] = points[12]
        new_points[10][0] = points[12][0] * self.HIP_COEFFICIENT
        new_points[11] = points[14]
        new_points[12] = points[16]
        new_points[7] = points[5]
        new_points[7][0] = points[5][0] / self.SHOULDER_COEFFICIENT_X
        new_points[7][1] = points[5][1] * self.SHOULDER_COEFFICIENT_Y
        new_points[8] = points[7]
        new_points[9] = points[9]
        new_points[13] = points[11]
        new_points[13][0] = points[11][0] / self.HIP_COEFFICIENT
        new_points[14] = points[13]
        new_points[15] = points[15]
        new_points[0] = get_middle_point([points[11], points[12]])
        new_points[1] = get_middle_point([new_points[16], new_points[0]])
        new_points[2] = get_middle_point([new_points[3], new_points[16]])
        return new_points


class AlphaPose(Library):
    ID = 2
    POINTS_NUMBER = 26
    HIP_COEFFICIENT = 1.01
    CONNECTIONS = [[0, 1], [1, 3], [0, 2], [2, 4]
        , [0, 17], [0, 18], [18, 19]
        , [5, 7], [7, 9], [6, 8], [8, 10]
        , [19, 11], [19, 12]
        , [12, 14], [14, 16], [11, 13], [13, 15]
        , [16, 25], [25, 23], [21, 23], [21, 25]
        , [15, 24], [24, 22], [20, 22], [20, 24]
                   ]

    def __init__(self, confidence=0.5, config_path='configs/halpe_26/resnet/256x192_res50_lr1e-3_1x.yaml',
                 params=Namespace(checkpoint='./pretrained_models/halpe26_fast_res50_256x192.pth',
                                  debug=False, detector='yolo', device=torch.device(type='cpu'), eval=False, flip=False,
                                  format=None, gpus=[-1], min_box_area=0, pose_flow=False, pose_track=False,
                                  profile=False, showbox=False, tracking=False, vis=False, vis_fast=False)
                 ):
        super().__init__(confidence=confidence)
        self.demo = alphapose_api.SingleImageAlphaPose(params, update_config(config_path))

    def get_skeleton_data(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pose = self.demo.process('', image)
        results = pose.get("result")
        keypoint_with_scores = np.ndarray([len(results), self.POINTS_NUMBER, 3])
        for i in range(len(results)):
            for j in range(self.POINTS_NUMBER):
                keypoint_with_scores[i][j][0] = results[i].get("keypoints")[j][0]
                keypoint_with_scores[i][j][1] = results[i].get("keypoints")[j][1]
                keypoint_with_scores[i][j][2] = results[i].get("kp_score")[j][0]

        return keypoint_with_scores

    def convert(self, points):
        """ Преобразование скелетной модели alphapose к универсальной модели """
        new_points = np.ndarray([UNIVERSAL_POINTS_NUMBER, 3])
        new_points[3] = get_middle_point([points[0], points[1], points[2], points[3], points[4]])
        new_points[4] = points[6]
        new_points[5] = points[8]
        new_points[6] = points[10]
        new_points[10] = points[12]
        new_points[10][0] = points[12][0] * self.HIP_COEFFICIENT
        new_points[11] = points[14]
        new_points[12] = points[16]
        new_points[7] = points[5]
        new_points[8] = points[7]
        new_points[9] = points[9]
        new_points[13] = points[11]
        new_points[13][0] = points[11][0] / self.HIP_COEFFICIENT
        new_points[14] = points[13]
        new_points[15] = points[15]
        new_points[0] = points[19]
        new_points[1] = get_middle_point([points[18], points[19]])
        new_points[2] = points[18]
        tmp = get_middle_point([points[5], points[6]])
        new_points[16] = get_middle_point([tmp, points[18]])
        return new_points


class OpenPose(Library):
    ID = 3

    def __init__(self, confidence=0.3, params=dict([("model_folder","models/")])):
        super().__init__(confidence=confidence)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        try:
            # Change these variables to point to the correct folder (Release/x64 etc.)
            sys.path.append(dir_path + '/bin/python/openpose/Release')
            os.environ['PATH'] = os.environ['PATH'] + ';' + dir_path + '/x64/Release;' + dir_path + '/bin;'
            import pyopenpose as op
            self.op = op
        except ImportError as e:
            print(
                'Error: OpenPose library could not be found. Did you enable `BUILD_PYTHON` in CMake and have this Python script in the right folder?')
            raise e

        self.opWrapper = self.op.WrapperPython()
        self.opWrapper.configure(params)
        self.opWrapper.start()

    def get_skeleton_data(self, image):
        # import pyopenpose as op
        datum = self.op.Datum()
        datum.cvInputData = image
        self.opWrapper.emplaceAndPop(self.op.VectorDatum([datum]))
        return datum.poseKeypoints

    def convert(self, points):
        """ Преобразование скелетной модели openpose к универсальной модели """
        new_points = np.ndarray([UNIVERSAL_POINTS_NUMBER, 3])
        new_points[3] = get_middle_point([points[0], points[15], points[16], points[17], points[18]])
        new_points[4] = points[2]
        new_points[5] = points[3]
        new_points[6] = points[4]
        new_points[10] = points[9]
        new_points[11] = points[10]
        new_points[12] = points[11]
        new_points[7] = points[5]
        new_points[8] = points[6]
        new_points[9] = points[7]
        new_points[13] = points[12]
        new_points[14] = points[13]
        new_points[15] = points[14]
        new_points[0] = points[8]
        new_points[16] = points[1]
        new_points[1] = get_middle_point([points[1], points[8]])
        new_points[2] = get_middle_point([points[0], points[1]])
        return new_points


def convert_kinect_v1(points):
    new_points = np.ndarray([UNIVERSAL_POINTS_NUMBER, 3])
    new_points[0] = points[0]
    new_points[1] = points[1]
    new_points[2] = get_middle_point([points[2], points[3]])
    new_points[3] = points[3]
    new_points[4] = points[4]
    new_points[5] = points[5]
    new_points[6] = points[6]
    new_points[7] = points[8]
    new_points[8] = points[9]
    new_points[9] = points[10]
    new_points[10] = points[12]
    new_points[11] = points[13]
    new_points[12] = points[14]
    new_points[13] = points[16]
    new_points[14] = points[17]
    new_points[15] = points[18]
    new_points[16] = points[2]
    return new_points


# def get_kinect_origin_points_1(kinect_origin_file):
#     kinect_origin_file.readline()
#     points = np.ndarray([UNIVERSAL_POINTS_NUMBER, 3])
#     for j in range(UNIVERSAL_POINTS_NUMBER):
#         tmp = kinect_origin_file.readline().split(' ')
#         points[j][0] = round(float(tmp[4]))
#         points[j][1] = round(float(tmp[5]))
#         points[j][2] = round(float(tmp[3])) * LIB.confidence * 1.1
#     return points

def get_kinect_origin_points_2(kinect_origin_file):
    skeleton_kinect = []
    key_points = [0, 1, 2, 3, 4, 5, 6, 8, 9, 10,
                  12, 13, 14, 16, 17, 18, 20, ]
    M = np.asarray([
        [1030, 0, 980],
        [0, -1100, 530],
        [0, 0, 1],
    ])
    video = np.loadtxt(kinect_origin_file, delimiter=' ', )
    for i, line in enumerate(video):
        skeleton_kinect.append([])
        # skeleton_kinect[f'frame {i}'] = []
        if np.sum(np.abs(line[:75])):
            skeleton_kinect[i] = line[:75].reshape(25, 3)[np.ix_(key_points)]
    for i in range(len(skeleton_kinect)):
        skeleton_kinect[i] = np.asarray([np.dot(M, s.T) for s in skeleton_kinect[i]])
        skeleton_kinect[i][:, :] /= skeleton_kinect[i][:, -1].reshape(-1, 1)
    return skeleton_kinect


def get_middle_point(points):
    cnt = 0
    x = 0
    y = 0
    conf = 0

    for point in points:
        if point[0] != 0 and point[1] != 0:
            cnt += 1
            x += point[0]
            y += point[1]
            conf += point[2]

    x = x / cnt
    y = y / cnt
    conf = conf / cnt
    middle = np.array([x, y, conf])
    return middle


class Main:
    kinect_connections = [[0, 1], [1, 16], [2, 16], [2, 3], [4, 16], [4, 5], [5, 6], [7, 16], [7, 8],
                          [8, 9], [0, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15]]

    def __init__(self, library, source, start_frame=0, end_frame=-1
                 , write_video=False
                 , kinect_origin_points=None
                 , write_points_to_file=False, file_name='FileSkeleton.txt'
                 , write_points_to_db=False, db_file='test'
                 , convert_to_universal=True
                 , show_frame=False, show_kinect_origin_skeleton=False
                 , skeleton_thickness=1, skeleton_points_radios=3):
        self.LIB = library
        self.capture = cv2.VideoCapture(source)
        self.current_frame = 0

        if start_frame > 0:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            self.current_frame = start_frame

        self.end_frame = end_frame

        self.write_video = write_video
        if write_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.outfile = cv2.VideoWriter(
                f'{(source.isdigit() if f"camera {source}" else Path(source).stem)}'
                f'_{type(self.LIB).__name__}_{datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")}.mp4'
                , fourcc, self.capture.get(cv2.CAP_PROP_FPS)
                , (round(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
                   , round(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))))

        self.WRITE_POINTS_TO_FILE = write_points_to_file
        if write_points_to_file:
            self.points_file = open(file_name, 'w')

        self.WRITE_POINTS_TO_DB = write_points_to_db
        if write_points_to_db:
            self.db_connection = sqlite3.connect(db_file)
            curs = self.db_connection.cursor()
            curs.execute(f'insert into files (file_name, library_id, confidence, experiment_date)'
                         f'select \'{(source.isdigit() if f"camera {source}" else Path(source).stem)}\', {self.LIB.ID}, {self.LIB.confidence}, now();')
            self.file_id = curs.lastrowid
            self.db_connection.commit()

        self.CONVERT_TO_UNIVERSAL_SKELETON = convert_to_universal
        self.DRAW_KINECT_ORIGIN_SKELETON = show_kinect_origin_skeleton
        self.kinect_origin_points = kinect_origin_points
        self.SHOW_FRAME = show_frame
        self.SKELETON_THICKNESS = skeleton_thickness
        self.SKELETON_POINTS_RADIOS = skeleton_points_radios

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.capture.release()
        if self.write_video:
            self.outfile.release()
        if self.WRITE_POINTS_TO_FILE:
            self.points_file.close()
        if self.WRITE_POINTS_TO_DB:
            self.db_connection.close()
        cv2.destroyAllWindows()

    def draw_skeleton(self, im, points, connections, skeleton_color):
        for point in points:
            if self.WRITE_POINTS_TO_FILE:
                self.points_file.write(f"{point[0]} {point[1]} {point[2]}\n")
            if point[2] > self.LIB.confidence:
                im = cv2.circle(im,
                                (point[0].astype(np.int64),
                                 point[1].astype(np.int64)),
                                radius=self.SKELETON_POINTS_RADIOS, color=skeleton_color, thickness=-1)
        for connect in connections:
            if points[connect[0]][2] > self.LIB.confidence and points[connect[1]][2] > self.LIB.confidence:
                cv2.line(im,
                         (points[connect[0]][0].astype(np.int64), points[connect[0]][1].astype(np.int64)),
                         (points[connect[1]][0].astype(np.int64), points[connect[1]][1].astype(np.int64)),
                         color=skeleton_color, thickness=self.SKELETON_THICKNESS)

    def write_points_to_db(self, points):
        cursor = self.db_connection.cursor()
        for j in range(UNIVERSAL_POINTS_NUMBER):
            cursor.execute(f'insert into point_difference (file_id, point_type, frame_number'
                           f', x_kinect, y_kinect, confidence_kinect'
                           f', x_converted, y_converted, confidence_converted)'
                           f'select {self.file_id}, {j}, {self.current_frame}'
                           f', {self.kinect_origin_points is not None if self.kinect_origin_points[self.current_frame][j][0] else "null"}'
                           f', {self.kinect_origin_points is not None if self.kinect_origin_points[self.current_frame][j][1] else "null"}'
                           f', {self.kinect_origin_points is not None if self.kinect_origin_points[self.current_frame][j][2] else "null"}'
                           f', {points[j][0]}, {points[j][1]}, {points[j][2]}')
        self.db_connection.commit()

    def get_average_skeleton_confidence(self, points):
        conf = 0
        for point in points:
            conf += point[2]
        return conf / len(points)

    def analyse_frame(self, frame):
        skeletons = self.LIB.get_skeleton_data(frame)
        frame = frame.astype(np.uint8)

        new_skeletons = np.ndarray([len(skeletons), UNIVERSAL_POINTS_NUMBER, 3])
        for j in range(len(skeletons)):
            if self.WRITE_POINTS_TO_FILE:
                self.points_file.write(f"Frame: {self.current_frame}\n")

            if self.get_average_skeleton_confidence(skeletons[j]) > self.LIB.confidence:
                if self.CONVERT_TO_UNIVERSAL_SKELETON:
                    new_skeletons[j] = self.LIB.convert(skeletons[j])
                    self.draw_skeleton(frame, new_skeletons[j], self.kinect_connections, (0, 0, 255))
                else:
                    self.draw_skeleton(frame, skeletons[j], self.LIB.CONNECTIONS, (0, 255, 0))

            if self.DRAW_KINECT_ORIGIN_SKELETON:
                self.draw_skeleton(frame, self.kinect_origin_points[self.current_frame], self.kinect_connections,
                                   (0, 255, 0))

            if self.WRITE_POINTS_TO_DB and self.CONVERT_TO_UNIVERSAL_SKELETON:
                self.write_points_to_db(new_skeletons[j])

        cv2.putText(frame, f'frame: {self.current_frame}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        if self.SHOW_FRAME:
            cv2.imshow('frame', frame)

        if self.write_video:
            self.outfile.write(frame)

        return new_skeletons

    def get_next_frame(self):
        if (self.capture.isOpened() and
                (self.capture.get(cv2.CAP_PROP_POS_FRAMES) <= self.end_frame or self.end_frame == -1)):
            ret, frame = self.capture.read()
            if ret:
                points = self.analyse_frame(frame)
                self.current_frame += 1
                return points
            else:
                return None


def main():
    filename = '0002-M.avi'
    # kinect_origin_file = open('FileSkeleton.txt', 'r')
    kinect_origin_file = open('0002-M.txt', 'r')
    filename = 'person_stream.mp4'  # файл, по которому строим скелетные модели

    kinect_points = get_kinect_origin_points_2(kinect_origin_file)
    lib = AlphaPose()
    with Main(lib, '0002-M.avi', start_frame=3000, end_frame=3100, show_frame=True) as main3:
        while True:
            ret = main3.get_next_frame()
            if ret is None:
                break
            print(ret)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    main()
