import os
import cv2
import scipy
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sqlite3
from pathlib import Path
import datetime
import enum
import posenet


class WorkTypes(enum.Enum):
    single_video = 0,
    images = 1,
    SYSU3DAction = 2


class Libraries(enum.Enum):
    move_net = 0,
    pose_net = 1


POSENET_POINTS_NUMBER = 17
UNIVERSAL_POINTS_NUMBER = 17
SKELETON_NUMBER = 6
CONFIDENCE = 0.3  # 0.4 это коэффициент уверенности в точке

# Уверенность для PoseNet
# CONFIDENCE = 0.05  # 0.15 - стандартное значение из примера

LIBRARY = Libraries.move_net

CONVERT_TO_UNIVERSAL_SKELETON = True
DRAW_KINECT_ORIGIN_SKELETON = False
WRITE_POINTS_TO_DB = False
DATABASE_FILE = 'test_db3'

WRITE_POINTS_TO_FILE = False
POINTS_FILENAME = 'test.txt'

SHOW_FRAME = True

SKELETON_THICKNESS = 1
SKELETON_POINTS_RADIOS = 3

HIP_COEFFICIENT = 1.01
SHOULDER_COEFFICIENT_Y = 1.05
SHOULDER_COEFFICIENT_X = 1.01

kinect_connections = [[0, 1], [1, 16], [2, 16], [2, 3], [4, 16], [4, 5], [5, 6], [7, 16], [7, 8],
                      [8, 9], [0, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15]]

posenet_connections = [[0, 1], [1, 3], [0, 2], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                       [6, 12], [12, 14], [14, 16], [5, 11], [11, 13], [13, 15], [11, 12]]

WORK_TYPE = WorkTypes.single_video

if LIBRARY == Libraries.move_net:
    model = hub.load(
        "https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/multipose-lightning/versions/1")
    movenet = model.signatures['serving_default']
else:
    if LIBRARY == Libraries.pose_net:
        sess = tf.compat.v1.Session()
        model_cfg, model_outputs = posenet.load_model(101, sess, '/posenet/_models')
        output_stride = model_cfg['output_stride']


def get_movenet_skeletal_data(image):

    image_width, image_height = image.shape[1], image.shape[0]

    input_image = cv2.resize(image, dsize=(256, 256))
    input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
    input_image = input_image.reshape(-1, 256, 256, 3)
    input_image = tf.cast(input_image, dtype=tf.int32)

    outputs = movenet(input_image)

    key_points_with_scores = outputs['output_0'].numpy()
    key_points_with_scores = np.squeeze(key_points_with_scores)

    key_points_list = []
    for key_points_with_score in key_points_with_scores:
        key_points = np.ndarray([POSENET_POINTS_NUMBER, 3])
        for index in range(POSENET_POINTS_NUMBER):
            key_points[index][0] = round(image_width *
                                         key_points_with_score[(index * 3) + 1])
            key_points[index][1] = round(image_height *
                                         key_points_with_score[(index * 3) + 0])
            key_points[index][2] = key_points_with_score[(index * 3) + 2]
        key_points_list.append(key_points)
    return key_points_list


def get_posenet_skeleton_data(image):
    # with tf.compat.v1.Session() as sess:
    #     model_cfg, model_outputs = posenet.load_model(101, sess, '/posenet/_models')
    #     output_stride = model_cfg['output_stride']

    input_image, draw_image, output_scale = posenet.read_img(
        image, output_stride=output_stride)
        # image, scale_factor=args.scale_factor, output_stride=output_stride)

    heatmaps_result, offsets_result, displacement_fwd_result, displacement_bwd_result = sess.run(
        model_outputs,
        feed_dict={'image:0': input_image}
    )

    pose_scores, keypoint_scores, keypoint_coords = posenet.decode_multiple_poses(
        heatmaps_result.squeeze(axis=0),
        offsets_result.squeeze(axis=0),
        displacement_fwd_result.squeeze(axis=0),
        displacement_bwd_result.squeeze(axis=0),
        output_stride=output_stride,
        max_pose_detections=10,
        min_pose_score=0.25)

    keypoint_coords *= output_scale
    keypoint_with_scores = np.ndarray([len(keypoint_coords), POSENET_POINTS_NUMBER, 3])
    for i in range(len(keypoint_coords)):
        for j in range(POSENET_POINTS_NUMBER):
            keypoint_with_scores[i][j][0] = keypoint_coords[i][j][1]
            keypoint_with_scores[i][j][1] = keypoint_coords[i][j][0]
            keypoint_with_scores[i][j][2] = keypoint_scores[i][j]

    return keypoint_with_scores


def draw_skeleton(im, points, connections, skeleton_color):
    for j in range(POSENET_POINTS_NUMBER):
        if WRITE_POINTS_TO_FILE:
            with open(POINTS_FILENAME, 'a') as f:
                f.write(f"{points[j][0]} {points[j][1]} {points[j][2]}\n")
        if points[j][2] > CONFIDENCE:
            im = cv2.circle(im,
                            (points[j][0].astype(np.int64),
                             points[j][1].astype(np.int64)),
                            radius=SKELETON_POINTS_RADIOS, color=skeleton_color, thickness=-1)
    for connect in connections:
        if points[connect[0]][2] > CONFIDENCE and points[connect[1]][2] > CONFIDENCE:
            cv2.line(im,
                     (points[connect[0]][0].astype(np.int64), points[connect[0]][1].astype(np.int64)),
                     (points[connect[1]][0].astype(np.int64), points[connect[1]][1].astype(np.int64)),
                     color=skeleton_color, thickness=SKELETON_THICKNESS)


# Получение точки посередине
def get_middle_point(a, b):
    x = (a[0] + b[0]) / 2
    y = (a[1] + b[1]) / 2
    conf = (a[2] + b[2]) / 2
    middle = np.array([x, y, conf])
    return middle


def convert_posenet(points):
    """ Преобразование скелетной модели posenet (movenet) к универсальной модели """
    new_points = np.ndarray([UNIVERSAL_POINTS_NUMBER, 3])
    new_points[3][0] = (points[0][0] + points[1][0] + points[2][0] + points[3][0] + points[4][0]) / 5
    new_points[3][1] = (points[0][1] + points[1][1] + points[2][1] + points[3][1] + points[4][1]) / 5
    new_points[3][2] = (points[0][2] + points[1][2] + points[2][2] + points[3][2] + points[4][2]) / 5
    new_points[16] = get_middle_point(points[5], points[6])
    new_points[4] = points[6]
    new_points[4][0] = points[6][0] * SHOULDER_COEFFICIENT_X
    new_points[4][1] = points[6][1] * SHOULDER_COEFFICIENT_Y
    new_points[5] = points[8]
    new_points[6] = points[10]
    new_points[10] = points[12]
    new_points[10][0] = points[12][0] * HIP_COEFFICIENT
    new_points[11] = points[14]
    new_points[12] = points[16]
    new_points[7] = points[5]
    new_points[7][0] = points[5][0] / SHOULDER_COEFFICIENT_X
    new_points[7][1] = points[5][1] * SHOULDER_COEFFICIENT_Y
    new_points[8] = points[7]
    new_points[9] = points[9]
    new_points[13] = points[11]
    new_points[13][0] = points[11][0] / HIP_COEFFICIENT
    new_points[14] = points[13]
    new_points[15] = points[15]
    new_points[0] = get_middle_point(points[11], points[12])
    new_points[1] = get_middle_point(new_points[16], new_points[0])
    new_points[2] = get_middle_point(new_points[3], new_points[16])
    return new_points


def convert_kinect_v1(points):
    new_points = np.ndarray([UNIVERSAL_POINTS_NUMBER, 3])
    new_points[0] = points[0]
    new_points[1] = points[1]
    new_points[2] = get_middle_point(points[2], points[3])
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


def get_kinect_origin_points(kinect_origin_file):
    kinect_origin_file.readline()
    points = np.ndarray([UNIVERSAL_POINTS_NUMBER, 3])
    for j in range(UNIVERSAL_POINTS_NUMBER):
        tmp = kinect_origin_file.readline().split(' ')
        points[j][0] = round(float(tmp[4]))
        points[j][1] = round(float(tmp[5]))
        points[j][2] = round(float(tmp[3])) * CONFIDENCE * 1.1
    return points


def write_points_to_db(connect, file_id, points, kinect_points, frame_number):
    cursor = connect.cursor()
    for j in range(UNIVERSAL_POINTS_NUMBER):
        cursor.execute(f'insert into point_difference (file_id, point_type, frame_number'
                       f', x_kinect, y_kinect, confidence_kinect'
                       f', x_converted, y_converted, confidence_converted)'
                       f'select {file_id}, {j}, {frame_number}'
                       f', {kinect_points[j][0]}, {kinect_points[j][1]}, {kinect_points[j][2]}'
                       f', {points[j][0]}, {points[j][1]}, {points[j][2]}')
    connect.commit()


def get_average_skeleton_confidence(points):
    conf = 0
    for point in points:
        conf += point[2]
    return conf / len(points)


def analyse_frame(frame, frame_number, out, origin_points, db_connection, file_id, start_time, start_frame = 0):
    skeletons = np.ndarray([0])
    match LIBRARY:
        case Libraries.move_net:
            skeletons = get_movenet_skeletal_data(frame)
        case Libraries.pose_net:
            skeletons = get_posenet_skeleton_data(frame)
    frame = frame.astype(np.uint8)

    for j in range(SKELETON_NUMBER):
        current_points = skeletons[j]
        if WRITE_POINTS_TO_FILE:
            with open(POINTS_FILENAME, 'a') as f:
                f.write(f"Frame: {frame_number}\n")

        if get_average_skeleton_confidence(current_points) > CONFIDENCE:
            if CONVERT_TO_UNIVERSAL_SKELETON:
                current_points = convert_posenet(current_points)
                draw_skeleton(frame, current_points, kinect_connections, (0, 0, 255))
            else:
                draw_skeleton(frame, current_points, posenet_connections, (0, 255, 0))
                current_points = convert_posenet(current_points)
                draw_skeleton(frame, current_points, kinect_connections, (0, 0, 255))

        if DRAW_KINECT_ORIGIN_SKELETON:
            draw_skeleton(frame, origin_points, kinect_connections, (0, 255, 0))

            if WRITE_POINTS_TO_DB:
                write_points_to_db(db_connection, file_id, current_points, origin_points, frame_number)

    cv2.putText(frame, f'frame: {frame_number}', (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    cv2.putText(frame, f'frame rate: {(frame_number - start_frame) / (datetime.datetime.now() - start_time).total_seconds()}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    if SHOW_FRAME:
        cv2.imshow('frame', frame)
    out.write(frame)


def analyse_images(db_connection, images_path, origin_points, out_filename, start_time):
    image_width = round(cv2.imread(os.path.join(images_path, os.listdir(images_path)[0])).shape[1])
    image_height = round(cv2.imread(os.path.join(images_path, os.listdir(images_path)[0])).shape[0])

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{out_filename}_{datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")}.mp4'
                          , fourcc, 24
                          , (image_width, image_height))
    file_id = -1
    if WRITE_POINTS_TO_DB:
        curs = db_connection.cursor()
        curs.execute(f'insert into files (file_name, library_id, confidence, experiment_date, width, height)'
                     f'select \'{out_filename}\', {LIBRARY.value}'
                     f', {CONFIDENCE}, datetime(\'now\',\'localtime\')'
                     f', {image_width}, {image_height};')
        file_id = curs.lastrowid
        db_connection.commit()

    i = 0
    for img in os.listdir(images_path):
        if os.path.isfile(os.path.join(images_path, img)):
            frame = cv2.imread(os.path.join(images_path, img))
            if frame is None:
                print(f'can not read {os.path.join(images_path, img)}')
                continue
            analyse_frame(frame, i, out, origin_points[i], db_connection, file_id, start_time)
            i += 1
    out.release()


def analyse_video(db_connection, filename, kinect_file, start_time, begin_frame=0, end_frame=-1):
    file_id = -1
    if WRITE_POINTS_TO_DB:
        curs = db_connection.cursor()
        curs.execute(f'insert into files (file_name, library_id, confidence, experiment_date)'
                     f'select \'{filename}\', {LIBRARY}, {CONFIDENCE}, now();')
        file_id = curs.lastrowid
        db_connection.commit()
    cap = cv2.VideoCapture(filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{Path(filename).stem}_{datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")}.mp4'
                          , fourcc, cap.get(cv2.CAP_PROP_FPS)
                          , (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))
    i = begin_frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, begin_frame)
    while cap.isOpened() and (cap.get(cv2.CAP_PROP_POS_FRAMES) <= end_frame or end_frame == -1):
        ret, frame = cap.read()
        if ret:
            current_kinect_points = np.ndarray([0, 0])
            if DRAW_KINECT_ORIGIN_SKELETON:
                current_kinect_points = get_kinect_origin_points(kinect_file)
            analyse_frame(frame, i, out, current_kinect_points, db_connection, file_id, start_time, start_frame=begin_frame)
            i += 1
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()


def analyse_SYSU3DAction_video(db_connection, path, actor, video):
    tmp_points = scipy.io.loadmat(
        os.path.join(path, actor, video, 'sklImage.mat'))[
        'S']
    points = np.ndarray([len(tmp_points[0][0]), len(tmp_points), 3])
    for i in range(len(points)):
        for j in range(len(points[0])):
            points[i][j][0] = tmp_points[j][0][i]
            points[i][j][1] = tmp_points[j][1][i]
            points[i][j][2] = 1.1 * CONFIDENCE

    origin_points = np.ndarray([len(points), UNIVERSAL_POINTS_NUMBER, 3])
    for i in range(len(points)):
        origin_points[i] = convert_kinect_v1(points[i])
    analyse_images(db_connection, os.path.join(path, actor, video, 'rgb'), origin_points, f'{actor}_{video}')


def analyse_SYSU3DAction(db_connection, path, analyse_all=True, activity_number=0):
    for actor in os.listdir(path):
        if analyse_all:
            for video in os.listdir(os.path.join(path, actor)):
                analyse_SYSU3DAction_video(db_connection, path, actor, video)
        else:
            video = f'video{activity_number}'
            analyse_SYSU3DAction_video(db_connection, path, actor, video)


def main():
    if WRITE_POINTS_TO_FILE:
        open(POINTS_FILENAME, 'w').close()
    kinect_origin_file = open('FileSkeleton.txt', 'r')

    db_connection = None
    if WRITE_POINTS_TO_DB:
        db_connection = sqlite3.connect(DATABASE_FILE)

    match WORK_TYPE:
        case WorkTypes.single_video:
            filename = 'person_stream.mp4'  # файл, по которому строим скелетные модели
            analyse_video(db_connection, filename, kinect_origin_file, datetime.datetime.now(), begin_frame=3000, end_frame=4000)
        case WorkTypes.SYSU3DAction:
            analyse_SYSU3DAction(db_connection, 'C:\\Users\\akova\\Documents\\SYSU3DAction\\3DvideoNorm', analyse_all=True)

    if WRITE_POINTS_TO_DB:
        db_connection.close()

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
