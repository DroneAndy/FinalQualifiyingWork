import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sqlite3
from pathlib import Path
import datetime

model = hub.load(
    "https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/multipose-lightning/versions/1")
movenet = model.signatures['serving_default']

POSENET_POINTS_NUMBER = 17
UNIVERSAL_POINTS_NUMBER = 17
SKELETON_NUMBER = 6
CONFIDENCE = 0.3  # 0.4 это коэффициент уверенности в точке

CONVERT_TO_UNIVERSAL_SKELETON = True
DRAW_KINECT_ORIGIN_SKELETON = True
WRITE_POINTS_TO_DB = False
DATABASE_FILE = 'test_db'

WRITE_POINTS_TO_FILE = False
POINTS_FILENAME = 'test.txt'

SKELETON_THICKNESS = 2

HIP_COEFFICIENT = 1.01

kinect_connections = [[0, 1], [1, 16], [2, 16], [2, 3], [4, 16], [4, 5], [5, 6], [7, 16], [7, 8],
                      [8, 9], [0, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15]]

posenet_connections = [[0, 1], [1, 3], [0, 2], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                       [6, 12], [12, 14], [14, 16], [5, 11], [11, 13], [13, 15], [11, 12]]


def get_posenet_skeletal_data(image):
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
        key_points = np.ndarray([17, 3])
        for index in range(17):
            key_points[index][0] = round(image_width *
                                         key_points_with_score[(index * 3) + 1])
            key_points[index][1] = round(image_height *
                                         key_points_with_score[(index * 3) + 0])
            key_points[index][2] = key_points_with_score[(index * 3) + 2]
        key_points_list.append(key_points)
    return key_points_list


def draw_skeleton(im, points, connections, skeleton_color):
    for j in range(POSENET_POINTS_NUMBER):
        if WRITE_POINTS_TO_FILE:
            with open(POINTS_FILENAME, 'a') as f:
                f.write(f"{points[j][0]} {points[j][1]} {points[j][2]}\n")
        if points[j][2] > CONFIDENCE:
            im = cv2.circle(im,
                            (points[j][0].astype(np.int64),
                             points[j][1].astype(np.int64)),
                            radius=5, color=skeleton_color, thickness=-1)
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


# Преобразование скелетной модели posenet (movenet) к универсальной модели
def convert_posenet(points):
    new_points = np.ndarray([UNIVERSAL_POINTS_NUMBER, 3])
    new_points[3] = points[0]
    new_points[16] = get_middle_point(points[5], points[6])
    new_points[4] = points[6]
    new_points[5] = points[8]
    new_points[6] = points[10]
    new_points[10] = points[12]
    new_points[10][0] = points[12][0] * HIP_COEFFICIENT
    new_points[11] = points[14]
    new_points[12] = points[16]
    new_points[7] = points[5]
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


def get_kinect_origin_points():
    kinect_origin_file.readline()
    points = np.ndarray([UNIVERSAL_POINTS_NUMBER, 3])
    for j in range(UNIVERSAL_POINTS_NUMBER):
        tmp = kinect_origin_file.readline().split(' ')
        points[j][0] = round(float(tmp[4]))
        points[j][1] = round(float(tmp[5]))
        points[j][2] = round(float(tmp[3])) * CONFIDENCE * 1.1
    return points


def write_points_to_db(connect, points, kinect_points):
    cursor = connect.cursor()
    for j in range(UNIVERSAL_POINTS_NUMBER):
        cursor.execute(f'insert into point_difference (file_id, point_type, x1, y1, confidence1, x2, y2, confidence2)'
                       f'select {file_id}, {j}'
                       f', {points[j][0]}, {points[j][1]}, {points[j][2]}'
                       f', {kinect_points[j][0]}, {kinect_points[j][1]}, {kinect_points[j][2]}')
    connect.commit()


def get_average_skeleton_confidence(points):
    conf = 0
    for point in points:
        conf += point[2]
    return conf / len(points)


if __name__ == "__main__":
    filename = 'person_stream.mp4'  # файл, по которому строим скелетные модели
    cap = cv2.VideoCapture(filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(f'{Path(filename).stem}_{datetime.datetime.now().strftime("%d_%m_%Y_%H_%M")}.mp4'
                          , fourcc, cap.get(cv2.CAP_PROP_FPS)
                          , (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    if WRITE_POINTS_TO_FILE:
        open(POINTS_FILENAME, 'w').close()
    i = 0
    kinect_origin_file = open('FileSkeleton.txt', 'r')

    db_connection = None
    if WRITE_POINTS_TO_DB:
        db_connection = sqlite3.connect(DATABASE_FILE)
        curs = db_connection.cursor()
        curs.execute(f'insert into files (file_name) values (\'{filename}\');')
        file_id = curs.lastrowid
        db_connection.commit()

    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            skeletons = get_posenet_skeletal_data(frame)
            frame = frame.astype(np.uint8)
            if DRAW_KINECT_ORIGIN_SKELETON:
                current_kinect_points = get_kinect_origin_points()

            for j in range(SKELETON_NUMBER):
                current_points = skeletons[j]
                if WRITE_POINTS_TO_FILE:
                    with open(POINTS_FILENAME, 'a') as f:
                        f.write(f"Frame: {i}\n")

                if get_average_skeleton_confidence(current_points) > CONFIDENCE:
                    if CONVERT_TO_UNIVERSAL_SKELETON:
                        current_points = convert_posenet(current_points)
                        draw_skeleton(frame, current_points, kinect_connections, (0, 0, 255))
                    else:
                        draw_skeleton(frame, current_points, posenet_connections, (0, 255, 0))
                        current_points = convert_posenet(current_points)
                        draw_skeleton(frame, current_points, kinect_connections, (0, 0, 255))

                if DRAW_KINECT_ORIGIN_SKELETON:
                    draw_skeleton(frame, current_kinect_points, kinect_connections, (0, 255, 0))

                    if WRITE_POINTS_TO_DB:
                        write_points_to_db(db_connection, current_points, current_kinect_points)

            cv2.imshow('frame', frame)
            i += 1
            out.write(frame)
        else:
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if WRITE_POINTS_TO_DB:
        db_connection.close()
    cap.release()
    out.release()
    cv2.destroyAllWindows()
