import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import sqlite3

model = hub.load(
    "https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/multipose-lightning/versions/1")
movenet = model.signatures['serving_default']


def get_posenet_skeletal_data(im):
    image = tf.expand_dims(im, axis=0)
    image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)
    return movenet(image)


# coef = 420 # 280 Я не понял пока как этот коэффициент высчитывать,
# но он зависит от разрешения. Нужен, чтобы смещать по высоте точки

points_number = 17
skeleton_number = 6  # here количество скелетов
confidence = 0.3  # 0.4 это коэффициент уверенности в точке

convert_to_universal_skeleton = True
draw_kinect_origin_skeleton = True
skeleton_thickness = 2

hip_coef = 1.01


# Функция для пересчета координат точек
def convert_points(im, points):
    ratio = im.shape[1] / im.shape[0]
    coef = im.shape[0] / 255 * 100
    for j in range(points_number):
        joints = points[j, :2]
        points[j][1] = np.round(joints[1] * im.shape[1]).astype(np.int64)
        points[j][0] = np.round(joints[0] * im.shape[0] * ratio - coef).astype(np.int64)
    return points


kinect_connections = [[0, 1], [1, 16], [2, 16], [2, 3], [4, 16], [4, 5], [5, 6], [7, 16], [7, 8],
                      [8, 9], [0, 10], [10, 11], [11, 12], [0, 13], [13, 14], [14, 15]]

posenet_connections = [[0, 1], [1, 3], [0, 2], [2, 4], [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],
                       [6, 12], [12, 14], [14, 16], [5, 11], [11, 13], [13, 15], [11, 12]]


def draw_skeleton(im, points, connections, skeleton_color):
    for j in range(points_number):
        with open('test.txt', 'a') as f:
            f.write(f"{points[j][0]} {points[j][1]} {points[j][2]}\n")
        if points[j][2] > confidence:
            im = cv2.circle(im,
                            (points[j][1].astype(np.int64),
                             points[j][0].astype(np.int64)),
                            radius=5, color=skeleton_color, thickness=-1)
    for connect in connections:
        if points[connect[0]][2] > confidence and points[connect[1]][2] > confidence:
            cv2.line(im,
                     (points[connect[0]][1].astype(np.int64), points[connect[0]][0].astype(np.int64)),
                     (points[connect[1]][1].astype(np.int64), points[connect[1]][0].astype(np.int64)),
                     color=skeleton_color, thickness=skeleton_thickness)


# Получение точки посередине
def get_middle_point(a, b):
    x = (a[0] + b[0]) / 2
    y = (a[1] + b[1]) / 2
    conf = (a[2] + b[2]) / 2
    middle = np.array([x, y, conf])
    return middle


# Преобразование скелетной модели posenet (movenet) к универсальной модели
def convert_posenet(points):
    new_points = np.ndarray([17, 3])

    new_points[3] = points[0]
    new_points[16] = get_middle_point(points[5], points[6])
    new_points[4] = points[6]
    new_points[5] = points[8]
    new_points[6] = points[10]
    new_points[10] = points[12]
    new_points[10][1] = points[12][1] * hip_coef
    new_points[11] = points[14]
    new_points[12] = points[16]
    new_points[7] = points[5]
    new_points[8] = points[7]
    new_points[9] = points[9]
    new_points[13] = points[11]
    new_points[13][1] = points[11][1] / hip_coef
    new_points[14] = points[13]
    new_points[15] = points[15]
    new_points[0] = get_middle_point(points[11], points[12])
    new_points[1] = get_middle_point(new_points[16], new_points[0])
    new_points[2] = get_middle_point(new_points[3], new_points[16])
    return new_points


def get_kinect_origin_points():
    kinect_origin_file.readline()
    points = np.ndarray([17, 3])
    for j in range(17):
        tmp = kinect_origin_file.readline().split(' ')
        points[j][0] = round(float(tmp[5]))
        points[j][1] = round(float(tmp[4]))
        points[j][2] = round(float(tmp[3])) * confidence * 1.1
    return points


def write_points_to_db(connect, file_id, points, kinect_points):
    cursor = connect.cursor()
    for j in range(17):
        cursor.execute(f'insert into point_difference (file_id, point_type, x1, y1, confidence1, x2, y2, confidence2)'
                       f'select {file_id}, {j}'
                       f', {points[j][1]}, {points[j][0]}, {points[j][2]}'
                       f', {kinect_points[j][1]}, {kinect_points[j][0]}, {kinect_points[j][2]}')
    connect.commit()


def draw_posenet_skeletal_data(im, sks, num):
    if draw_kinect_origin_skeleton:
        kinect_points = get_kinect_origin_points()
    
    for j in range(skeleton_number):
        points = sks['output_0'][0][j].numpy()
        points = points[:51].reshape([points_number, 3])  # размерность массива на выходе 56,
        # первые 51 это координаты точек
        with open('test.txt', 'a') as f:
            f.write(f"Frame: {num}\n")

        points = convert_points(im, points)
        if convert_to_universal_skeleton:
            points = convert_posenet(points)
            draw_skeleton(im, points, kinect_connections, (0, 0, 255))
        else:
            draw_skeleton(im, points, posenet_connections, (0, 255, 0))
            points = convert_posenet(points)
            draw_skeleton(im, points, kinect_connections, (0, 0, 255))

        if draw_kinect_origin_skeleton:
            draw_skeleton(im, kinect_points, kinect_connections, (0, 255, 0))
            write_points_to_db(db_connection, file_id1, points, kinect_points)

    return im


filename = 'person_stream.mp4'  # файл, по которому строим скелетные модели
database = 'test_db'
cap = cv2.VideoCapture(filename)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('output.mp4', fourcc, 24.0, (1920, 1080))
open('test.txt', 'w').close()
i = 0
kinect_origin_file = open('FileSkeleton.txt', 'r')
db_connection = sqlite3.connect(database)
curs = db_connection.cursor()
curs.execute(f'insert into files (file_name) values (\'{filename}\');')
file_id1 = curs.lastrowid
db_connection.commit()

while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        skeletons = get_posenet_skeletal_data(frame)
        frame = frame.astype(np.uint8)
        frame = draw_posenet_skeletal_data(frame, skeletons, i)
        cv2.imshow('frame', frame)
        i += 1
        out.write(frame)
    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

db_connection.close()
cap.release()
out.release()
cv2.destroyAllWindows()
