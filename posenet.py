import cv2
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

model = hub.load("https://www.kaggle.com/models/google/movenet/frameworks/TensorFlow2/variations/multipose-lightning/versions/1")
movenet = model.signatures['serving_default']


def get_posenet_skeletal_data(im):
    image = tf.expand_dims(im, axis=0)
    image = tf.cast(tf.image.resize_with_pad(image, 256, 256), dtype=tf.int32)
    return movenet(image)


coef = 420  # Я не понял пока как этот коэффициент высчитывать, 280
# но он зависит от разрешения. Нужен, чтобы смещать по высоте точки

points_number = 17
skeleton_number = 6  # here количество скелетов
confidence = 0.4  # 0.4 это коэффициент уверенности в точке


def draw_posenet_skeletal_data(im, sks, num):
    ratio = im.shape[1] / im.shape[0]
    coef = im.shape[0] / 255 * 100
    print(coef)
    for i in range(skeleton_number):
        points = sks['output_0'][0][i].numpy()
        points = points[:51].reshape([points_number, 3])  # размерность массива на выходе 56,
        # первые 51 это координаты точек
        # print(points)
        with open('test.txt', 'a') as f:
            f.write(f"Frame: {num}\n")

        for j in range(points_number):
            with open('test.txt', 'a') as f:
                f.write(f"{points[j][0]} {points[j][1]} {points[j][2]}\n")

            if points[j][2] > confidence:
                joints = points[j, :2]
                im = cv2.circle(im,
                                (np.round(joints[1] * im.shape[1]).astype(np.int64),
                                 # np.round(joints[0] * im.shape[0]).astype(np.int64)),
                                 np.round(joints[0] * im.shape[0] * ratio - coef).astype(np.int64)),
                                radius=5, color=(255, 255, 255), thickness=-1)
    return im


cap = cv2.VideoCapture(0)  # 1 - номер устройства
# cap = cv2.VideoCapture("C:\\Users\\akova\\Documents\\posenet-python-master\\posenet-python-master\\person_stream.mp4")  # 1 - номер устройства
open('test.txt', 'w').close()
i = 0
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        skeletons = get_posenet_skeletal_data(frame)
        frame = frame.astype(np.uint8)
        frame = draw_posenet_skeletal_data(frame, skeletons, i)
        cv2.imshow('frame', frame)
        i += 1
    else:
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
