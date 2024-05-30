import cv2
import skeleton_converter.skeleton_converter as sc


def main():
    # filename = '0002-M.avi'
    # # kinect_origin_file = open('FileSkeleton.txt', 'r')
    # kinect_origin_file = open('0002-M.txt', 'r')
    # filename = 'person_stream.mp4'  # файл, по которому строим скелетные модели
    #
    # kinect_points = get_kinect_origin_points_2(kinect_origin_file)
    lib = sc.Movenet()
    with sc.converter(lib, '0002-M.avi', start_frame=3000, end_frame=3100, show_frame=True) as main3:
    # with Main(lib, 0, show_frame=True) as main3:
        while True:
            frame, points = main3.get_next_frame()
            if points is None:
                break
            # print(points)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == "__main__":
    main()