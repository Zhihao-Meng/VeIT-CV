import cv2
import cv2 as cv
import mediapipe as mp
import numpy as np
import sys
import socket

from matplotlib import pyplot as plt

from utils import DLT, get_projection_matrix, write_keypoints_to_disk

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_pose = mp.solutions.pose

frame_shape = [720, 1280]

# add here if you need more keypoints
# https://developers.google.com/static/mediapipe/images/solutions/pose_landmarks_index.png
# pose_keypoints = [16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28]
pose_keypoints = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]


def run_mp(input_stream1, input_stream2, P0, P1):
    # input video stream
    cap0 = cv.VideoCapture(input_stream1)
    cap1 = cv.VideoCapture(input_stream2)
    caps = [cap0, cap1]

    # set camera resolution if using webcam to 1280x720. Any bigger will cause some lag for hand detection
    for cap in caps:
        cap.set(3, frame_shape[1])
        cap.set(4, frame_shape[0])

    # create body keypoints detector objects.
    pose0 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
    pose1 = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    # containers for detected keypoints for each camera. These are filled at each frame.
    # This will run you into memory issue if you run the program without stop
    kpts_cam0 = []
    kpts_cam1 = []
    kpts_3d = []

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')  # Create a 3D axis for plotting

    # Setup UDP socket
    UDP_IP = "192.168.1.5"
    UDP_PORT = 5007
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    while True:
        # read frames from stream
        ret0, frame0 = cap0.read()
        ret1, frame1 = cap1.read()

        if not ret0 or not ret1:
            print(ret0, ret1)
            break

        # change orientation in unity
        frame0 = cv2.flip(frame0, 0)
        frame1 = cv2.flip(frame1, 0)

        # crop to 720x720.
        # Note: camera calibration parameters are set to this resolution.If you change this, make sure to also change camera intrinsic parameters
        if frame0.shape[1] != 720:
            frame0 = frame0[:, frame_shape[1] // 2 - frame_shape[0] // 2:frame_shape[1] // 2 + frame_shape[0] // 2]
            frame1 = frame1[:, frame_shape[1] // 2 - frame_shape[0] // 2:frame_shape[1] // 2 + frame_shape[0] // 2]

        # the BGR image to RGB.
        frame0 = cv.cvtColor(frame0, cv.COLOR_BGR2RGB)
        frame1 = cv.cvtColor(frame1, cv.COLOR_BGR2RGB)

        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        frame0.flags.writeable = False
        frame1.flags.writeable = False
        results0 = pose0.process(frame0)
        results1 = pose1.process(frame1)

        # reverse changes
        frame0.flags.writeable = True
        frame1.flags.writeable = True
        frame0 = cv.cvtColor(frame0, cv.COLOR_RGB2BGR)
        frame1 = cv.cvtColor(frame1, cv.COLOR_RGB2BGR)

        # check for keypoints detection
        frame0_keypoints = []
        if results0.pose_landmarks:
            for i, landmark in enumerate(results0.pose_landmarks.landmark):
                if i not in pose_keypoints: continue  # only save keypoints that are indicated in pose_keypoints
                pxl_x = landmark.x * frame0.shape[1]
                pxl_y = landmark.y * frame0.shape[0]
                pxl_x = int(round(pxl_x))
                pxl_y = int(round(pxl_y))
                cv.circle(frame0, (pxl_x, pxl_y), 8, (0, 0, 255), -1)  # add keypoint detection points into figure
                kpts = [pxl_x, pxl_y]
                frame0_keypoints.append(kpts)
        else:
            # if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame0_keypoints = [[-1, -1]] * len(pose_keypoints)

        # this will keep keypoints of this frame in memory
        kpts_cam0.append(frame0_keypoints)

        frame1_keypoints = []
        if results1.pose_landmarks:
            for i, landmark in enumerate(results1.pose_landmarks.landmark):
                if i not in pose_keypoints: continue
                pxl_x = landmark.x * frame1.shape[1]
                pxl_y = landmark.y * frame1.shape[0]
                pxl_x = int(round(pxl_x))
                pxl_y = int(round(pxl_y))
                cv.circle(frame1, (pxl_x, pxl_y), 8, (0, 0, 255), -1)
                kpts = [pxl_x, pxl_y]
                frame1_keypoints.append(kpts)

        else:
            # if no keypoints are found, simply fill the frame data with [-1,-1] for each kpt
            frame1_keypoints = [[-1, -1]] * len(pose_keypoints)

        # update keypoints container
        kpts_cam1.append(frame1_keypoints)

        # calculate 3d position
        frame_p3ds = []
        for uv1, uv2 in zip(frame0_keypoints, frame1_keypoints):
            if uv1[0] == -1 or uv2[0] == -1:
                _p3d = [-1, -1, -1]
            else:
                _p3d = DLT([uv1, uv2], [P0, P1])  # calculate 3d position of keypoint
                _p3d = _p3d.tolist()
                decimal_places = 3
                _p3d = [round(num, decimal_places) for num in _p3d]
            frame_p3ds.append(_p3d)

        '''
        This contains the 3d position of each keypoint in current frame.
        For real time application, this is what you want.
        '''
        # frame_p3ds = np.array(frame_p3ds).reshape((12, 3))

        # prepare clean data to send to unity
        clean_frame_p3ds = ""
        for char in str(frame_p3ds):
            if char not in "[ ]":
                clean_frame_p3ds += char
        print(clean_frame_p3ds)

        # send frame_p3ds over UDP
        # sock.sendto(np.array2string(frame_p3ds).encode(), (UDP_IP, UDP_PORT))
        sock.sendto(str.encode(clean_frame_p3ds), (UDP_IP, UDP_PORT))

        # uncomment these if you want to see the full keypoints detections
        mp_drawing.draw_landmarks(frame0, results0.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        mp_drawing.draw_landmarks(frame1, results1.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

        # TODO: Run display here
        # visualize_3d(frame_p3ds, ax)
        # print(frame_p3ds)
        # cv.imshow('cam0', frame0)
        cv.imshow('cam1', frame1)

        k = cv.waitKey(1)
        if k & 0xFF == 27:
            print("breaking")
            break  # 27 is ESC key.

    cv.destroyAllWindows()
    for cap in caps:
        cap.release()


# pose_keypoints = np.array([16, 14, 12, 11, 13, 15, 24, 23, 25, 26, 27, 28])
pose_keypoints = np.array([11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28])


# def visualize_3d(kpts3d, ax):
#     """Now visualize in 3D"""
#     ax.clear()  # Clear the previous plot
#     # torso = [[0, 1], [1, 7], [7, 6], [6, 0]]
#     # armr = [[1, 3], [3, 5]]
#     # arml = [[0, 2], [2, 4]]
#     # legr = [[6, 8], [8, 10]]
#     # legl = [[7, 9], [9, 11]]
#     body = [torso, arml, armr, legr, legl]
#     colors = ['red', 'blue', 'green', 'black', 'orange']

#     for bodypart, part_color in zip(body, colors):
#         for _c in bodypart:
#             ax.plot(xs=[kpts3d[_c[0], 0], kpts3d[_c[1], 0]], ys=[kpts3d[_c[0], 1], kpts3d[_c[1], 1]],
#                     zs=[kpts3d[_c[0], 2], kpts3d[_c[1], 2]], linewidth=4, c=part_color)

#     ax.set_xticks([])
#     ax.set_yticks([])
#     ax.set_zticks([])

#     # ax.set_xlim3d(-10, 10)
#     ax.set_xlabel('x')
#     # ax.set_ylim3d(-10, 10)
#     ax.set_ylabel('y')
#     # ax.set_zlim3d(-10, 10)
#     ax.set_zlabel('z')
#     plt.pause(0.1)


if __name__ == '__main__':

    # this will load the sample videos if no camera ID is given
    input_stream1 = 1
    input_stream2 = 2

    # put camera id as command line arguements
    if len(sys.argv) == 3:
        input_stream1 = int(sys.argv[1])
        input_stream2 = int(sys.argv[2])

    # get projection matrices
    P0 = get_projection_matrix(0)
    P1 = get_projection_matrix(1)

    run_mp(input_stream1, input_stream2, P0, P1)
