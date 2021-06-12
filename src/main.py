import cv2

from core.cv import face_detection  # OpenVino_face_detector
from core.cv import head_pose_estimation  # OpenVino_head_pose_estimator
from core.cv import emotion_recognition
from core.cv import openvino_face_landmarks_estimation
from core.cv import openvino_gaze_estimation
from core.cv import eye_state_classification, mouth_state_classification
from core.cv.utils import draw_detection_roi, draw_head_pose
import time
import numpy as np

face_detector = face_detection.OpenVino_face_detector()
head_pose_estimator = head_pose_estimation.OpenVino_head_pose_estimator()
emotion_recognizer = emotion_recognition.OpenVino_emotion_recognition()
landmarks_estimator = openvino_face_landmarks_estimation.OpenVino_face_landmarks_estimator()
gaze_estimator = openvino_gaze_estimation.OpenVino_gaze_estimator()
eye_state_classifier = eye_state_classification.FastAI_OpenVino_eye_state_classifier()
mouth_state_classifier = mouth_state_classification.FastAI_OpenVino_mouth_state_classifier()

face_ID = 0

output = {}
videostream = cv2.VideoCapture(0)
# videostream = cv2.VideoCapture('/home/nikolay/workspace/training_extensions/pytorch_toolkit/open_closed_eye/data/videos/5_yo_mc')
# videostream = cv2.VideoCapture('data/1.webm')
awake = True
process_start_t = time.time()
time_sleeping = 0
gaze_lines1_1_x = []
gaze_lines1_1_y = []
gaze_lines1_2_x = []
gaze_lines1_2_y = []

gaze_lines2_1_x = []
gaze_lines2_1_y = []
gaze_lines2_2_x = []
gaze_lines2_2_y = []

while True:
    # try:
    frame_start_time = time.time()

    ret, frame = videostream.read()
    drawed_frame = frame.copy()

    # Inference
    output, face_bboxes = face_detector.predict(frame, output)
    output, head_pose_angles = head_pose_estimator.predict(frame, output)
    if len(face_bboxes) > 0:
        output, emotions = emotion_recognizer.predict(frame, output)
        output, landmarks = landmarks_estimator.predict(frame, output)
        output, gaze_vec_norm, gaze_lines = gaze_estimator.predict(frame, output)

        left_eye_bbox, right_eye_bbox = landmarks_estimator.createEyeBoundingBox(landmarks[0])
        start_t = time.time()
        eye_state, eye_bbox = eye_state_classifier.predict(frame, face_bboxes[0])  # [:,:,[2,1,0]]
        mouth_state, mouth_bbox = mouth_state_classifier.predict(frame, face_bboxes[0])  # [:,:,[2,1,0]]

        end_t = time.time()
        print('eye_state fps:', 1 / (end_t - start_t))

        # DRAWING
        # Eyes
        # bbox = eye_bbox
        # x_center = int((bbox[0] + bbox[2]) / 2)
        # y_center = int((bbox[1] + bbox[3]) / 2)
        # center = (x_center, y_center)

        # x_axes = int((bbox[2] - bbox[0]) / 2)
        # y_axes = int((bbox[1] - bbox[3]) / 2)
        # axes = (x_axes, y_axes)


        if eye_state:
            cv2.rectangle(drawed_frame, tuple(eye_bbox[:2]), tuple(eye_bbox[2:4]), (0, 250, 0), 1)
            # cv2.ellipse(drawed_frame, center, axes, 0.0, 0.0, 360.0, (0, 220, 0), 0)
        else:
            cv2.rectangle(drawed_frame, tuple(eye_bbox[:2]), tuple(eye_bbox[2:4]), (0, 0, 0), 1)
            # cv2.ellipse(drawed_frame, center, axes, 0.0, 0.0, 360.0, (0, 0, 0), 0)

        # DRAWING
        # Mouth
        bbox = mouth_bbox
        x_center = int((bbox[0] + bbox[2]) / 2)
        y_center = int((bbox[1] + bbox[3]) / 1.95)
        center = (x_center, y_center)

        x_axes = int((bbox[2] - bbox[0]) / 2)
        y_axes = int((bbox[1] - bbox[3]) / 2)
        axes = (x_axes, y_axes)
        if mouth_state:
            # cv2.rectangle(drawed_frame, tuple(mouth_bbox[:2]), tuple(mouth_bbox[2:4]), (0, 0, 0), 2)
            cv2.ellipse(drawed_frame, center, axes, 0.0, 0.0, 360.0, (0, 0, 0), 1)
        else:
            # cv2.rectangle(drawed_frame, tuple(mouth_bbox[:2]), tuple(mouth_bbox[2:4]), (0, 250, 0), 2)
            cv2.ellipse(drawed_frame, center, axes, 0.0, 0.0, 360.0, (0, 250, 0), 1)

        # Emotions
        cv2.putText(drawed_frame, emotions[0], (face_bboxes[0][0], face_bboxes[0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (150, 150, 150), 1)

        # Put text
        cv2.putText(drawed_frame, f'gaze_vector: {[round(i, 2) for i in gaze_vec_norm]}', (20, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(drawed_frame, f'yaw, pitch, roll: {[round(i, 2) for i in head_pose_angles[0]]}', (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        if eye_state:
            eye_state_str = 'Open'
        else:
            eye_state_str = 'Closed'
        cv2.putText(drawed_frame, f'eye state: {eye_state_str}', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0),
                    1)

        if mouth_state:
            mouth_state_str = 'Open'
        else:
            mouth_state_str = 'Closed'
        cv2.putText(drawed_frame, f'mouth state: {mouth_state_str}', (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (255, 255, 0), 1)

        eye_on_the_road = True
        x_gaze, y_gaze = gaze_vec_norm
        if 0.5 < x_gaze or x_gaze < -0.5:
            eye_on_the_road = False
        if 0.3 < y_gaze or y_gaze < -0.3:
            eye_on_the_road = False
        cv2.putText(drawed_frame, f'eye_on_the_road: {eye_on_the_road}', (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        print('gaze_lines', gaze_lines)
        print('gaze_lines[0]', gaze_lines[0])
        print('gaze_lines[0][0]', gaze_lines[0][0])
        print('gaze_lines[0][0][0]', gaze_lines[0][0][0])

        print('gaze_lines[1]', gaze_lines[1])
        print('gaze_lines[1][0]', gaze_lines[1][0])
        print('gaze_lines[1][0][0]', gaze_lines[1][0][0])
        # print('gaze_lines[1]', gaze_lines[1])
        # Draw Gaze lines
        gaze_lines1_1_x.append(gaze_lines[0][0][0])
        gaze_lines1_1_y.append(gaze_lines[0][0][1])
        gaze_lines1_2_x.append(gaze_lines[0][1][0])
        gaze_lines1_2_y.append(gaze_lines[0][1][1])

        gaze_lines2_1_x.append(gaze_lines[1][0][0])
        gaze_lines2_1_y.append(gaze_lines[1][0][1])
        gaze_lines2_2_x.append(gaze_lines[1][1][0])
        gaze_lines2_2_y.append(gaze_lines[1][1][1])

        gaze_lines1_1_x = gaze_lines1_1_x[-6:]
        gaze_lines1_1_y = gaze_lines1_1_y[-6:]
        gaze_lines1_2_x = gaze_lines1_2_x[-6:]
        gaze_lines1_2_y = gaze_lines1_2_y[-6:]

        gaze_lines2_1_x = gaze_lines2_1_x[-6:]
        gaze_lines2_1_y = gaze_lines2_1_y[-6:]
        gaze_lines2_2_x = gaze_lines2_2_x[-6:]
        gaze_lines2_2_y = gaze_lines2_2_y[-6:]

        gaze_line1_1_x = int(np.mean(gaze_lines1_1_x))
        gaze_line1_1_y = int(np.mean(gaze_lines1_1_y))
        gaze_line1_2_x = int(np.mean(gaze_lines1_2_x))
        gaze_line1_2_y = int(np.mean(gaze_lines1_2_y))

        gaze_line2_1_x = int(np.mean(gaze_lines2_1_x))
        gaze_line2_1_y = int(np.mean(gaze_lines2_1_y))
        gaze_line2_2_x = int(np.mean(gaze_lines2_2_x))
        gaze_line2_2_y = int(np.mean(gaze_lines2_2_y))
        print('gaze_lines', gaze_lines)
        if eye_on_the_road:
            cv2.arrowedLine(drawed_frame, (gaze_line1_1_x, gaze_line1_1_y), (gaze_line1_2_x, gaze_line1_2_y), (0, 255, 0), 2)
            cv2.arrowedLine(drawed_frame, (gaze_line2_1_x, gaze_line2_1_y), (gaze_line2_2_x, gaze_line2_2_y), (0, 255, 0), 2)
        else:
            cv2.arrowedLine(drawed_frame, (gaze_line1_1_x, gaze_line1_1_y), (gaze_line1_2_x, gaze_line1_2_y), (20, 0, 150), 2)
            cv2.arrowedLine(drawed_frame, (gaze_line2_1_x, gaze_line2_1_y), (gaze_line2_2_x, gaze_line2_2_y), (20, 0, 150), 2)


        # for gaze_line in gaze_lines[:2]:
        #     if eye_on_the_road:
        #         cv2.arrowedLine(drawed_frame, gaze_line[0], gaze_line[1], (0, 255, 0), 2)
        #     else:
        #         cv2.arrowedLine(drawed_frame, gaze_line[0], gaze_line[1], (20, 0, 150), 2)

        if not eye_state:
            time_sleeping += time.time() - start_t
        else:
            time_sleeping = 0

        if time_sleeping > 0.15:
            awake = False
        else:
            awake = True

        cv2.putText(drawed_frame, f'awake: {awake}', (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        cv2.putText(drawed_frame, f'time_sleeping: {time_sleeping*10}', (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)

        # Head BBOX
        if awake:
            # Head BBOX
            bbox = face_bboxes[0]
            x_center = int((bbox[0] + bbox[2])/2)
            y_center = int((bbox[1] + bbox[3])/2)
            center = (x_center, y_center)

            x_axes = int((bbox[2] - bbox[0]) / 1.5)
            y_axes = int((bbox[3] - bbox[1]) / 1.5)
            axes = (x_axes, y_axes)

            # cv2.rectangle(drawed_frame, tuple(face_bboxes[0][:2]), tuple(face_bboxes[0][2:4]), (0, 220, 0), 2)
            cv2.ellipse(drawed_frame, center, axes, 0.0, 0.0, 360.0, (0, 220, 0), 1)

        else:
            # cv2.rectangle(drawed_frame, tuple(face_bboxes[0][:2]), tuple(face_bboxes[0][2:4]), (20, 0, 150), 2)
            cv2.ellipse(drawed_frame, center, axes, 0.0, 0.0, 360.0, (20, 0, 150), 1)

        # Head Pose
        upper_point_ellipse = center[1] - y_axes
        draw_head_pose(drawed_frame, face_bboxes[0], head_pose_angles[0], upper_point_ellipse)

    # except:    # except:
    #     pass

    frame_end_time = time.time()
    print(1 / (frame_end_time - frame_start_time))

    cv2.imshow('drawed_frame', drawed_frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        videostream.release()
        cv2.destroyAllWindows()
        exit(1)
