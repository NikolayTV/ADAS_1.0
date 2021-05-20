import cv2

from core.cv import face_detection  # OpenVino_face_detector
from core.cv import head_pose_estimation  # OpenVino_head_pose_estimator
from core.cv import emotion_recognition
from core.cv import openvino_face_landmarks_estimation
from core.cv import openvino_gaze_estimation
from core.cv import eye_state_classification, mouth_state_classification
from core.cv.utils import draw_detection_roi, draw_head_pose
import time

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

while True:
    try:
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
            if eye_state:
                cv2.rectangle(drawed_frame, tuple(eye_bbox[:2]), tuple(eye_bbox[2:4]), (0, 250, 0), 2)
            else:
                cv2.rectangle(drawed_frame, tuple(eye_bbox[:2]), tuple(eye_bbox[2:4]), (0, 0, 0), 2)

            # DRAWING
            # Mouth
            if mouth_state:
                cv2.rectangle(drawed_frame, tuple(mouth_bbox[:2]), tuple(mouth_bbox[2:4]), (0, 0, 0), 2)
            else:
                cv2.rectangle(drawed_frame, tuple(mouth_bbox[:2]), tuple(mouth_bbox[2:4]), (0, 250, 0), 2)

            # Head Pose
            draw_head_pose(drawed_frame, face_bboxes[0], head_pose_angles[0])
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
            if eye_on_the_road:
                for gaze_line in gaze_lines[:2]:
                    cv2.arrowedLine(drawed_frame, gaze_line[0], gaze_line[1], (0, 255, 0), 3)
            else:
                for gaze_line in gaze_lines[:2]:
                    cv2.arrowedLine(drawed_frame, gaze_line[0], gaze_line[1], (20, 0, 150), 3)

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
                cv2.rectangle(drawed_frame, tuple(face_bboxes[0][:2]), tuple(face_bboxes[0][2:4]), (0, 220, 0), 2)
            else:
                cv2.rectangle(drawed_frame, tuple(face_bboxes[0][:2]), tuple(face_bboxes[0][2:4]), (20, 0, 150), 2)
    except:
        pass

    frame_end_time = time.time()
    print(1 / (frame_end_time - frame_start_time))

    cv2.imshow('drawed_frame', drawed_frame)
    key = cv2.waitKey(1)
    if key == ord('q'):
        videostream.release()
        cv2.destroyAllWindows()
        exit(1)
