from flask_restful import reqparse, abort, Api, Resource
from flask import jsonify, make_response
from flask import Flask, request

import numpy as np
import cv2

from core.cv import face_detection   # OpenVino_face_detector
from core.cv import head_pose_estimation   # OpenVino_head_pose_estimator
from core.cv import emotion_recognition
from core.cv import openvino_face_landmarks_estimation
from core.cv import openvino_gaze_estimation

import base64
import time
import json


flask_app = Flask(__name__)
api = Api(app=flask_app)

class FaceAnalizer(Resource):
    def __init__(self):
        self.face_detector = face_detection.OpenVino_face_detector()
        self.head_pose_estimator = head_pose_estimation.OpenVino_head_pose_estimator()
        self.emotion_recognizer = emotion_recognition.OpenVino_emotion_recognition()
        self.landmarks_estimator = openvino_face_landmarks_estimation.OpenVino_face_landmarks_estimator()
        self.gaze_estimator = openvino_gaze_estimation.OpenVino_gaze_estimator()

    def post(self):
        try:

            frame_start_time = time.time()
            self.output = {}
            self.output['success'] = True

            face_ID = 0

            # usage()

            flip_flag = False

            imageString = base64.b64decode(request.form['image'])
            nparr = np.fromstring(imageString, np.uint8)
            # decode image
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

            if flip_flag == True:
                frame = cv2.flip(frame, 1)                                             # flip image


            self.output, face_bboxes = self.face_detector.predict(frame, self.output)
            print('face_bboxes', face_bboxes)
            self.output, head_pose_angles = self.head_pose_estimator.predict(frame, self.output)
            if len(face_bboxes) > 0:
                self.output, emotions = self.emotion_recognizer.predict(frame, self.output)
                self.output, landmarks = self.landmarks_estimator.predict(frame, self.output)
                self.output, gaze_vec_norm = self.gaze_estimator.predict(frame, self.output)

            print(self.output)
            #     # Visualization
            #     # Draw centroids of tracked persons
            #     for person_num in range(len(tracked_persons)):
            #         objectID = tracked_persons[person_num].person_ID
            #         centroid = tracked_persons[person_num].current_face_centroid
            #         text = "ID {}".format(objectID)
            #         cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            #                     (0, 255, 0), 2)
            #         cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            #
            #     # Draw faces
            #     for num, bbox in enumerate(face_bboxes):
            #         draw_detection_roi(frame, bbox)
            #         draw_head_pose(frame, bbox, head_pose_angles[num])
            #         # Mood
            #         cv2.putText(frame, emotions[num], (bbox[0], bbox[1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (150, 150, 150), 1)
            #         # Landmarks
            #         # for lm in landmarks[num]:
            #         #     cv2.circle(frame, (lm[0], lm[1]), 2, (255, 255, 0), -1)
            #
            #     if len(output['faces']) > 0:
            #         # Drawing gaze lines
            #         for gaze_line in gaze_lines:
            #             cv2.arrowedLine(frame, gaze_line[0], gaze_line[1], (100, 100, 0), 3)
            #             # cv2.line(frame, gaze_line[0], gaze_line[1], (100, 100, 200), 2)
            #
            #     # Display the resulting frame
            #     cv2.imshow('frame', frame)
            #     if cv2.waitKey(1) & 0xFF == ord('q'):
            #         break
            #
            #     update_fps(frame_start_time)

            return make_response(jsonify(self.output), 200)
        except Exception:
            self.output = {}
            self.output['success'] = False
            self.output['Exception'] = Exception
            return make_response(jsonify(self.output), 404)


api.add_resource(FaceAnalizer, '/', methods=['GET', 'POST'])
if __name__ == '__main__':
    flask_app.run(debug=False, host='0.0.0.0', port=7777)

