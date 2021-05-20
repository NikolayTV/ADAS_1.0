import cv2
from openvino.inference_engine import IECore
import numpy as np


class OpenVino_face_landmarks_estimator(object):

    def __init__(self, model_path='models/facial-landmarks-35-adas-0002/FP32/facial-landmarks-35-adas-0002'):
        # Prep for face detection
        ie = IECore()

        net_hp = ie.read_network(model=model_path + '.xml', weights=model_path + '.bin')
        self.input_name_hp = next(iter(net_hp.inputs))  # Input blob name
        self.input_shape_hp = net_hp.inputs[self.input_name_hp].shape  # [1,3,60,60]
        self.out_name_hp = next(iter(net_hp.outputs))  # Output blob name
        self.out_shape_hp = net_hp.outputs[self.out_name_hp].shape  # [1,70]
        self.exec_net_hp = ie.load_network(network=net_hp, device_name='CPU', num_requests=1)
        self._X = 0
        self._Y = 1
        self.eye_ratio = 0.7
        del net_hp

    def predict(self, img, output):
        landmarks = []
        # For each face in json crop image and run head pose model
        for num, i in enumerate(output['faces']):
            xmin, ymin, xmax, ymax = i['bbox'].values()
            face = img[ymin:ymax, xmin:xmax]
            face_W, face_H = face.shape[:2]

            face_processed = cv2.resize(face, (self.input_shape_hp[3], self.input_shape_hp[2]))
            face_processed = face_processed.transpose((2, 0, 1))
            face_processed = face_processed.reshape(self.input_shape_hp)

            # Inference
            res_lm = self.exec_net_hp.infer(inputs={self.input_name_hp: face_processed})  # Run landmark detection
            lm = res_lm[self.out_name_hp][0][:8].reshape(4, 2)  # lm[0] (nose) lm[1] (eye)  lm[2] (eye) lm[3]

            #  [[left0x, left0y], [left1x, left1y], [right0x, right0y], [right1x, right1y] ]

            lm = [(int(coord[0] * face_H + xmin), int(coord[1] * face_W + ymin)) for coord in lm]
            left_eye1, left_eye2, right_eye1, right_eye2 = lm
            landmarks.append(lm)

            # REFRESH OUTPUT JSON
            output['faces'][num]['landmarks'] = {}
            output['faces'][num]['landmarks'] = {'left_eye1': left_eye1, 'left_eye2': left_eye2,
                                                 'right_eye1': right_eye1, 'right_eye2': right_eye2}

        return output, landmarks

    def createEyeBoundingBox(self, landmarks, scale=2):
        """
        Create a Eye bounding box using Two points that we got from headposition model

        Args:
        landmarks: 4 points - left_eye (2 points) and right_eye (2 points)
        """
        point1_ly = landmarks[0]
        point2_ly = landmarks[1]

        point1_ry = landmarks[2]
        point2_ry = landmarks[3]

        size_left_eye = cv2.norm(np.float32(point1_ly) - point2_ly)
        size_right_eye = cv2.norm(np.float32(point1_ry) - point2_ry)

        size = np.mean([size_left_eye, size_right_eye])
        width = int(scale * size)
        height = width

        # Left EYE
        midpoint_x_ly = (point1_ly[0] + point2_ly[0]) / 2
        midpoint_y_ly = (point1_ly[1] + point2_ly[1]) / 2

        startX_ly = midpoint_x_ly - (width / 2)
        startY_ly = midpoint_y_ly - (height / 2)

        endX_ly = midpoint_x_ly + (width / 2)
        endY_ly = midpoint_y_ly + (height / 2)

        ly_bbox = [int(startX_ly), int(startY_ly), int(endX_ly), int(endY_ly)]

        # Right EYE
        midpoint_x_ry = (point1_ry[0] + point2_ry[0]) / 2
        midpoint_y_ry = (point1_ry[1] + point2_ry[1]) / 2

        startX_ry = midpoint_x_ry - (width / 2)
        startY_ry = midpoint_y_ry - (height / 2)

        endX_ry = midpoint_x_ry + (width / 2)
        endY_ry = midpoint_y_ry + (height / 2)

        ry_bbox = [int(startX_ry), int(startY_ry), int(endX_ry), int(endY_ry)]

        return ly_bbox, ry_bbox
