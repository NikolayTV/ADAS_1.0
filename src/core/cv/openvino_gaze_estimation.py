import cv2
from openvino.inference_engine import IECore
import numpy as np
import math


class OpenVino_gaze_estimator(object):

    def __init__(self, model_path='models/gaze-estimation-adas-0002/FP32/gaze-estimation-adas-0002'):

        # Prep for face detection
        ie = IECore()

        net_hp = ie.read_network(model=model_path + '.xml', weights=model_path + '.bin')
        self.input_name_hp = next(iter(net_hp.inputs))  # Input blob name
        self.input_shape_hp = net_hp.inputs[self.input_name_hp].shape  # [1,3,60,60]
        self.input_shape_gaze = [1, 3, 60, 60]
        self.out_name_hp = next(iter(net_hp.outputs))  # Output blob name
        self.out_shape_hp = net_hp.outputs[self.out_name_hp].shape  # [1,70]
        self.exec_net_hp = ie.load_network(network=net_hp, device_name='CPU', num_requests=1)
        self._X = 0
        self._Y = 1
        self.eye_ratio = 0.7
        del net_hp

    def predict(self, img, output):
        gaze_lines = []
        gaze_vec_norm = []

        # For each face in json crop image and run head pose model
        for num, i in enumerate(output['faces']):
            left_eye1, left_eye2, right_eye1, right_eye2 = i['landmarks'].values()
            yaw, pitch, roll = i['head_pose'].values()

            # eye size in the cropped face image
            eye_sizes = [abs(left_eye1[0] - left_eye2[0]),
                         abs(right_eye2[0] - right_eye1[0])]

            if eye_sizes[0] < 4 or eye_sizes[1] < 4:
                continue

            # eye center coordinate in the cropped face image
            eye_centers = [[int((left_eye1[0] + left_eye2[0]) / 2),
                            int((left_eye1[1] + left_eye2[1]) / 2)],

                           [int((right_eye2[0] + right_eye1[0]) / 2),
                            int((right_eye2[1] + right_eye1[1]) / 2)]]

            # print('eye_sizes', eye_sizes)

            eyes = []
            for i in range(2):
                # Crop eye images
                x1 = int(eye_centers[i][0] - eye_sizes[i] * self.eye_ratio)
                x2 = int(eye_centers[i][0] + eye_sizes[i] * self.eye_ratio)
                y1 = int(eye_centers[i][1] - eye_sizes[i] * self.eye_ratio)
                y2 = int(eye_centers[i][1] + eye_sizes[i] * self.eye_ratio)

                eye = cv2.resize(img[y1:y2, x1:x2].copy(),
                                 (self.input_shape_gaze[3], self.input_shape_gaze[2]))  # crop and resize
                eyes.append(eye)

                # rotate eyes around Z axis to keep them level
                rotMat = cv2.getRotationMatrix2D((int(self.input_shape_gaze[3] / 2), int(self.input_shape_gaze[2] / 2)),
                                                 roll, 1.0)
                eyes[i] = cv2.warpAffine(eyes[i], rotMat, (self.input_shape_gaze[3], self.input_shape_gaze[2]),
                                         flags=cv2.INTER_LINEAR)

                eyes[i] = eyes[i].transpose((2, 0, 1))
                eyes[i] = eyes[i].reshape(self.input_shape_gaze)

            # Inference
            res_gaze = self.exec_net_hp.infer(inputs={'left_eye_image': eyes[0],
                                                      'right_eye_image': eyes[1],
                                                      'head_pose_angles': [yaw, pitch, roll]})

            gaze_vec = res_gaze['gaze_vector'][0]  # result is in orthogonal coordinate system (x,y,z. not yaw,pitch,roll)and not normalized
            gaze_vec_norm = gaze_vec / np.linalg.norm(gaze_vec)  # normalize the gaze vector

            vcos = math.cos(math.radians(roll))
            vsin = math.sin(math.radians(roll))
            tmpx = gaze_vec_norm[0] * vcos + gaze_vec_norm[1] * vsin
            tmpy = -gaze_vec_norm[0] * vsin + gaze_vec_norm[1] * vcos
            gaze_vec_norm = [tmpx, tmpy]

            # REFRESH OUTPUT JSON
            output['faces'][num]['eye_gaze'] = {'x_coord': float(round(tmpx, 2)), 'y_coord': float(round(tmpy, 2))}

            # Prepare lines for visualization
            xmin, ymin, xmax, ymax = output['faces'][num]['bbox'].values()
            faceBoundingBoxWidth = xmax - xmin
            # Store gaze line coordinations
            for i in range(2):
                coord1 = (eye_centers[i][0], eye_centers[i][1])
                coord2 = (eye_centers[i][0] + int(gaze_vec_norm[0] * faceBoundingBoxWidth),
                          eye_centers[i][1] - int(gaze_vec_norm[1] * faceBoundingBoxWidth))
                gaze_lines.append([coord1, coord2, False])  # line(coord1, coord2); False=spark flag

        return output, gaze_vec_norm, gaze_lines

        # left_eye1, left_eye2
        #
        # leftEyeMidpoint_start = int(((left_eye_x + right_eye_x)) / 2)
        # leftEyeMidpoint_end = int(((left_eye_y + right_eye_y)) / 2)
        # rightEyeMidpoint_start = int((nose_tip_x + left_lip_corner_x) / 2)
        # rightEyeMidpoint_End = int((nose_tip_y + left_lip_corner_y) / 2)
        #
        # # Gaze out
        #
        # arrowLength = 0.4 * faceBoundingBoxWidth
        # gaze = gaze_vector[0]
        # gazeArrow_x = int((gaze_vec_norm[0]) * arrowLength)
        # gazeArrow_y = int(-(gaze_vec_norm[1]) * arrowLength)
        #
        # for i in range(2):
        #     coord1 = (eye_centers[i][0], eye_centers[i][1])
        #     coord2 = (eye_centers[i][0] + int(0.4 * faceBoundingBoxWidth),
        #               eye_centers[i][1] - int((gaze_vec_norm[1] + 0.) * 800))
        #     gaze_lines.append([coord1, coord2, False])  # line(coord1, coord2); False=spark flag
        #
        # cv2.arrowedLine(frame,
        #                 (eye_centers[i][0], eye_centers[i][1]),
        #                 ((eye_centers[i][0] + gazeArrow_x),
        #                  leftEyeMidpoint_end + (gazeArrow_y)),
        #                 (0, 255, 0), 3)
        #
        # cv2.arrowedLine(frame,
        #                 (rightEyeMidpoint_start, rightEyeMidpoint_End),
        #                 ((rightEyeMidpoint_start + gazeArrow_x),
        #                  rightEyeMidpoint_End + (gazeArrow_y)),
        #                 (0, 255, 0), 3)
