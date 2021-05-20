import cv2
from openvino.inference_engine import IECore
import numpy as np


class OpenVino_head_pose_estimator(object):
    '''
    Find faces

    Example call:
        img = cv2.imread('tmp/1.jpg')

        face_detector = face_detection.OpenVino_face_detector()
        output = {}
        output = face_detector.predict(frame, output)

    Output Example:
        {'faces_num': 1, 'faces': [{'confidence': 0.998, 'bbox': {'xmin': 1241, 'ymin': 617, 'xmax': 1335, 'ymax': 707}}]}
    '''

    def __init__(self, model_path='models/head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001'):

        # Prep for face detection
        ie = IECore()

        net_hp = ie.read_network(model=model_path+'.xml', weights=model_path+'.bin')
        self.input_name_hp  = next(iter(net_hp.inputs))                              # Input blob name
        self.input_shape_hp = net_hp.inputs[self.input_name_hp].shape                     # [1,3,60,60]
        self.out_name_hp    = next(iter(net_hp.outputs))                             # Output blob name
        self.out_shape_hp   = net_hp.outputs[self.out_name_hp].shape                      # [1,70]
        self.exec_net_hp    = ie.load_network(network=net_hp, device_name='CPU', num_requests=1)
        del net_hp


    def predict(self, img, output):

        angles = []
        # For each face in json crop image and run head pose model
        for num, i in enumerate(output['faces']):
            xmin, ymin, xmax, ymax = i['bbox'].values()
            face = img[ymin:ymax, xmin:xmax]

            face_processed = cv2.resize(face, (self.input_shape_hp[3], self.input_shape_hp[2]))
            face_processed = face_processed.transpose((2, 0, 1))
            face_processed = face_processed.reshape(self.input_shape_hp)

            # Inference
            res_hp = self.exec_net_hp.infer(inputs={self.input_name_hp: face_processed})  # Run head pose estimation

            yaw = res_hp['angle_y_fc'][0][0]
            pitch = res_hp['angle_p_fc'][0][0]
            roll = res_hp['angle_r_fc'][0][0]

            # REFRESH OUTPUT JSON
            output['faces'][num]['head_pose'] = {}
            output['faces'][num]['head_pose'] = {'yaw': round(float(yaw), 2), 'pitch': round(float(pitch), 2), 'roll': round(float(roll), 2)}
            angles.append([yaw, pitch, roll])

        return output, angles
