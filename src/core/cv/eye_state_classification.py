import cv2
from openvino.inference_engine import IECore
import numpy as np
import math


def get_eye_bbox_from_face_bbox(bbox):
    x_min,y_min,x_max,y_max = bbox

    height = y_max - y_min
    width = x_max - x_min
    eye_top = y_min + height*0.2
    eye_bot = y_min + height*0.6
    eye_left = x_min + width * 0.1
    eye_right = x_max - width * 0.1
    eye_bbox = [int(eye_left), int(eye_bot), int(eye_right), int(eye_top)]
    return eye_bbox


class FastAI_OpenVino_eye_state_classifier(object):

    def __init__(self,
                 model_path='models/eye_state/eye_state'):
        # Prep for face detection
        ie = IECore()

        net_hp = ie.read_network(model=model_path + '.xml', weights=model_path + '.bin')
        self.input_name_hp = next(iter(net_hp.inputs))  # Input blob name
        self.out_name_hp = next(iter(net_hp.outputs))  # Input blob name
        _, _, self.height, self.width = net_hp.inputs[self.input_name_hp].shape

        self.out_name_hp = next(iter(net_hp.outputs))  # Output blob name
        self.out_shape_hp = net_hp.outputs[self.out_name_hp].shape  # [1,70]
        self.exec_net_hp = ie.load_network(network=net_hp, device_name='CPU', num_requests=1)

        self.normalize_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        self.normalize_scale = np.array([0.229, 0.224, 0.225], dtype=np.float32)

        del net_hp

    def predict(self, img, bbox):
        eye_bbox = get_eye_bbox_from_face_bbox(bbox)

        eye_image = img[eye_bbox[3]:eye_bbox[1], eye_bbox[0]:eye_bbox[2]]
        eye_image = cv2.resize(eye_image, (self.width, self.height))

        eye_image = eye_image.astype(np.float32) / 255
        eye_image = cv2.cvtColor(eye_image, cv2.COLOR_BGR2RGB)

        # normalize and convert to Channel Width Height
        eye_image = (eye_image - self.normalize_mean) / self.normalize_scale
        eye_image = eye_image.transpose((2, 0, 1))
        eye_image = np.expand_dims(eye_image, axis=0)

        out = self.exec_net_hp.infer(inputs={self.input_name_hp: eye_image})
        out = out[self.out_name_hp]
        probs = np.squeeze(out)
        probs = np.exp(probs) / sum(np.exp(probs))
        is_open = probs[0] < probs[1]

        return is_open, eye_bbox


