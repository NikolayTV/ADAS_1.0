import cv2
from openvino.inference_engine import IECore
import numpy as np
import math


def cut_mouth_from_bbox(bbox):
    x_min,y_min,x_max,y_max = bbox

    height = y_max - y_min
    width = x_max - x_min
    mouth_top = y_min + height*0.5
    mouth_bot = y_min + height*0.9
    mouth_left = x_min
    mouth_right = x_max
    mouth_bbox = [int(mouth_left), int(mouth_bot), int(mouth_right), int(mouth_top)]
    return mouth_bbox


class FastAI_OpenVino_mouth_state_classifier(object):

    def __init__(self,
                 model_path='models/mouth_state/mouth_state'):
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
        mouth_bbox = cut_mouth_from_bbox(bbox)

        mouth_image = img[mouth_bbox[3]:mouth_bbox[1], mouth_bbox[0]:mouth_bbox[2]]
        mouth_image = cv2.resize(mouth_image, (self.width, self.height))

        mouth_image = mouth_image.astype(np.float32) / 255
        mouth_image = cv2.cvtColor(mouth_image, cv2.COLOR_BGR2RGB)

        # normalize and convert to Channel Width Height
        mouth_image = (mouth_image - self.normalize_mean) / self.normalize_scale
        mouth_image = mouth_image.transpose((2, 0, 1))
        mouth_image = np.expand_dims(mouth_image, axis=0)

        out = self.exec_net_hp.infer(inputs={self.input_name_hp: mouth_image})
        out = out[self.out_name_hp]
        probs = np.squeeze(out)
        probs = np.exp(probs) / sum(np.exp(probs))
        is_open = probs[0] < probs[1]
        return is_open, mouth_bbox



# class FastAI_OpenVino_mouth_state_classifier(object):
#
#     def __init__(self,
#                  model_path= '/home/nikolay/workspace/training_extensions/pytorch_toolkit/open_closed_mouth/Notebooks/ir/model_fastai'):
#         # Prep for face detection
#         ie = IECore()
#
#         net_hp = ie.read_network(model=model_path + '.xml', weights=model_path + '.bin')
#         self.input_name_hp = next(iter(net_hp.inputs))  # Input blob name
#         self.out_name_hp = next(iter(net_hp.outputs))  # Input blob name
#         _, _, self.height, self.width = net_hp.inputs[self.input_name_hp].shape
#
#         self.out_name_hp = next(iter(net_hp.outputs))  # Output blob name
#         self.out_shape_hp = net_hp.outputs[self.out_name_hp].shape  # [1,70]
#         self.exec_net_hp = ie.load_network(network=net_hp, device_name='CPU', num_requests=1)
#
#         self.normalize_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
#         self.normalize_scale = np.array([0.229, 0.224, 0.225], dtype=np.float32)
#
#         del net_hp
#
#     def predict(self, img, left_mouth_bbox, right_mouth_bbox):
#         mouths_img = prepare_image(img, left_mouth_bbox, right_mouth_bbox)
#         img = cv2.resize(mouths_img, (self.width, self.height))
#
#         img = img.astype(np.float32) / 255
#         img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#
#         img = cv2.resize(img, (self.width, self.height))
#
#         # normalize and convert to Channel Width Height
#         img = (img - self.normalize_mean) / self.normalize_scale
#         img = img.transpose((2, 0, 1))
#         img = np.expand_dims(img, axis=0)
#
#         out = self.exec_net_hp.infer(inputs={self.input_name_hp: img})
#         out = out[self.out_name_hp]
#         probs = np.squeeze(out)
#         probs = np.exp(probs) / sum(np.exp(probs))
#         is_open = probs[0] < probs[1]
#
#         return is_open
