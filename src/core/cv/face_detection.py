import cv2
from openvino.inference_engine import IECore
import numpy as np



class OpenVino_face_detector(object):
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

    def __init__(self, model_path='models/face-detection-retail-0004/FP32/face-detection-retail-0004'):

        # Prep for face detection
        ie = IECore()
        net_det  = ie.read_network(model=model_path+'.xml', weights=model_path+'.bin')
        self.input_name_det  = next(iter(net_det.inputs))                            # Input blob name "data"
        self.input_shape_det = net_det.inputs[self.input_name_det].shape                  # [1,3,384,672]
        self.out_name_det    = next(iter(net_det.outputs))                           # Output blob name "detection_out"
        self.exec_net_det    = ie.load_network(network=net_det, device_name='CPU', num_requests=1)
        del net_det

    def predict(self, img, output, confidence=0.5):
        height, width = img.shape[:2]

        # Prepare image for inference
        img = cv2.resize(img, (self.input_shape_det[3], self.input_shape_det[2]))
        img = img.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        img = img.reshape(self.input_shape_det)

        # Inference
        res_det = self.exec_net_det.infer(inputs={self.input_name_det: img})  # Detect faces

        # Extract bboxes
        output['faces_num'] = 0
        output['faces'] = []
        bboxes = []
        face_ID = 0
        for obj in res_det[self.out_name_det][0][0][:1]:  # obj = [ image_id, label, conf, xmin, ymin, xmax, ymax ]
            if obj[2] > confidence:  # Confidence > 0.5%
                # Rescale bbox to original img size
                xmin = abs(int(obj[3] * width))
                ymin = abs(int(obj[4] * height))
                xmax = abs(int(obj[5] * width))
                ymax = abs(int(obj[6] * height))


                # Clip bboxes by borders
                ymin, ymax = np.clip([ymin, ymax], 0, height)
                xmin, xmax = np.clip([xmin, xmax], 0, width)


                ### REFRESH OUTPUT JSON
                face_ID += 1
                output['faces_num'] = face_ID

                face_json = {}
                face_json['confidence'] = round(float(obj[2]), 3)
                face_json['bbox'] = {}
                face_json['bbox'] = {'xmin': int(xmin), 'ymin': int(ymin), 'xmax': int(xmax), 'ymax': int(ymax)}
                output['faces'].append(face_json)
                bboxes.append([xmin, ymin, xmax, ymax])

        return output, bboxes
