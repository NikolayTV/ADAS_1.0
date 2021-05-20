import cv2
from openvino.inference_engine import IECore
import numpy as np


class OpenVino_emotion_recognition(object):
    def __init__(self, confidence_thr=0.3,  model_path='models/emotions-recognition-retail-0003/FP32/emotions-recognition-retail-0003'):


        # Prep for face detection
        ie = IECore()
        net_det = ie.read_network(model=model_path + '.xml', weights=model_path + '.bin')
        self.input_name_det = next(iter(net_det.inputs))  # Input blob name "data"
        self.input_shape_det = net_det.inputs[self.input_name_det].shape  # [1,3,384,672]
        self.out_name_det = next(iter(net_det.outputs))  # Output blob name "detection_out"
        self.exec_net_det = ie.load_network(network=net_det, device_name='CPU', num_requests=1)
        self.SENTIMENT_LABEL = ['neutral', 'happy', 'sad', 'surprise', 'anger']

        del net_det

    def predict(self, img_ori, output):

        emotions = []
        # For each face in json crop image and run head pose model
        for num, i in enumerate(output['faces']):
            xmin, ymin, xmax, ymax = i['bbox'].values()
            face = img_ori[ymin:ymax, xmin:xmax]

            face_processed = cv2.resize(face, (self.input_shape_det[3], self.input_shape_det[2]))
            face_processed = face_processed.transpose((2, 0, 1))
            face_processed = face_processed.reshape(self.input_shape_det)

            # Inference
            res_hp = self.exec_net_det.infer(inputs={self.input_name_det: face_processed})  # Run head pose estimation

            emotion_int = np.argmax(res_hp[self.out_name_det])
            emotion = self.SENTIMENT_LABEL[emotion_int]
            emotions.append(emotion)

            # REFRESH OUTPUT JSON
            output['faces'][num]['emotion'] = emotion

        return output, emotions
