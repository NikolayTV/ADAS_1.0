import cv2
import sys
import time 

VIDEO_URL = "data/petrosyan.mp4"

cap = cv2.VideoCapture(VIDEO_URL)
fps = cap.get(cv2.CAP_PROP_FPS)

url = 'http://0.0.0.0:7777/'

# prepare headers for http request
content_type = 'image/jpeg'
headers = {'content-type': content_type}

while(True):
    # read one frame
    ret, frame = cap.read()
    if ret:
        frame_start_time = time.time()

        # TODO: perform frame processing here

        # encode image as jpeg
        retval, buffer = cv2.imencode('.jpg', frame)
        frame_encoded = base64.b64encode(buffer)
        response = requests.post(url, data={"image":frame_encoded})
        print(response.json())

        if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
            break

        now = time.time()
        frame_time = now - frame_start_time
        fps = 1.0 / frame_time
        print(fps)
