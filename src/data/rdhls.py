import cv2
import sys

#VIDEO_URL = "http://bitdash-a.akamaihd.net/content/sintel/hls/playlist.m3u8"
#VIDEO_URL = "https://srv5.pplcase.ru:4443/hls/8_1597143364624_1597143374624.m3u8"
VIDEO_URL = "http://srv5.pplcase.ru:9898/hls/102_1597152079649_1597152099649.m3u8"

cap = cv2.VideoCapture(VIDEO_URL)
if (cap.isOpened() == False):
    print('!!! Unable to open URL')
    sys.exit(-1)

# retrieve FPS and calculate how long to wait between each frame to be display
fps = cap.get(cv2.CAP_PROP_FPS)
wait_ms = int(1000/fps)
print('FPS:', fps)

while(True):
    # read one frame
    ret, frame = cap.read()

    # TODO: perform frame processing here

    # display frame
    cv2.imshow('frame',frame)
    if cv2.waitKey(wait_ms) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
