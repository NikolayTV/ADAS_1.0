# Compatability:
- Openvino 2020.4
# Instruction on how to run

# Build docker
* docker build --tag adas:1.0 .

# Run docker 
*  xhost +
*  docker run -ti --rm --net=host --ipc=host \
   --name proc --device=/dev/video0:/dev/video0 \
   -v /tmp/.X11-unix:/tmp/.X11-unix \
   -e DISPLAY=$DISPLAY \
   adas:1.0


### Add --detach flag to run in background

FPS On CPU ~ 30.

![image](https://user-images.githubusercontent.com/32233366/149417093-0bace31d-8893-4b82-8623-b9522b92ca63.png)
1) Face detection
2) Gaze estimation
3) Blink detection
4) Head pose estimation 

# Install Openvino from 0:

wget https://registrationcenter-download.intel.com/akdlm/irc_nas/16803/l_openvino_toolkit_p_2020.4.287.tgz

Follow installation process

https://docs.openvinotoolkit.org/2020.4/openvino_docs_install_guides_installing_openvino_linux.html

then Run 
python3 main.py
