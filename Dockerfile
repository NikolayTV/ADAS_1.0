FROM openvino/ubuntu18_runtime:2020.4

USER root

# CMD source ${INTEL_OPENVINO_DIR}/bin/setupvars.sh & python3 app.py
RUN rm -r -f /ie-serving-py
COPY ./ /ie-serving-py
WORKDIR /ie-serving-py/src
RUN pip3 install -r requirements.txt
RUN ls
RUN apt-get update -y
RUN apt-get install -y libcanberra-gtk-module libcanberra-gtk3-module
CMD /bin/bash -c "source /opt/intel/openvino/bin/setupvars.sh && python3 main.py"

# CMD /bin/bash -c "source /opt/intel/openvino/bin/setupvars.sh && gunicorn app:flask_app -w 2"
# CMD /bin/bash -c "source /opt/intel/openvino/bin/setupvars.sh && python3 app.py"

