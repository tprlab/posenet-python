FROM python:3.7-stretch

RUN pip3 install flask
RUN pip3 install protobuf
RUN pip3 install opencv_python
RUN pip3 install pyyaml
RUN pip3 install scipy
RUN pip3 install tensorflow==1.15

ADD https://github.com/tprlab/posenet-python/archive/master.zip /
RUN unzip /master.zip

EXPOSE 80

CMD ["python3", "/posenet-python-master/app.py"]
