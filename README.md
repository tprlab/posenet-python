## PoseNet Python

This is a fork of:

https://github.com/rwightman/posenet-python

Which is a port of:

https://github.com/tensorflow/tfjs-models/tree/master/posenet

### Local usage

Run:

*python3 pose.py <path_to_image>*

The programs will print found keypoins and write an output image into posenet.jpg

### Local server

There is a flask web-service wrapping posenet estimation.

Run:

*python3 app.py*

It exports REST entry points:
- POST **/pose/keypoints** - get keypoints in JSON
- POST **/pose/pic** - get an image with drawn keypoints

There is a basic python client client/pose_client.py

#### Curl

Also could be called via curl:
- curl -X POST -F "file=@image.jpg" host:port/pose/keypoints
- curl -X POST -F "file=@image.jpg" host:port/pose/pic --output out.jpg

### Docker

The flask service wrapped into a [docker image](https://hub.docker.com/repository/docker/tprlab/tf-posenet-python).

- docker pull tprlab/tf-posenet-python
- docker run -p <host:port>:80 tprlab/tf-posenet-python
