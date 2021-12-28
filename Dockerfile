FROM    tensorflow/tensorflow
LABEL   email="wolafu@gmial.com"

RUN pip install flask flask-restx

COPY . /usr/src/app

WORKDIR /usr/src/app
EXPOSE 80

CMD python flask_api.py