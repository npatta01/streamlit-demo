FROM python:3.8

WORKDIR /code

ADD requirements.txt . 

RUN pip install -r requirements.txt 

ADD . . 

CMD [ "bash" ,"launch.sh"]



