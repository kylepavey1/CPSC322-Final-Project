FROM continuumio/anaconda3:2020.11

ADD . /code
WORKDIR /code
ENTRYPOINT ["python", "flask_app.py"]