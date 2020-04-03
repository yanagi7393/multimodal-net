FROM pytorch/pytorch:latest

RUN apt-get update \
  && apt-get upgrade -y \
  && apt-get install build-essential -y \
  && apt-get install -y \
    git-core \
    vim \
    gettext-base \
    ffmpeg \
  && apt-get clean \
  && rm -rf /var/lib/apt/lists/*


ADD dockerfiles/requirements.txt requirements.txt
RUN pip install -r requirements.txt

ADD . /app
WORKDIR /app

ENV PYTHONPATH $PYTHONPATH:/app:/app/src
