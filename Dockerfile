FROM ubuntu:jammy
WORKDIR /chat
RUN apt-get update
RUN apt install -y python3 pip bash git python3-virtualenv
