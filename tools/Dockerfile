ARG FEDORA_VERSION=34

FROM fedora:$FEDORA_VERSION

ARG TOX_VERSION=3.25.0

RUN  dnf -y update \
  && dnf -y install python \
  && dnf -y install pip \
  && dnf -y install python37 \
  && dnf -y install python38 \
  && dnf -y install python39 \
  && dnf -y install python310 \
  && dnf -y install git \
  && dnf -y install graphviz \
  && dnf clean all

RUN python -m pip install tox==$TOX_VERSION

RUN  curl -Os https://uploader.codecov.io/latest/linux/codecov \
  && chmod +x codecov \
  && mv codecov /usr/bin
