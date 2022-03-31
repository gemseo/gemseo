ARG FEDORA_VERSION=34

FROM fedora:$FEDORA_VERSION

ARG TOX_VERSION=3.24.4
ARG CODECOV_VERSION=2.1.12

# Fedora tox is outdated, install with pip.
# openturns for python 27 requires libnsl.
RUN  dnf -y update \
  && dnf -y install python \
  && dnf -y install pip \
  && dnf -y install python27 \
  && dnf -y install python36 \
  && dnf -y install python37 \
  && dnf -y install python38 \
  && dnf -y install python39 \
  && dnf -y install git \
  && dnf -y install graphviz \
  && dnf -y install libnsl \
  && dnf clean all \
  && python3.9 -m pip install \
     tox==$TOX_VERSION \
     codecov==$CODECOV_VERSION
