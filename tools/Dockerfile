ARG FEDORA_VERSION=34

FROM fedora:$FEDORA_VERSION

RUN  dnf -y update \
  && dnf -y install pipx \
  && dnf -y install python38 \
  && dnf -y install python39 \
  && dnf -y install python310 \
  && dnf -y install git \
  && dnf -y install graphviz \
  && dnf clean all

ARG TOX_VERSION=4.4.7

RUN  pipx install 'tox<4' \
  && pipx install 'tox<4' --suffix 3 \
  && pipx install tox==$TOX_VERSION --suffix 4 \
  && pip cache purge

# Explicitely update PATH because gitlab runner run the container in a way
# that prevent .bashrc to be sourced.
ENV PATH="/root/.local/bin:${PATH}"
