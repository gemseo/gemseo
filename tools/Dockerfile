ARG FEDORA_VERSION=41

FROM fedora:$FEDORA_VERSION

RUN  dnf -y update \
  && dnf -y install python-unversioned-command \
                    python3-pip \
                    python39 \
                    python310 \
                    python311 \
                    python312 \
                    git \
                    graphviz \
  && dnf clean all \
  && rm -rf /var/cache/dnf

RUN pip install 'tox<5' tox-uv --no-cache-dir --break-system-packages

RUN  curl -Os https://cli.codecov.io/latest/linux/codecov \
  && chmod +x codecov \
  && mv codecov /usr/bin
