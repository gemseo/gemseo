ARG FEDORA_VERSION=43

FROM fedora:$FEDORA_VERSION

RUN  dnf -y update \
  && dnf -y install python-unversioned-command \
                    python3-pip \
                    python310 \
                    python311 \
                    python312 \
                    python313 \
                    python314 \
                    graphviz \
                    git \
                    just \
  && dnf clean all \
  && rm -rf /var/cache/dnf

RUN pip install 'tox<5' tox-uv --no-cache-dir --break-system-packages
