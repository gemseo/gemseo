ARG FEDORA_VERSION=45@sha256:af8cdc432037f5e8e288bbc26c2b55b96000a911ec5b37959cd464d830f6cc5b

FROM fedora:$FEDORA_VERSION

RUN  dnf -y update \
  && dnf -y install python-unversioned-command \
                    python3-pip \
                    python310 \
                    python311 \
                    python312 \
                    python313 \
                    graphviz \
                    git \
                    just \
  && dnf clean all \
  && rm -rf /var/cache/dnf

RUN pip install 'tox<5' tox-uv --no-cache-dir --break-system-packages
