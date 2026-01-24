ARG FEDORA_VERSION=44@sha256:64de49837325632232088e59c20518565a9d657095f9f3c2fa6033a42fb1412b

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
  && dnf clean all \
  && rm -rf /var/cache/dnf

RUN pip install 'tox<5' tox-uv --no-cache-dir --break-system-packages
