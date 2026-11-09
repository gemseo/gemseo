ARG FEDORA_VERSION=43@sha256:f01112b9a1f1d0504ad962d9adfa401425033f8ab067e0a821904ebdde5b8ca5

FROM fedora:$FEDORA_VERSION

RUN  dnf -y --setopt=install_weak_deps=False --nodocs update \
  && dnf -y --setopt=install_weak_deps=False --nodocs install \
                    python-unversioned-command \
                    python311 \
                    python312 \
                    python313 \
                    python314 \
                    graphviz \
                    git \
                    just \
                    uv \
  && dnf clean all
