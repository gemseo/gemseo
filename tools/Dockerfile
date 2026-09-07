ARG FEDORA_VERSION=43

FROM fedora:$FEDORA_VERSION

RUN  dnf -y --setopt=install_weak_deps=False --nodocs update \
  && dnf -y --setopt=install_weak_deps=False --nodocs install \
                    python-unversioned-command \
                    python310 \
                    python311 \
                    python312 \
                    python313 \
                    python314 \
                    graphviz \
                    git \
                    just \
                    uv \
  && dnf clean all
