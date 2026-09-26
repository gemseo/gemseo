ARG FEDORA_VERSION=43@sha256:a651ddf48ea28a06ed4e1e6519f51c9f47e7a5a138722ade87369b8fbb7e5b42

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
