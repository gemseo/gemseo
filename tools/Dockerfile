ARG FEDORA_VERSION=36

FROM fedora:$FEDORA_VERSION

RUN  dnf -y update \
  && dnf -y install pipx \
                    python-unversioned-command \
                    python3-pip \
                    python39 \
                    python310 \
                    python311 \
                    git \
                    graphviz \
                    jq \
                    https://gitlab.com/gitlab-org/cli/-/releases/v1.37.0/downloads/glab_1.37.0_Linux_x86_64.rpm \
  && dnf clean all

RUN  pipx install 'tox<5'

# For pipx installed tools,
# explicitly update PATH because gitlab runner run the container in a way
# that prevent .bashrc to be sourced.
ENV PATH="/root/.local/bin:${PATH}"

RUN  curl -Os https://cli.codecov.io/latest/linux/codecov \
  && chmod +x codecov \
  && mv codecov /usr/bin
