ARG FEDORA_VERSION=39

FROM fedora:$FEDORA_VERSION

RUN  dnf -y update \
  && dnf -y install pipx \
                    python-unversioned-command \
                    python3-pip \
                    python39 \
                    python310 \
                    python311 \
                    python312 \
                    git \
                    graphviz \
                    jq \
                    https://gitlab.com/gitlab-org/cli/-/releases/v1.46.1/downloads/glab_1.46.1_Linux_x86_64.rpm \
  && dnf clean all \
  && rm -rf /var/cache/dnf

RUN  pipx install 'tox<5' --python python3.12 \
  && pipx inject tox tox-uv

# For pipx installed tools,
# explicitly update PATH because gitlab runner run the container in a way
# that prevent .bashrc to be sourced.
ENV PATH="/root/.local/bin:${PATH}"

RUN  curl -Os https://cli.codecov.io/latest/linux/codecov \
  && chmod +x codecov \
  && mv codecov /usr/bin
