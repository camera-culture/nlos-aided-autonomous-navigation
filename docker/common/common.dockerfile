# SPDX-License-Identifier: MIT
# This has the common setup for all dockerfiles in the build system. Should come at the end of the dockerfile.
# Assumes the user is root

# Install dependencies
ARG APT_DEPENDENCIES=""
RUN apt-get update && \
        [ -z "${APT_DEPENDENCIES}" ] || apt-get install --no-install-recommends -y ${APT_DEPENDENCIES} && \
        apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Update apt such that it never runs without --no-install-recommends
RUN apt-config dump | grep -we Recommends -e Suggests | sed s/1/0/ | sudo tee /etc/apt/apt.conf.d/999norecommend

# Install python packages
ARG PIP_REQUIREMENTS=""
RUN [ -z "${PIP_REQUIREMENTS}" ] || pip install --no-cache-dir ${PIP_REQUIREMENTS}
# If a requirements.txt file is provided, install the packages
ARG PIP_REQUIREMENTS_FILE="requirements.txt"
COPY ${PIP_REQUIREMENTS_FILE} /tmp/requirements.txt
RUN [ -f /tmp/requirements.txt ] && \
        pip install --no-cache-dir -r /tmp/requirements.txt && \
        rm -rf /tmp/requirements.txt || true

# Update shell config
ARG DEFAULT_SHELL_ADD_ONS="export TERM=xterm-256color"
ARG USER_SHELL_ADD_ONS=""
RUN echo "${DEFAULT_SHELL_ADD_ONS}" >> ${USERSHELLPROFILE} && \
        sed -i 's/#force_color_prompt=yes/force_color_prompt=yes/' ${USERSHELLPROFILE} && \
        [ -z "${USER_SHELL_ADD_ONS}" ] || echo "${USER_SHELL_ADD_ONS}" >> ${USERSHELLPROFILE}