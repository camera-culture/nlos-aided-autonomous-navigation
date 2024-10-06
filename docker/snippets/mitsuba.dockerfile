# syntax = devthefuture/dockerfile-x
# Install mitsuba with cuda support

RUN apt-get update && \
        apt-get install --no-install-recommends -y \
        git && \
        apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

ARG MITSUBA_INSTALL_PATH="/opt/mitsuba"
ARG MITSUBA_BUILD_PATH="${MITSUBA_INSTALL_PATH}/build"
ARG MITSUBA_REPO="https://github.com/mitsuba-renderer/mitsuba3"
RUN git clone --recursive ${MITSUBA_REPO} ${MITSUBA_INSTALL_PATH}

RUN apt-get update && \
        apt-get install --no-install-recommends -y \
        libpng-dev \
        libjpeg-dev \
        libpython3-dev \
        python3-distutils \
        clang-15 \
        libc++-15-dev \
        libc++abi-15-dev \
        cmake \
        ninja-build && \
        apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

ARG MITSUBA_VARIANTS='"scalar_rgb","cuda_ad_rgb_polarized","cuda_ad_rgb","llvm_ad_rgb_polarized","llvm_ad_rgb"'
RUN mkdir -p ${MITSUBA_BUILD_PATH} && \
        cd ${MITSUBA_BUILD_PATH} && \
        CC=clang-15 CXX=clang++-15 cmake -GNinja .. && \
        sed -i "/\"enabled\": \[/ {n; s/.*/    ${MITSUBA_VARIANTS}/}" mitsuba.conf && \
        ninja

RUN echo "source ${MITSUBA_BUILD_PATH}/setpath.sh" >> ${USERSHELLPROFILE}