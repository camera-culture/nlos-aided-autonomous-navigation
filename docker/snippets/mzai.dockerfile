# syntax = devthefuture/dockerfile-x

# Install necessary packages for building the driver
RUN apt-get update && \
    apt-get install --no-install-recommends -y \
    build-essential \
    i2c-tools \
    linux-headers-generic \
    nano && \
    apt-get clean && apt-get autoremove -y && rm -rf /var/lib/apt/lists/*

# Define the path where the driver files will be copied
ARG DRIVER_HOST_PATH="docker/data/vl5317ch_driver"
ARG DRIVER_INSTALL_PATH="/opt/vl53l7ch_driver"
ARG DRIVER_BUILD_PATH="${DRIVER_INSTALL_PATH}/uld-driver"
COPY ${DRIVER_HOST_PATH} ${DRIVER_INSTALL_PATH}

# # Build the driver in user mode
# RUN cd ${DRIVER_BUILD_PATH}/user/test && \
#     make

# # Build the kernel module (if needed, for kernel mode)
# RUN cd ${DRIVER_BUILD_PATH}/kernel && \
#     make clean && make

# # Optionally insert the kernel module (kernel mode)
# RUN cd ${DRIVER_BUILD_PATH}/kernel && \
#     make insert || true