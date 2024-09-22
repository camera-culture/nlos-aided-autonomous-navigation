# Enhancing Autonomous Navigation by Imaging Hidden Objects using Single-Photon LiDAR

## How to Run

> [!NOTE]
> See [Docker](#docker) for a docker-centric way to run the code.

To install the necessary packages, run the following command:

```bash
pip install -r requirements.txt
```

### Docker

This repository also includes Dockerfiles and orchestration scripts. To run the code, we use a utility called [`atk`](https://projects.sbel.org/autonomy-toolkit). To install `atk`, run the following command:

```bash
pip install autonomy-toolkit
```

Then, to build the Docker image(s), run the following command:

```bash
atk dev -b -s tbrn vnc
```

To run and attach to the Docker container, run the following command:

```bash
$ atk dev -u -s vnc # optional
$ atk dev -ua -s tbrn -o gpus vnc
```

The above command will spin up the `tbrn-tbrn` container and attach gpus to it. This will only work on a machine with nvidia GPUs. Omit `gpus` if your machine does not have GPUs.

The `vnc` keyword is optional and will start a separate container with an vnc server. This is useful for debugging and visualizing the results. You may also omit the separate `vnc` start up command and use `x11` as the optional for the `tbrn` command instead. This enables x11 forwarding from the container to the host for a more native-like experience.

## Install Instructions

### PySpin