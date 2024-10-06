# Enhancing Autonomous Navigation by Imaging Hidden Objects using Single-Photon LiDAR

This is the code for the paper

**<a href="assets/young2024robospad.pdf">Enhancing Autonomous Navigation by Imaging Hidden Objects using Single-Photon LiDAR</a>**
<br>
<a href="https://AaronYoung5.github.io/">Aaron Young</a><sup>*</sup>, 
<a href="https://www.linkedin.com/in/nevindu-b-664a3613b/">Nevindu M. Batagoda</a><sup>*</sup>, 
<a href="https://www.linkedin.com/in/haorui-zhang1018">Harry Zhang</a>, 
<a href="https://akshatdave.github.io/">Akshat Dave</a>, 
<a href="https://sites.google.com/view/adithyapediredla/">Adithya Pediredla</a>, 
<a href="https://sbel.wisc.edu/negrut-dan/">Dan Negrut</a>, 
<a href="https://www.media.mit.edu/people/raskar/overview/">Ramesh Raskar</a>
<br>

[Paper](assets/young2024robospad.pdf) | [Code](https://github.com/camera-culture/nlos-aided-autonomous-navigation) | [Video](https://youtu.be/0GoUi0wrNMM)

## Abstract

Robust autonomous navigation in environments with limited visibility remains a critical challenge in robotics. We present a novel approach that leverages Non-Line-of-Sight (NLOS) sensing using single-photon LiDAR to improve visibility and enhance autonomous navigation. Our method enables mobile robots to ``see around corners" by utilizing multi-bounce light information, effectively expanding their perceptual range without additional infrastructure. We propose a three-module pipeline: (1) Sensing, which captures multi-bounce histograms using SPAD-based LiDAR; (2) Perception, which estimates occupancy maps of hidden regions from these histograms using a convolutional neural network; and (3) Control, which allows a robot to follow safe paths based on the estimated occupancy. We evaluate our approach through simulations and real-world experiments on a mobile robot navigating an L-shaped corridor with hidden obstacles. Our work represents the first experimental demonstration of NLOS imaging for autonomous navigation, paving the way for safer and more efficient robotic systems operating in complex environments. We also contribute a novel dynamics-integrated transient rendering framework for simulating NLOS scenarios, facilitating future research in this domain.

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

## Citation

```bibtex
@inproceedings{young2024robospad,
    author = {Young, Aaron and Batagoda, Nevindu M. and Zhang, Harry and Dave, Akshat and 
        Pediredla, Adithya and Negrut, Dan and Raskar, Ramesh},
    title = {Enhancing Autonomous Navigation by Imaging Hidden Objects using Single-Photon LiDAR},
    booktitle = {ArXiv},
    year = {2024}
}
```