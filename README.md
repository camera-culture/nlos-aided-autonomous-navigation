# Enhancing Autonomous Navigation by Imaging Hidden Objects using Single-Photon LiDAR

This is the repo for the paper

**[Enhancing Autonomous Navigation by Imaging Hidden Objects using Single-Photon LiDAR](docs/assets/young2024robospad.pdf)**  
[Aaron Young](https://AaronYoung5.github.io/)<sup>\*</sup>, [Nevindu M. Batagoda](https://www.linkedin.com/in/nevindu-b-664a3613b/)<sup>\*</sup>, [Harry Zhang](https://www.linkedin.com/in/haorui-zhang1018), [Akshat Dave](https://akshatdave.github.io/), [Adithya Pediredla](https://sites.google.com/view/adithyapediredla/), [Dan Negrut](https://sbel.wisc.edu/negrut-dan/), [Ramesh Raskar](https://www.media.mit.edu/people/raskar/overview/)  

[Paper](docs/assets/young2024robospad.pdf) | [Code](https://github.com/camera-culture/nlos-aided-autonomous-navigation) | [Video](https://youtu.be/0GoUi0wrNMM)

## Abstract

Robust autonomous navigation in environments with limited visibility remains a critical challenge in robotics. We present a novel approach that leverages Non-Line-of-Sight (NLOS) sensing using single-photon LiDAR to improve visibility and enhance autonomous navigation. Our method enables mobile robots to ``see around corners" by utilizing multi-bounce light information, effectively expanding their perceptual range without additional infrastructure. We propose a three-module pipeline: (1) Sensing, which captures multi-bounce histograms using SPAD-based LiDAR; (2) Perception, which estimates occupancy maps of hidden regions from these histograms using a convolutional neural network; and (3) Control, which allows a robot to follow safe paths based on the estimated occupancy. We evaluate our approach through simulations and real-world experiments on a mobile robot navigating an L-shaped corridor with hidden obstacles. Our work represents the first experimental demonstration of NLOS imaging for autonomous navigation, paving the way for safer and more efficient robotic systems operating in complex environments. We also contribute a novel dynamics-integrated transient rendering framework for simulating NLOS scenarios, facilitating future research in this domain.

## Citation

```bibtex
@inproceedings{young2024enhancing,
    author = {Young, Aaron and Batagoda, Nevindu M. and Zhang, Harry and Dave, Akshat and 
        Pediredla, Adithya and Negrut, Dan and Raskar, Ramesh},
    title = {Enhancing Autonomous Navigation by Imaging Hidden Objects using Single-Photon LiDAR},
    booktitle = {ArXiv},
    year = {2024}
}
```