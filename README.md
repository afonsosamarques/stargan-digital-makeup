# Applying StarGAN to Digital Makeup
## (October 2018)


To install required packages, simply run
```bash
pip install -r stargan-digital-makeup/requirements.txt
```


### Instructions

Our code sits under the main directory `codebase` and its structure follows a clear pattern: main and solver, a directory with the network architecture, and another with utilities for processing data.

To run, simply define the arguments required in `main.py`, the entrypoint into the codebase. Inside `data_utils`, there is also a pre-processing script that may be run separately for purposes of data augmentation. Again, simply define the arguments required in `preprocess.py`.


### References

This is an adaptation and application to the digital makeup space of StarGAN.
* Choi, Y., Choi, M., Kim, M., Ha, J., Kim, S., & Choo, J. "StarGAN: Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation." In 2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2018.

We implement three main network architectures for our Generator. We consider, from the literature, U-Net and Dilated Residual Network, and propose a new architecture, Dilated Residual U-Net, which combines elements of the two.
* Ronneberger, O., Fischer, P., and Brox, T. "U-Net: Convolutional networks for biomedical image segmentation." In MICCAI, volume 9351 of Lecture
Notes in Computer Science, pages 234–241. 2015.
* Yu, F., Koltun, V., and Funkhouser, T. "Dilated Residual Networks". In Computer Vision and Pattern Recognition (CVPR), 2017 IEEE Conference, pages 636–644. 2017.

