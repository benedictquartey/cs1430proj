# Project Brief 

The input to this project is a video capturing a real world scene. To train nerf models on real life scenes we use the approach of local light field fusion (LLFF) to generate poses from images. We first extract frames from the video file, then perform structure from motion using COLMAP to obtain 6-DoF camera poses and near/far depth bounds for the scene. We also generate a config file with various parameters for training a nerf model on the data we collected. Finally we can train a nerf model.

Update on dynamic nerf model training  ...


## Running Instructions
1. Recover camera poses and generate config file with Nerf training parameters

    python data_pipeline.py --expname arm --fps 20 --video sample_videos/helmet.MOV 


2. Train nerf using generated config file
    python run_nerf.py --config configs/dancingQ.txt --no_ndc --spherify --lindisp

Additional arguments "--no_ndc --spherify --lindisp" recommended for a spherically captured 360 scene, no need to add them for forward facing scenes.


## Citations
Kindly find citations to the various resources used in this project below

```
@inproceedings{mildenhall2020nerf,
  title={NeRF: Representing Scenes as Neural Radiance Fields for View Synthesis},
  author={Ben Mildenhall and Pratul P. Srinivasan and Matthew Tancik and Jonathan T. Barron and Ravi Ramamoorthi and Ren Ng},
  year={2020},
  booktitle={ECCV},
}
```
```
@misc{lin2020nerfpytorch,
  title={NeRF-pytorch},
  author={Yen-Chen, Lin},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished={\url{https://github.com/yenchenlin/nerf-pytorch/}},
  year={2020}
}
```
```
@article{mildenhall2019llff,
  title={Local Light Field Fusion: Practical View Synthesis with Prescriptive Sampling Guidelines},
  author={Ben Mildenhall and Pratul P. Srinivasan and Rodrigo Ortiz-Cayon and Nima Khademi Kalantari and Ravi Ramamoorthi and Ren Ng and Abhishek Kar},
  journal={ACM Transactions on Graphics (TOG)},
  year={2019},
}
```