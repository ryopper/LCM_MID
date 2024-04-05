# LCM_MID

Accelerating Pedestrian Trajectory Prediction with Diffusion Models - Application and Efficacy Evaluation of LCM-
https://onsite.gakkai-web.net/ipsj/abstract/data/pdf/2ZA-05.html

> As autonomous driving technology advances, the need for precise trajectory prediction between vehicles and pedestrians becomes increasingly critical. In pedestrian trajectory prediction, generative models such as Generative Adversarial Network (GAN) have been traditionally employed. However, these models have faced challenges, notably with unstable learning and generating diverse data. Diffusion models have significantly mitigated these issues, offering enhanced stability and diversity in data generation. However, the inherent computational complexity of their generation process limits their applicability in real-time inference scenarios. To address this challenge, this study applied the Latent Consistency Model (LCM), which is capable of generating high-quality predictions with fewer steps, to pedestrian trajectory prediction. Specifically, the approach combined predictions conditioned on past trajectories with those not conditioned on past trajectories. The LCM distillation model demonstrated improved accuracy over existing models, especially notable in the one-step sampling, where it exhibited a 29.7％ reduction in Average Distance Error (ADE) and a 32.0％ reduction in Final Distance Error (FDE), as observed in the results from the five different ETH/UCY datasets. Moreover, in the ETH dataset, compared to the two-step predictions of existing models, the one-step generation of the LCM distillation model showed about 0.04 seconds faster generation time, and a 13.2％ improvement in FDE. Furthermore, the diversity of the generated trajectories was also enhanced across the five ETH/UCY datasets, showing a 40.7％ increase in the one-step sampling and a 34.1％ increase in the two-step sampling, underscoring the model's effectiveness in both precision and diversity of trajectory prediction.

## Acknowledgments
This project was inspired by or based on the following repositories:
- [MID](https://github.com/Gutianpei/MID) by [Tianpei Gu et al] - [MIT License]
- [latent-consistency-model](https://github.com/luosiallen/latent-consistency-model) by [Simian Luo et al.] - [MIT License]
I am grateful for the contributions of these authors and their work which have significantly influenced the development of this project.

# Code

## Environment
    PyTorch == 1.12.1
    CUDA == 11.3

## Prepare Data
    Please refer to the [original document](https://github.com/Gutianpei/MID?tab=readme-ov-file#prepare-data). 

## Training

To apply the Latent Consistency Model to MID for distillation, the following two steps are necessary:
1. Train the MID model.
2. Use the trained MID model for distillation to create the LCM model.

### Step 1: Train the MID model.
    Please refer to the [original document](https://github.com/Gutianpei/MID?tab=readme-ov-file#training)
 
### Step 2: Use the trained MID model for distillation to create the LCM model.
    Basically the learning methods are same as Step1.
```python lcm_main.py --config configs/lcm_eth.yaml --dataset DATASET``` 

## Evaluation
    I will write later.


### Citations
```
    # MID
    @inproceedings{gu2022stochastic,
      title={Stochastic Trajectory Prediction via Motion Indeterminacy Diffusion},
      author={Gu, Tianpei and Chen, Guangyi and Li, Junlong and Lin, Chunze and Rao, Yongming and Zhou, Jie and Lu, Jiwen},
      booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
      pages={17113--17122},
      year={2022}
    }

    # LCM
    @misc{luo2023latent,
          title={Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference}, 
          author={Simian Luo and Yiqin Tan and Longbo Huang and Jian Li and Hang Zhao},
          year={2023},
          eprint={2310.04378},
          archivePrefix={arXiv},
          primaryClass={cs.CV}
    }
```