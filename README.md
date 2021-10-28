# ExGAN

<p>
  <a href="https://aaai.org/Conferences/AAAI-21/">
    <img src="http://img.shields.io/badge/AAAI-2021-red.svg">
  </a>
    <a href="https://arxiv.org/pdf/2009.08454.pdf">
      <img src="http://img.shields.io/badge/Paper-PDF-brightgreen.svg">
  </a>
   <a href="https://www.comp.nus.edu.sg/~sbhatia/assets/pdf/ExGAN_poster.pdf">
      <img src="http://img.shields.io/badge/Poster-PDF-green.svg">
  </a>
  <a href="https://www.comp.nus.edu.sg/~sbhatia/assets/pdf/ExGAN_slides.pdf">
      <img src="http://img.shields.io/badge/Slides-PDF-ff9e18.svg">
  </a>
  <a href="https://youtu.be/7s7RkCeeoeg">
    <img src="http://img.shields.io/badge/Talk-Youtube-ff69b4.svg">
  </a>
    <a href="https://colab.research.google.com/github/Stream-AD/ExGAN/blob/master/ExGAN-Colab.ipynb">
    <img src="https://colab.research.google.com/assets/colab-badge.svg">
  </a>
  <a href="https://github.com/Stream-AD/ExGAN/blob/master/LICENSE">
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg">
  </a>
</p>


Implementation of

- [ExGAN: Adversarial Generation of Extreme Samples](https://arxiv.org/pdf/2009.08454.pdf). *Siddharth Bhatia⁺, Arjit Jain⁺, Bryan Hooi.* (⁺ denotes equal contribution). AAAI 2021.

![](https://www.comp.nus.edu.sg/~sbhatia/assets/img/exgan.png)
Our goal is to generate samples which are both realistic and extreme, based on any user-specified extremeness criteria
(in this case, high total rainfall). Left: Existing GAN-based approaches generate typical rainfall patterns, which have low (green)
to moderate (red) rainfall. Right: Extreme samples generated by our approach have extreme (violet) rainfall, and realistic spatial
patterns resembling that of real floods.

[KDnuggets](https://www.kdnuggets.com/2021/02/adversarial-generation-extreme-samples.html) and [AIhub](https://aihub.org/2020/10/01/adversarial-generation-of-extreme-samples/) covered ExGAN!

## Getting Started

### Environment
This code has been tested on Debian GNU/Linux 9 with a 12GB Nvidia GeForce RTX 2080 Ti GPU, CUDA Version 10.2 and PyTorch 1.5.  

### Reproducing the Experiments

The first step is to get the data. We have prepared a script to download precipitation data from [water.weather.gov/precip/](https://water.weather.gov/precip/). The data downloaded is for the duration 2010 to 2016 as mentioned in the paper.

```
python PrepareData.py
```
Now, we can train a DCGAN Baseline on this data. 

```
python DCGAN.py
```
Distribution Shifting on this DCGAN can be performed using 
```
python DistributionShifting.py
```
Finally, we can train ExGAN on the distribution shifted dataset. 
```
python ExGAN.py
```

The training of ExGAN and DCGAN can be monitored using TensorBoard. 
```
tensorboard --logdir [DCGAN\EXGAN]
```

### Evaluation and Visualizing the Results

Generate samples from DCGAN of different extremeness probabilities, and mark the time taken in sampling.
```
python DCGANSampling.py
```
Similarly, Generate samples from ExGAN of different extremeness probabilities, and mark the time taken in sampling.
```
python ExGANSampling.py
```

We provide FID.py to calculate the FID score, as described in the paper, on the trained models. 
We also provide DCGANRecLoss.py, and ExGANRecLoss.py to evaluate DCGAN and ExGAN on their Reconstruction Loss
Note that, both of these metrics are calculated on a test set. PrepareData.py can be used to curate the test set for the duration described in the paper.

The python file, plot.py, contains the code for plotting rainfall maps like the figures included in the paper. Note that this requires the Basemap library from matplotlib. 

We also provide an IPython notebook, EVT_Analysis.ipynb to play with and visualize the effect of different thresholds for the Peaks over 
Threshold approach.


## Citation

If you use this code for your research, please consider citing our arXiv preprint

```bibtex
@inproceedings{bhatia2021exgan,
    title={ExGAN: Adversarial Generation of Extreme Samples},
    author={Siddharth Bhatia and Arjit Jain and Bryan Hooi},
    booktitle={AAAI Conference on Artificial Intelligence (AAAI)},
    year={2021}
}
```

## Things to Check
1. What does the image look like?
2. How to visulize it?


