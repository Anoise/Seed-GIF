# SSD-GIF: A State Space based Diffusion Model for Long-Term Wireless Traffic Generation, Imputation, and Forecasting

[![](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/Y-debug-sys/Diffusion-TS/blob/main/LICENSE) 
<img src="https://img.shields.io/badge/python-3.8-blue">
<img src="https://img.shields.io/badge/pytorch-2.0-orange">

> **Abstract:** Wireless Traffic (WT) analysis is crucial for network planning, network resource allocation, user behavior prediction, etc., and has received widespread attention recently. 
However, existing methods only focus on short-term pattern analysis with regular and complete data, while ignoring the fact that traffic data is not always available, or may be available but incomplete, especially for long-term periods.
To cope with this, in this paper, we focus on large-scale, long-term irregular WT data, and proposes to capture the macroscopic variation patterns of WT by unifying and enhancing the Generation, Imputation, and Forecasting (GIF) tasks.
In detail, we propose a unified state-space-based diffusion model to implement long-term WT-GIF tasks (SSD-GIF) by manipulating different guidance signals, including a trend-season decomposition mechanism to improve the model's performance and interpretability, and a self-guided training procedure based on Fourier reconstruction loss to efficiently implement the diffusion process.
Then, we compare the proposed method to various state-of-the-art methods on large-scale WT datasets. Extensive experimental results show that the proposed SSD-GIF outperforms those existing methods, with an average of 44\%, 6\% and 5\% performance improvement on generation, imputation, and forecasting tasks, respectively.

<p align="center">
  <img src="figs/scene_v3.jpg" alt="">
  <br>
  <b>Figure 1</b>: Overall Architecture of SSD-GIF.
</p>


## Dataset Preparation

All the four real-world datasets (Stocks, ETTh1, Energy and fMRI) can be obtained from [Google Drive](https://drive.google.com/file/d/11DI22zKWtHjXMnNGPWNUbyGz-JiEtZy6/view?usp=sharing). Please download **dataset.zip**, then unzip and copy it to the folder `./Data` in our repository.


## Running the Code

 The code requires conda3 (or miniconda3), and one CUDA capable GPU. The instructions below guide you regarding running the codes in this repository. 

### Environment & Libraries

The full libraries list is provided as a `requirements.txt` in this repo. Please create a virtual environment with `conda` or `venv` and run

~~~bash
(myenv) $ pip install -r requirements.txt
~~~

### Training & Sampling

For training, you can reproduce the experimental results of all benchmarks by runing

~~~bash
(myenv) $ python main.py --name {name} --config_file {config.yaml} --gpu 0 --train
~~~

**Note:** We also provided the corresponding `.yml` files under the folder `./Config` where all possible option can be altered. You may need to change some parameters in the model for different scenarios. For example, we use the whole data to train model for unconditional evaluation, then *training_ratio* is set to 1 by default. As for conditional generation, we need to divide data set thus it should be changed to a value < 1. 

While training, the script will save check points to the *results* folder after a fixed number of epochs. Once trained, please use the saved model for sampling by running

#### Unconstrained
```bash
(myenv) $ python main.py --name {name} --config_file {config.yaml} --gpu 0 --sample 0 --milestone {checkpoint_number}
```

#### Imputation
```bash
(myenv) $ python main.py --name {name} --config_file {config.yaml} --gpu 0 --sample 1 --milestone {checkpoint_number} --mode infill --missing_ratio {missing_ratio}
```

#### Forecasting
```bash
(myenv) $ python main.py --name {dataset_name} --config_file {config.yaml} --gpu 0 --sample 1 --milestone {checkpoint_number} --mode predict --pred_len {pred_len}
```


## Visualization and Evaluation

After sampling, synthetic data and orginal data are stored in `.npy` file format under the *output* folder, which can be directly read to calculate quantitative metrics such as discriminative, predictive, correlational and context-FID score. You can also reproduce the visualization results using t-SNE or kernel plotting, and all of these evaluational codes can be found in the folder `./Utils`. Please refer to `.ipynb` tutorial files in this repo for more detailed implementations.

**Note:** All the metrics can be found in the `./Experiments` folder. Additionally, by default, for datasets other than the Sine dataset (because it do not need normalization), their normalized forms are saved in `{...}_norm_truth.npy`. Therefore, when you run the Jupternotebook for dataset other than Sine, just uncomment and rewrite the corresponding code written at the beginning.

### Main Results

#### Standard TS Generation
<p align="center">
  <b>Table 1</b>: Results of 24-length Time-series Generation.
  <br>
  <img src="figures/fig2.jpg" alt="">
</p>

#### Long-term TS Generation
<p align="center">
  <b>Table 2</b>: Results of Long-term Time-series Generation.
  <br>
  <img src="figures/fig3.jpg" alt="">
</p>

#### Conditional TS Generation
<p align="center">
  <img src="figures/fig4.jpg" alt="">
  <br>
  <b>Figure 2</b>: Visualizations of Time-series Imputation and Forecasting.
</p>



## Citation
If you find this repo useful, please cite our paper via
```bibtex
The citation will coming soon!
```


## Acknowledgement

We appreciate the following github repos a lot for their valuable code base:

https://github.com/lucidrains/denoising-diffusion-pytorch

https://github.com/cientgu/VQ-Diffusion

https://github.com/XiangLi1999/Diffusion-LM

https://github.com/philipperemy/n-beats

https://github.com/salesforce/ETSformer

https://github.com/ermongroup/CSDI

https://github.com/jsyoon0823/TimeGAN
