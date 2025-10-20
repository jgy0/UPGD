
A Lightweight Polarization-Guided Plug-in for Underwater Image
Enhancement
<img src="https://github.com/jgy0/UPGD/blob/main/img/upgd33.png">
## Introduction
In this project, we use Ubuntu 22.04.3 LTS, Python 3.10.13, Pytorch 2.1.1 cuda11.8 and one NVIDIA RTX 3090 GPU.

## datasets
in the paper, we mainly used two polarization datasets
<ol>
<li>RGBP-UIE</li> 
<li>UCPD</li>

</ol> 
you can get them by follow      

<a href="https://github.com/yudongLi-dlmu/RGBP-UIE">RGBP-UIE</a>    

<a href="https://github.com/jgy0/UPGD/blob/main/dataset/readme.md">UCPD</a>  

In the generalization experiment, we used two classic datasets and performed random splitting. [LSUI and UIEBD](https://pan.baidu.com/s/10cw8KLKZEkEhUkLpkHCqdQ?pwd=3ph5 )



### Ushape+PPM

#### Test
First, you need to download the [trained model weights](https://drive.google.com/drive/folders/1AOBtjGVVCA4w3jR5agVwh-A_pYUWiVg3?usp=drive_link), or retrain the model weights yourself. [Baidu Netdisk](https://pan.baidu.com/s/1AumnlX634cOP2I4dfRkqoA?pwd=zhth )(zhth )

In the article we tested a total of three datasets, if you need to test the indoor dataset, run test.py, outdoor dataset, run test_real2.py, ucpd dataset run
test_real_opt_data.py. They all load the same weight file, only slightly different in the data processing part.

#### Training

When you train your model, First, download the  [pre-training model](https://drive.google.com/file/d/19a_kDJTT5S96kzwQntEMhSxAPYw4xY2P/view), then you only need to run train_domain.py (with Domain-adversarial Training). Or you can use the simplified version train.py


### PUIE+PPM

You just need to download our code; the other files that need to be executed for the train and test methods are the same as the [original project code](https://github.com/zhenqifu/puie-net).

### ucolor+PPM
You just need to download our code; the other files that need to be executed for the train and test methods are the same as the [original project code](github.com/CV-Reimplementation/Ucolor-Reimplementation).

 
## Citation
```

```

## Acknowledgement

Our code is based on [Ushape](https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement), [PUIE](https://github.com/zhenqifu/puie-net), and [Ucolor](github.com/Li-Chongyi/Ucolor). Thanks for their outstanding contributions.
