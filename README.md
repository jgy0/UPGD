
A Lightweight Polarization-Guided Plug-in for Underwater Image
Enhancement

## Introduction
In this project, we use Ubuntu 22.04.3 LTS, Python 3.10.13, Pytorch 2.1.1 cuda11.8 and one NVIDIA RTX 3090 GPU.

## datasets
in the paper ,we use three datasets
<ol>
<li>RGBP-UIE</li>   [link](https://github.com/LintaoPeng/U-shape_Transformer_for_Underwater_Image_Enhancement)
<li>UCPD[]()</li>
<li>U300</li>
</ol> 



## Running
### Ushape+PPM

#### Test
First, you need to download the [trained model weights](https://drive.google.com/drive/folders/1AOBtjGVVCA4w3jR5agVwh-A_pYUWiVg3?usp=drive_link), or retrain the model weights yourself. [Baidu Netdisk](https://pan.baidu.com/s/1AumnlX634cOP2I4dfRkqoA?pwd=zhth )(zhth )

In the article we tested a total of three datasets, if you need to test the indoor dataset, run test.py, outdoor dataset, run test_real2.py, ucpd dataset run
test_real_opt_data.py. They all load the same weight file, only slightly different in the data processing part

#### Training

This part of the code will be released after accepted


### PUIE+PPM

You just need to download our code; the other files that need to be executed for the train and test methods are the same as the original project code.

### ucolor+PPM
You just need to download our code; the other files that need to be executed for the train and test methods are the same as the original project code.

 
## Compare the results of the method

This part  will be released after accepted
