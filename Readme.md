# Modular Quantization-Aware Training for 6D Object Pose Estimation

> [Saqib Javed](https://saqibjaved1.github.io/), [Chengkun Li](https://charlieleee.github.io/), [Andrew Price](https://spaceguy-price.github.io/), [Yinlin Hu](https://yinlinhu.github.io/), [Mathieu Salzmann](https://scholar.google.com/citations?user=n-B0jr4AAAAJ&hl=en)

[![Website](https://img.shields.io/badge/Project-Website-87CEEB)](https://saqibjaved1.github.io/MQAT_/)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-FFF933)](https://arxiv.org/abs/2303.06753)

We introduce Modular Quantization-Aware Training (MQAT), a novel mixed-precision Quantization-Aware Training (QAT) approach that leverages the modular structure of modern architectures for efficient neural network compression. MQAT is the first to exploit the modular trend, using an optimized module-wise quantization sequence shown to be effective for 6D pose estimation networks. With MQAT, we achieve substantial accuracy gains over competitive QAT methods, even surpassing full-precision accuracy for certain single-stage 6D pose estimation networks. Extensive validation across datasets, architectures, and quantization algorithms demonstrates MQAT’s robustness and versatility.

## Citation
If you find our work useful. Please consider giving a star :star: and a citation.
```
@article{javed2024modular,
  author    = {Javed, Saqib and Li, Chengkun and Price, Andrew and Hu, Yinlin and Salzmann, Mathieu},
  title     = {Modular Quantization-Aware Training for 6D Object Pose Estimation},
  journal   = {TMLR},
  year      = {2024},
  url       = {https://openreview.net/forum?id=lIy0TEUou7}
}
```

## Installation
In the following, I assume you have pulled the repository to your local machine at your home folder, which I will denote with `~`.

First, install Horovod's prerequisites by following the [official guide](https://horovod.readthedocs.io/en/stable/gpus_include.html).

Then, install prerequisites using 
```
$ ./setup.sh
```
For data and logs configuration, edit `cfg/hard_storage.json` and add path to the base directory of your repository. Put data under `data` directory and your logs will be stored under `logs` directory.
You can download data from [here](https://u.pcloud.link/publink/show?code=XZ7ExHVZNw3kUckPM8SOWzepcE6ANF9jpPYX).

Pretrained model can be downloaded from [here](https://drive.google.com/drive/folders/1tkxBM4K4Bl1RqiYSwC0tL1rMFyevHj7h).



## Running

If you want to train the network with MQAT, current codebase is only for 2 bit FPN. To run the Swisscube example, simply run `train.sh` or invoke the below command.
```
$ CUDA_VISIBLE_DEVICES="0" python3 main.py --problem=SwissCube --topology=Widedepth
```
You have to uncomment few lines `problems/Swisscube/Widedepth/quantize.py` to quantize other parts of the network. We will clean and provide the whole codebase soon.
Furthermore, `problems/Swisscube/Widedepth/config.json` must be edited and path of pretrained WDR model should be provided under `experiment/url`. Other parts of the config file are self explanatory.


For testing the provided quantized model for Widedepth with 2 bit FPN, run `test.sh` or invoke the below command.
```
$ CUDA_VISIBLE_DEVICES="0" python3 main.py --problem=SwissCube --topology=Widedepth --mode="test"
```

### Acknowledgements
Our code is based on [Quantlab](https://github.com/pulp-platform/quantlab) and [WDR-Pose](https://github.com/cvlab-epfl/wide-depth-range-pose) repository. We thank the authors for releasing their code. 

