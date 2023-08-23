# IMQ-6D: Informed Modular Quantization for 6D Pose Estimation



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

If you want to train the network with IMQ-6D, current codebase is only for 2 bit FPN. To run the Swisscube example, simply run `train.sh` or invoke the below command.
```
$ CUDA_VISIBLE_DEVICES="0" python3 main.py --problem=SwissCube --topology=Widedepth
```
You have to uncomment few lines `problems/Swisscube/Widedepth/quantize.py` to quantize other parts of the network. We will clean and provide the whole codebase once the paper is published.
Furthermore, `problems/Swisscube/Widedepth/config.json` must be edited and path of pretrained WDR model should be provided under `experiment/url`. Other parts of the config file are self explanatory.


For testing the provided quantized model for Widedepth with 2 bit FPN, run `test.sh` or invoke the below command.
```
$ 
$ CUDA_VISIBLE_DEVICES="0" python3 main.py --problem=SwissCube --topology=Widedepth --mode="test"
```

