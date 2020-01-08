Copyright (c) 2019 ETH Zurich, Michael Hersche


# Compressing Subject-specific Brain--Computer Interface Models into One Model by Superposition in Hyperdimensional Space

In this repository, we share the code for compressing subject-specific BCI models.  
For details, please refer to the papers below. 

If this code proves useful for your research, please cite
> Michael Hersche, Philipp Rupp, Luca Benini, Abbas Rahimi, "Compressing Subject-specific Brain--Computer Interface Models into One Model by Superposition in Hyperdimensional Space", in ACM/IEEE Design, Automation, and Test in Europe Conference (DATE), 2020.  
<!--DOI (preprint): [10.3929/ethz-b-000282732](https://www.research-collection.ethz.ch/handle/20.500.11850/282732). Available on [arXiv](https://arxiv.org/pdf/1808.05488). -->



#### Installing Dependencies
You will need a machine with a CUDA-enabled GPU and the Nvidia SDK installed to compile the CUDA kernels.
Further, we have used conda as a python package manager and exported the environment specifications to `conda-env-bci-superpos.yml`. 
You can recreate our environment by running 

```
conda env create -f conda-env-bci-superpos.yml -n myBCIsupposEnv 
```
Make sure to activate the environment before running any code. 

#### Download the BCI competition IV 2a dataset
EEGNet: 
Download the `.mat` files of 4-class MI dataset with 9 subjects (001-2014) from [here](http://bnci-horizon-2020.eu/database/data-sets), unpack it, and put into folder `dataset/EEGNet`

ShallowConvnet: 
Download `.gdf` files from [here](http://bbci.de/competition/iv/) by requesting access under "Download of data sets". You'll receive an account and can download files. Then put them into folder `dataset/shallowconvnet`. The labels need to be downloaded seperately also [here](http://bbci.de/competition/iv/) under "True Labels of Competition's Evaluation Sets".

<!--
#### Download the pretrained models 


```
curl https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/278294/model-poseDetection-full.tar | tar xz
curl https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/278294/model-poseDetection-baseline.tar | tar xz
curl https://www.research-collection.ethz.ch/bitstream/handle/20.500.11850/278258/dataset-poseDetection.tar | tar xz
```
-->


### Step-by-step Guide
There are two networks to test the compression -- EEGNet and Shallow ConvNet. You can test them by running `main.py` either in `code/EEGnet/` or `code/ShallowConvNet/`. The original and compressed models are stored in the corresponding `models/` folder. Accuracy results are available in `results/`. 

<!--
### Software Architecture
The whole setup consists of several parts: 

- The CBinfer implementation of the convolution and max-pooling operations
- Infrastructure common to both applications, such as `tx2power` and `evalTools`
- Application-specific code

#### pycbinfer
PyCBinfer consists of several classes and functions. 

- The conversion function to port a normal DNN model to a CBinfer model as well as some utility functions to reset the model's state can be found in the `__init__.py`. It also includes the function to tune the threshold parameters, which takes many application-specific functions as arguments, such as dataset loader, preprocessor, postprocessor, accuracy evaluator, the model, ...
- The CBConv2d and CBPoolMax2d classes can be found in the `conv2.py` file. They are implementations of a torch.nn.module and include the handling of the tensors, switching between fine-grained and coarse-grained CBinfer, and the overall algorithm flow. 
- The individual processing steps and the lower-level interface to the C/CUDA functions using cffi can be found in `conv2d_cg.py` and `conv2d_fg.py`, respectively. 
- The `cbconv2d_*_backend.cu` contain the CUDA kernels and the C launchers thereof. 

#### evalTools
They contain functions to apply a model to a frame sequence, benchmark performance, measure power, and to filter layers on CBinfer-based models. The benchmark functions are generic and take the following as arguments:

- a model
- a frame set and the number of frames to process
- a preprocessing function to prepare the data
- a postprocessing function to transform the network output to a meaningful result

#### application-specific function for pose detection

- The application-specific code consists of a `modelConverter` script, which load the baseline network, invokes the loading of the dataset, contains all the details of how the network should be converted (which layers, etc.)  and ultimately calls pycbinfer's threshold tuning function. 
- The `poseDetEvaluator` script contains a class with all the metrics and pre-/post-processing functions. 
- The `videoSequenceReader` contains the loader functions for the dataset.
- The `openPose` folder contains all the files to run the OpenPose network and pre-/post-processing until the final extraction of the skeleton. 
- The two evaluation scripts `eval01` and `eval03` contain the code to perform the analysis and visualize the results shown in the paper, e.g. throughput-accuracy trade-offs, power and energy usage, baseline performance measurement, change propagation analyses, etc.

#### tx2power
The `tx2power` file provides some standalone easy-to-use current/voltage/power/energy measurement tools for the Tegra X2 module and the Jetson TX2 board. 
It allows to analyze where the power is dissipated (module: main/all, cpu, ddr, gpu, soc, wifi; board: main/all, 5V0-io-sys, 3v3-sys, 3v3-io-sleep, 1v8-io, 3v3-m.2)
The PowerLogger class provides the means to measure power traces, record events (markers), visualize the data, and obtain key metrics such as total energy. 
-->

### License and Attribution
Please refer to the LICENSE file for the licensing of our code.
<!--For the pose detection application demo, we heavily modified [this](https://github.com/tensorboy/pytorch_Realtime_Multi-Person_Pose_Estimation) OpenPose implementation. -->

