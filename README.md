# DeepRF
[![DOI](https://zenodo.org/badge/366596735.svg)](https://zenodo.org/badge/latestdoi/366596735)

DeepRF is an AI-powered RF pulse design framework which utilizes the self-learning characteristics 
of deep reinforcement learning (DRL) to generate a novel RF pulse. \
For more details, see [preprint.](https://arxiv.org/abs/2105.03061)

## System requirements

### Hardware
At least one NVIDIA GPU is needed, which supporting CUDA 10.2.
Computing environment in which we tested the code is as follows.

- CPU: Intel Xeon Gold 5218 2.30 GHz
- GPU: NVIDIA Quadro RTX 8000
- RAM: 128 GB

### Software
- Ubuntu 18.04 LTS
- MATLAB 2019a
- Anaconda & Python packages:
instead of listing all Python packages to run DeepRF, we provide .yml or .txt file that can be used to 
create an Anaconda environment based on exact package versions. See below.

## Installation guide
To install all Python packages to run DeepRF,

0. Install appropriate versions of NVIDIA driver, CUDA, and cuDNN for your GPU
1. Download and install [Anaconda](https://www.anaconda.com/products/individual)
2. Create a new Anaconda environment using either of following commands:
>conda env create --name your_environment_name --file ./pkgs/pkgs.yml

or

>conda create --name your_environment_name --file ./pkgs/pkgs.txt

The installation time depends on internet speed but usually takes within an hour.

3. At last, you can clone this repository:
> git clone https://gitfront.io/r/user-4833002/1be179452bed1e48e1048c7fd71b4fd83293983c/DeepRF/

## Demo
We provide shell scripts and MATLAB scripts in the folder 'demo' 
for the demonstration of an RF pulse design using DeepRF.
After running these scripts, a slice-selective excitation pulse will be designed, 
and the analysis result will be displayed 
(see Fig. 2 and Supplementary Fig. 5 in the paper).\
To run the demo, first, activate an Anaconda environment and type:
> cd demo\
> ./1_exc_generation.sh

This shell script is to run the RF generation module (see METHODS in the paper). 
The execution time was less than 30 minutes per DRL run, 
and the total time was 23 hours.\
If the .sh file is not executable, use following command.
> chmod +x 1_exc_generation.sh

Second, run MATLAB script '2_exc_seed_rf.m' using MATLAB.\
Third, execute the other shell script using following command:
> ./3_exc_refinement.sh

This shell script is to run the RF refinement module (see METHODS). The execution time was roughly 17 hours. 
If available size of GPU memory is not enough, the execution reports out-of-memory error. Then, open the shell script and
modify the following line:
> python ../envs/refinement.py --tag "exc_refinement" --env "Exc-v51" --gpu "0" --samples 256 --preset "../logs/exc_generation/seed_rfs.mat"

Decrease the argument value of '--samples', for example, as 64. However, this may lead to degraded design result 
than the result shown in the paper.

Finally, to analyze the design result, run '4_exc_plot_result.m' using MATLAB.
You can see the pulse shapes and slice profiles of the DeepRF-designed pulse and corresponding SLR RF pulse.

## Instructions for your own RF pulse design
To design your own RF pulse using DeepRF, 

1. Define a reward function tailored for your purpose.
2. Make your customized [gym environment](https://gym.openai.com/) by modifying the Python scripts 
in the 'envs/deeprf'.
You can start the modification by copying and pasting the demo code that already exists in those scripts.
3. Run DeepRF using your new gym environment as demonstrated in the demo.

## Acknowledgement
DeepRF was implemented by modifying 
[the python code from Niraj Amalkanti.](https://github.com/namalkanti/bloch-simulator-python) 

## License
We provide software for academic research purpose only and NOT for commercial or clinical use.  
For commercial use of our software, contact us (snu.list.software@gmail.com) for licensing 
via Seoul National University.  
Please email to “snu.list.software@gmail.com” with the following information.  
  
Name:  
Affiliation:  
Software:  
  
When sending an email, an academic e-mail address (e.g. .edu, .ac.) is required.  

## Contact
Dongmyung Shin, Ph.D. candidate, Seoul National University.  
shinsae11@gmail.com  
http://list.snu.ac.kr
