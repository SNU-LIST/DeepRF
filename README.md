# DeepRF

DeepRF is an AI-powered RF pulse design framework which utilizes the self-learning characteristics 
of deep reinforcement learning (DRL) to generate a novel RF pulse. \
For more details, see [preprint.](https://arxiv.org/abs/2105.03061)

## System requirements

### Hardware
At least one NVIDIA GPU is needed, which supporting CUDA 10.2.\
Computing environment in which we tested the code is as follows.\
CPU: Intel Xeon Gold 5218 2.30 GHz\
GPU: NVIDIA Quadro RTX 8000\
RAM: 128 GB

### Software
Ubuntu 18.04 LTS\
MATLAB 2019a\
Anaconda + Python packages:\
instead of listing all Python packages to run DeepRF, we provide .yml or .txt file that can be used to 
create an Anaconda environment based on exact package versions. See below.

## Installation guide
To install all Python packages to run DeepRF,
0. Install appropriate versions of NVIDIA driver, CUDA, and cuDNN for your GPU
1. Download and install [Anaconda](https://www.anaconda.com/products/individual)
2. Create a new Anaconda environment using either of following commands:
>conda env create --name your_environment_name --file pkgs.yml

or

>conda create --name your_environment_name --file pkgs.txt

The pkgs.yml and pkgs.txt files are in the folder 'pkgs'. The installation time depends on internet speed 
but usually takes within an hour.\
After that, you can clone this repository:
> git clone https://github.com/SNU-LIST/DeepRF

or download ZIP file of this repository and unzip it.

## Demo
Here, we provide shell scripts and MATLAB scripts in the folder 'demo' for a quick demo of DeepRF.\
After running these scripts, a slice-selective excitation pulse will be designed, 
and the analysis result will be displayed 
(see Fig. 2 and Supplementary Fig. 5 in the paper).\
To run the demo, first, activate an Anaconda environment and type:
> cd demo\
> ./1_exc_generation.sh

The execution time was roughly 30 minutes per DRL run.\
If the .sh file is not executable, use following command.
> chmod +x 1_exc_generation.sh

Second, run MATLAB script '2_exc_seed_rf.m' using MATLAB.\
Third, execute the other shell script using following command:
> ./3_exc_refinement.sh

If available size of GPU memory is not enough, the execution reports out-of-memory error. Then, open '3_exc_refinement.sh' and
modify the following line:
> python ../envs/refinement.py --tag "exc_refinement" --env "Exc-v51" --gpu "0" --samples 256 --preset "../logs/exc_generation/seed_rfs.mat"

Decrease the argument value of '--samples', for example, as 128. However, this may lead to degraded design result 
than the result shown in the paper.

Finally, to analyze the final design, run '4_exc_plot_result.m' using MATLAB.
You can see the pulse shapes and slice profiles of the final DeepRF pulse and SLR pulse.

## Instructions for personal RF design
To design your own RF pulse using DeepRF, first, you need to define a reward function tailored for your purpose.\
Then, make your customized [gym environment](https://gym.openai.com/) by modifying the python scripts 
in the 'envs/deeprf'.\
You can start the modification by copying and pasting the demo code that already exists in the scripts.\
After that, run DeepRF using your new gym environment as similar as the demo above.

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