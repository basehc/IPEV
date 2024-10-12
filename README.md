 ![Build](https://img.shields.io/badge/Build-passing-brightgreen) ![TensorFlow](https://img.shields.io/badge/TensorFlow-V2.3.1-brightgreen) ![license](https://img.shields.io/badge/license-GPL--v3-blue) 

## IPEV: Identification of Prokaryotic and Eukaryotic Virus in virome data using deep learning

  ![0](./pic/logo.png)

## Contents
- [Citation](#Citation)
  
- [Introduction](#Introduction)

- [Installation](#Installation)

- [Quickstart](#Quickstart)

- [Contact](#Contact)

- [License](#License)

## Citation

Hengchuang Yin, Shufang Wu, Jie Tan, Qian Guo, Mo Li, Jinyuan Guo, Yaqi Wang, Xiaoqing Jiang, and Huaiqiu Zhu. "IPEV: Identification of Prokaryotic and Eukaryotic Virus-Derived Sequences in Virome Using Deep Learning." GigaScience 13 (2024): giae018. https://doi.org/10.1093/gigascience/giae018.



## Introduction

IPEV applied CNN to distinguish prokaryotic and eukaryotic Virus from virome data. It is built on **Python3.8.6** , **Tensorflow  2.3.1**. IPEV calculates a set of scores that reflect the probability that the input sequence fragments are prokaryotic and eukaryotic viral sequences. By using parallelism and algorithmic optimization, IPEV gets the results of the calculations very quickly.

## Installation

File tree:

```
.
├── core/
│   ├── 1.hdf5
│   ├── 2.hdf5
│   ├── 3.hdf5
│   ├── 4.hdf5
│   └── decontamination/
│       ├── 1.hdf5
│       ├── 2.hdf5
│       ├── 3.hdf5
│       └── 4.hdf5
├── run.py             # Main program
└── example.fasta      # Example virus sequences

```

We provide two ways to use the IPEV tool: use repository from GitHub or use image from Docker Hub.



## 

###### How to use IPEV  from GitHub

- **Clone the Program**: First, clone the IPEV program repository from GitHub using the command:
  
  ```bash
  git clone https://github.com/basehc/IPEV.git
  ```

- **Create a New Environment**: To avoid potential conflicts with existing packages, it's recommended to create a new environment. You can do this with Conda and virtualenv:
  
  - Using Conda:
    
    ```bash
    conda create -n your_env_name python=3.8.6
    conda activate your_env_name
    ```
  - Using virtualenv:
    
    ```bash
    python3 -m venv /path/to/new/virtual/environment
    source /path/to/new/virtual/environment/bin/activate
    ```

- **Set Up the Operating Environment**: Install the necessary dependencies by navigating to the program directory and running:
  
  ```bash
  python -m pip install -r requirements.txt
  ```
  
  



###### How to use IPEV from  Docker Hub

1. Pull the dryinhc/ipev_v1 image from Docker Hub. Open a terminal window and run the following command:
   
   `docker pull dryinhc/ipev_v1`
   
   This will download the image to your local machine.

2. Run the dryinhc/ipev_v1 image. In the same terminal window, run the following command:
   
   `docker run -it --rm dryinhc/ipev_v1`
   
   This will start a container based on the image and run the IPEV tool. 
   
   Firstly, you need to  run `docker cp data.fasta dryinhc/ipev_v1:/app/tool/`in new terminal  , run `cd tool` in  container and `python run.py data.fasta`

3. To exit the container, press Ctrl+D or type `exit`.

## 

**Non-Virus Removal Feature in IPEV Program**:

The IPEV program offers a specialized function for filtering out non-viral components from virome datasets. This feature is easily accessible and can be utilized with the following command:

```bash
python3 run.py example.fasta -filter yes (default: no)
```

## Quickstart

```
1.cd ./IPEV

2.python run.py example.fasta
```

###### Program Output and Recommendations

- **Sequence Scoring File**: The final scores for the sequences will be stored in a TSV (Tab-Separated Values) file. This file is placed in a folder named with the current date and time. The TSV file includes scores for each sequence from your FASTA file, structured as follows:
  
  | Sequence_ID | Prokaryotic_Virus_Score | Eukaryotic_Virus_Score | Virus_Taxon |
  | ----------- | ----------------------- | ---------------------- | ----------- |
  | Sample_ID   | Score1                  | Score2                 | Category    |

- **Histogram of Scores**: Along with the TSV file, the program creates a histogram showing how often different scores occur for the sequences in your FASTA file. This chart makes it easier to see the range and frequency of the scores.

- **Efficient Running Time**: To save time, it's best to put all your sequences in one single FASTA file. This way, the program runs faster as it has fewer files to open and process.


## Contact

If you have any questions, please don't hesitate to ask me: yinhengchuang@pku.edu.cn or hqzhu@pku.edu.cn

## License

The source code of IPEV is distributed as open source under the [GNU GPL v3](https://www.gnu.org/licenses/gpl-3.0.en.html) , the IPEV program and all datasets  also can be freely available at  [zhulab homepage](https://cqb.pku.edu.cn/zhulab/info/1006/1156.htm)
