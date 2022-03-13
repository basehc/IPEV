## IPEV: Identification of Prokaryotic and Eukaryotic Virus in virome data using deep learning

![0](./pic/logo.png)

## Contents

- [Introduction](#Introduction)

- [Installation](#Installation)
- [Quick start](#Quick start)
- [Citation](#Citation)
- [Contact](#Contact)
- [License](#License)



## Introduction

IPEV applied CNN to distinguish prokaryotic and eukaryotic Virus from virome data. It is built on **Python3.8.6** , **Tensorflow  2.3.1**. IPEV  calculates a set of scores that reflect the probability that the input sequence fragments are prokaryotic and eukaryotic viral sequences. By using parallelism and algorithmic optimization, IPEV get the results of the calculations very quickly.

## Installation

```
.
├── core/
│   ├── 1.hdf5
│   ├── 2.hdf5
│   ├── 3.hdf5
│   └── 4.hdf5
├── run.py             #main program
└── example.fasta      #example virus sequences
```

You can use (please create new environment to avoid unnecessary trouble bu using `conda create -n env_name python=3.6;    `

`     source activate my_env_name;   python3 -m venv /path/to/new/virtual/environment`): 

```python
python -m pip install -r requirement.txt
```

## Quick start


```
1.Put *fasta  file into current file path

2.python run.py 'your file'

```



![1](./pic/2.jpg)







## Citation

## Contact

If you have any questions, please don't hesitate to ask me: yinhengchuang@pku.edu.cn

## License

The source code of IPEV is distributed as open source under the [GNU GPL license](https://www.gnu.org/licenses/gpl-3.0.en.html) V3, the IPEV program and all datasets  also can be freely available at http://cqb.pku.edu.cn/ZhuLab/IPEV

