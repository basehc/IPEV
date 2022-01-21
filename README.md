# IPEV

####   identification of prokaryotic and eukaryotic virus in virome data based on deep learning

![0](./pic/logo_1.png)

- [Introduction]: 

  

- [Version](https://github.com/shufangwu/DeePhage#version)

- [Requirements](https://github.com/shufangwu/DeePhage#requirements)

- [Installation](https://github.com/shufangwu/DeePhage#installation)

- [Usage](https://github.com/shufangwu/DeePhage#usage)

- [Output](https://github.com/shufangwu/DeePhage#output)

- [Citation](https://github.com/shufangwu/DeePhage#citation)

- [Contact](https://github.com/shufangwu/DeePhage#contact)



#### Installation

#### Dependencies



You can use (please create new environment to avoid unnecessary trouble bu using `conda create -n env_name python=3.6;         source activate my_env_name;   python3 -m venv /path/to/new/virtual/environment`): 

```python
python -m pip install -r requirement.txt
```

#### 2.Usage


```
1.Put *fasta  file into current file path

2.python run.py 'your file'

3.
```



![1](./pic/2.jpg)

#### 3.Permance

The ROC curves and AUC scores of IPEV performances in each set of five-fold crossvalidation



![2](./pic/1.jpg)

Figure1:The ROC curves and AUC scores of IPEV performances in each set of five-fold crossvalidation. (Group A refers to 100-400bp length, Group B refers to 400-800bp length, Group C refers to 800-1200bp length, Group D refers to 1200-1800bp length)



**Accuracy** :AUC of IPEV can arrive to 0.99 on 5-fold crossvalidation when length of sequence is between 1200-1800..



**Run time** :  .



#### 4.Contact

If you have any questions, please don't hesitate to ask me: yinhengchuang@pku.edu.cn

#### 5. License

The source code of IPEV is distributed as open source under the [GNU GPL license](https://www.gnu.org/licenses/gpl-3.0.en.html), the IPEV program and all datasets  also can be freely available at http://cqb.pku.edu.cn/ZhuLab/IPEV

