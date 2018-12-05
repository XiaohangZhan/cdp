# Implementation of "Consensus-Driven Propagation in Massive Unlabeled Data for Face Recognition" (CDP)

### Notice
<span style="color:red">Modules for "Mediator" will be released after CVPR deadline.</span>

<span style="color:red">A multi-task face recognition framework based on PyTorch will be released after CVPR deadline.</span>

### Paper

Xiaohang Zhan, Ziwei Liu, Junjie Yan, Dahua Lin, Chen Change Loy, ["Consensus-Driven Propagation in Massive Unlabeled Data for Face Recognition"](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xiaohang_Zhan_Consensus-Driven_Propagation_in_ECCV_2018_paper.pdf), ECCV 2018

Project Page:
[link](http://mmlab.ie.cuhk.edu.hk/projects/CDP/)

### Dependency
Please use Python3, as we cannot guarantee its compatibility with python2. The version of PyTorch we use is 0.3.1. Other depencencies:
```
pip install nmslib
```

### Before Start
1. Prepare your data list. If you want to evaluate the performance of CDP, copy the meta file as well. You can also download the ready-made data [here](https://drive.google.com/open?id=1Pke9zLf8f4TCzurp7DA17MhJbE5UD5Zl) to the repo root and unzip it.
```
unzip data.zip
```
2. Prepare your feature files. Extract face features corresponding to the `list.txt` with your trained face models, and save it as binary files via `feature.tofile("xxx.bin")` in numpy. Finally link them to `data/unlabeled/data_name/features/model_name.bin`. Besides, you can also use the ready-made features in `data/unlabeled/emore_u200k`.

Although CDP can handle single-model case, we recommend more than one models to obtain better performance.

3. Prepare the config file. Please refer to the examples in `experiments/`

### Usage
Single model case (using ready-made data):
```
python -u main.py --config experiments/emore_u200k_vote0/config.yaml
```
Multi-model voting case (using ready-made data):
```
python -u main.py --config experiments/emore_u200k_vote4/config.yaml
```

### Bibtex
```
@inproceedings{zhan2018consensus,
  title={Consensus-Driven Propagation in Massive Unlabeled Data for Face Recognition},
  author={Zhan, Xiaohang and Liu, Ziwei and Yan, Junjie and Lin, Dahua and Change Loy, Chen},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  pages={568--583},
  year={2018}
}
```
