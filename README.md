# Implementation of "Consensus-Driven Propagation in Massive Unlabeled Data for Face Recognition" (CDP)

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
1. Prepare your data list. If you want to evaluate the performance of CDP, copy the meta file as well. The example of `list.txt` and `meta.txt` can be found in `data/example_data/`.
```
mkdir data/your_data
cp /somewhere/list_file data/your_data/list.txt
cp /somewhere/meta_file data/your_data/meta.txt # optional
```
2. Prepare your feature files. Extract face features corresponding to the `list.txt` with your trained face models, and save it as binary files via `feature.tofile("xxx.bin")` in numpy. Finally link them to `data/data_name/features/model_name.bin`.
```
mkdir data/data_name/features
ln -s /somewhere/feature.bin data/your_data/features/resnet18.bin # for example
```
Although CDP can handle single-model case, we recommend more than one models to obtain better performance.

3. Prepare the config file. Please refer to the examples in `experiments/`

### Usage
```
python -u main.py --config experiments/example_vote/config.yaml
```
or
```
python -u main.py --config experiments/example_mediator/config.yaml
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
