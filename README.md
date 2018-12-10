# Implementation of "Consensus-Driven Propagation in Massive Unlabeled Data for Face Recognition" (CDP)

### Notice
<span style="color:red">Code is in continuous updating, please pull before execution.</span>

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

3. Prepare the config file. Please refer to the examples in `experiments/` or directly use `experiments/emore_u200k_vote*/config.yaml`

### Usage
Single model case (using ready-made data):
```
python -u main.py --config experiments/emore_u200k_vote0/config.yaml
```
Multi-model voting case (using ready-made data):
```
python -u main.py --config experiments/emore_u200k_vote4/config.yaml
```
Multi-model mediator case (using ready-made data):
```
python -u main.py --config experiments/emore_u200k_mediator/config.yaml
```
The results are stored in `experiments/*/output/*/sz*_step*/`, including: `meta.txt` and `pred.npy`.

### Evaluation Results
* data: emore_u200k

| k  | strategy | committee | optimal setting                | prec, recall, fscore |
|----|----------|-----------|--------------------------------|----------------------|
| 15 | vote     |     0     | accept0_th0.66/sz600_step0.05  | 89.35, 88.98, 89.16  |
| 15 | vote     |     4     | accept4_th0.605/sz600_step0.05 | 92.87, 92.91, 92.89  |
| 15 | mediator |     4     | 110_th0.9922/sz600_step0.05    | 94.45, 92.56, 93.49  |
| 15 | mediator |     4     | 111_th0.9915/sz600_step0.05    | 96.46, 95.20, 95.83  |
| 20 | vote     |     0     | accept0_th0.665/sz600_step0.05 | 89.88, 88.16, 89.01  |
| 20 | vote     |     4     | accept4_th0.625/sz600_step0.05 | 92.23, 92.90, 92.56  |
| 20 | mediator |     4     | 110_th0.9915/sz600_step0.05    | 92.97, 92.82, 92.90  |
| 20 | mediator |     4     | 111_th0.9915/sz600_step0.05    | 96.30, 95.65, 95.97  |

note: for mediator, `110` means using `relationship` and `affinity`; `111` means using `relationship`, `affinity` and `structure`.

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
