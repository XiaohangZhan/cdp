# Implementation of "Consensus-Driven Propagation in Massive Unlabeled Data for Face Recognition" (CDP)

### Notice
<span style="color:red">Code is in continuous updating, please pull before execution.</span>

### Paper

Xiaohang Zhan, Ziwei Liu, Junjie Yan, Dahua Lin, Chen Change Loy, ["Consensus-Driven Propagation in Massive Unlabeled Data for Face Recognition"](http://openaccess.thecvf.com/content_ECCV_2018/papers/Xiaohang_Zhan_Consensus-Driven_Propagation_in_ECCV_2018_paper.pdf), ECCV 2018

Project Page:
[link](http://mmlab.ie.cuhk.edu.hk/projects/CDP/)

### Dependency
Please use Python3, as we cannot guarantee its compatibility with python2. The version of PyTorch we use is 0.3.1. Other depencencies:

    ```
    pip install nmslib
    ```

### Usage
0. Clone the repo.

    ```shell
    git clone git@github.com:XiaohangZhan/cdp.git
    cd cdp
    ```

#### Using ready-made data

1. Download the data [here](https://drive.google.com/open?id=1Fs8oN1JiGJRC93TkfDV-PaCeXQz-Htea) to the repo root, and uncompress it.

    ```shell
    tar -xf data.tar.gz
    ```

2. Make sure the structure looks like the following:

    ```shell
    cdp/data/
    cdp/data/labeled/emore_l200k/
    cdp/data/unlabeled/emore_u200k/
    # ... other directories and files ...
    ```

3. Run CDP

    * Single model case:

        ```shell
        python -u main.py --config experiments/emore_u200k_single/config.yaml
        ```

    * Multi-model voting case (committee size: 4):

        ```shell
        python -u main.py --config experiments/emore_u200k_cmt4/config.yaml
        ```

    * Multi-model mediator case (committee size: 4):

        ```shell
        # edit `experiments/emore_u200k_cmt4/config.yaml` as following:
        # strategy: mediator
        python -u main.py --config experiments/emore_u200k_cmt4/config.yaml
        ```

4. Collect the results

    Take `Multi-model mediator case` for example, the results are stored in `experiments/emore_u200k_cmt4/output/k15_mediator_111_th0.9915/sz600_step0.05/meta.txt`. The order is the same as that in `data/unlabeled/emore_u200k/list.txt`. The samples labeled as `-1` are discarded by CDP. You may assign them with new unique labels if you must use them.

#### Using your own data

1. Create your data directory, e.g. `mydata`

    ```shell
    mkdir data/unlabeled/mydata
    ```

2. Prepare your data list as `list.txt` and copy it to the directory.

3. (optional) If you want to evaluate the performance on your data, prepare the meta file as `meta.txt` and copy it to the directory.

4. Prepare your feature files. Extract face features corresponding to the `list.txt` with your trained face models, and save it as binary files via `feature.tofile("xxx.bin")` in numpy. Finally link/copy them to `data/unlabeled/mydata/features/`. We recommand renaming the feature files using model names, e.g., `resnet18.bin`. CDP works for single model case, but we recommend you to use multiple models (i.e., preparing multiple feature files extracted from different models) with `mediator` for better results.

5. The structure should look like:

    ```shell
    cdp/data/unlabeled/mydata/
    cdp/data/unlabeled/mydata/list.txt
    cdp/data/unlabeled/mydata/meta.txt (optional)
    cdp/data/unlabeled/mydata/features/
    cdp/data/unlabeled/mydata/features/*.bin
    ```

    (You do not need to prepare knn files.)

6. Prepare the config file. Please refer to the examples in `experiments/`

    ```shell
    mkdir experiments/myexp
    cp experiments/emore_u200k_cmt4/config.yaml experiments/myexp/
    # edit experiments/myexp/config.yaml to fit your case.
    # you may need to change `base`, `committee`, `data_name`, etc.
    ```

7. Tips for paramters adjusting
    * Modify `threshold` to obtain roughly closed `precision` and `recall` to achieve higher `fscore`.
    * Higher threshold results in higher precision and lower recall.
    * Larger `max_sz` results in lower precision and higher recall.

### Evaluation Results

* data: emore_u200k, images number: 200K, identity number: 2577 (original annotation)

* KNN using nmslib

| k  | strategy | committee | setting         | prec, recall, fscore | knn time | cluster time | total time |
|----|----------|-----------|-----------------|----------------------|----------|--------------|------------|
| 15 | vote     |     0     | accept0_th0.66  | 89.35, 88.98, 89.16  |   14.8s  |     7.7s     |    22.5s   |
| 15 | vote     |     4     | accept4_th0.605 | 93.36, 92.91, 93.13  |   78.7s  |     6.0s     |    84.7s   |
| 15 | mediator |     4     | 110_th0.9938    | 94.06, 92.45, 93.25  |   78.7s  |     77.7s    |   156.4s   |
| 15 | mediator |     4     | 111_th0.9925    | 96.66, 94.93, 95.79  |   78.7s  |    137.8s    |   216.5s   |

* KNN using sklearn

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
