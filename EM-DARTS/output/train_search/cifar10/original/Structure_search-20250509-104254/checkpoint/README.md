# EM-DARTS: Preventing Performance Collapse in Differentiable Architecture Search with The Edge Mutation


<img src="./fig/Edges_Mutation.jpg" width="800px"/>

Our project references the following works: [NAS-Bench-201](https://github.com/D-X-Y/NAS-Bench-201/), [DARTS](https://github.com/quark0/darts/tree/master), and [$\Lambda$ -DARTS](https://github.com/dr-faustus/Lambda-DARTS).
## Requirements
```
Python >= 3.7
PyTorch >= 2.0
timm >= 0.9.16
pytorch_lightning >= 2.0
```
## Experiment
### Experiment on NAS-Bench-201 search space
**Architecture Search**
```
python train_search.py data --config configs/NAS_search.yaml  --p_max 0.2
```
**Architecture Evaluation**

We utilize the NAS-Bench-201 API provided by [NAS-Bench-201.md](https://github.com/D-X-Y/NAS-Bench-201/blob/8558547969c131f75af2725869ff1ece98e98f23/README.md), to evaluate the performance of the discovered architectures on the CIFAR-10, CIFAR-100, and ImageNet16-120 datasets.

**Table1:** Under the NAS-Bench-201 framework, the accuracy (%) results obtained from experiments on the CIFAR-10, CIFAR-100, and ImageNet16-120 datasets.
| Dataset      | Exp1   | Exp2   | Exp3   | Exp4   |
|--------------|------|------|------|------|
| **CIFAR-10** | 91.55   | 91.55 | 91.55  | 91.55  |
| **CIFAR-100**| 73.49  | 73.49  | 73.49  | 73.49 |
| **ImageNet16-120** | 46.37  | 46.37  | 46.37  | 46.37  |

### Experiment on DARTS search space
**Architecture Search**
```
python train_search.py  data --config configs/DARTS_search.yaml --p_max 0.125 --dataset cifar10
```
**Architecture Evaluation**

-  CIFAR-10
```
python train.py  data  --config configs/DARTS_cifar10.yaml --arch  exp2_[1/2/3/4]
```
-  CIFAR-100
```
python train.py  data  --config configs/DARTS_cifar100.yaml --arch  exp2_[1/2/3/4]
```
-  ImageNet
```
torchrun --nproc_per_node=4  train_imagenet.py --arch  exp2_4
```
**Table2:** Under the DARTS framework, the accuracy (%) results obtained from experiments on the CIFAR-10, CIFAR-100, and ImageNet datasets.
| Dataset      | Exp1  | Exp2  | Exp3  | Exp4  |
|--------------|-------|-------|-------|-------|
| **CIFAR-10** | 97.64 | 97.57 | 97.59 | 97.67 |
| **CIFAR-100**| 84.05 | 83.78 | 83.83 | 84.19 |
| **ImageNet** | -     | -     | -     | 76.2  |

### Experiment on Reduced search spaces
**Architecture Search**

```
python train_search.py data --config configs/DARTS_search.yaml --p_max {p_max_value} --search_space [s1/s2/s3/s4] --dataset [cifar10/cifar100/svhn]
```
**Table3:** In different datasets and search spaces, the value of the maximum mutation probability $p_{\max}$.
| Dataset      | s1   | s2   | s3   | s4   |
|--------------|------|------|------|------|
| **CIFAR-10** | 0.2  | 0.25 | 0.2  | 0.2  |
| **CIFAR-100**| 0.4  | 0.2  | 0.2  | 0.4  |
| **SVHN**     | 0.4  | 0.4  | 0.2  | 0.4  |

**Architecture Evaluation**
-  CIFAR-10
```
python train.py  data  --config configs/DARTS_cifar10.yaml --arch  exp3_cifar10_[s1/s2/s3/s4]
```
-  CIFAR-100
```
python train.py  data  --config configs/cifar100_eval.yaml --arch  exp3_cifar100_[s1/s2/s3/s4]
```
-  SVHN
```
python train.py  data  --config configs/svhn_eval.yaml --arch  exp3_svhn_[s1/s2/s3/s4]
```
**Table4:** Under the Reduced DARTS framework, the test error rate (%) results obtained from experiments on the CIFAR-10, CIFAR-100, and SVHN datasets.
| Dataset      | s1    | s2    | s3    | s4    |
|--------------|-------|-------|-------|-------|
| **CIFAR-10** | 2.74  | 2.39  | 2.44  | 2.32  |
| **CIFAR-100**| 23.27 | 21.47 | 20.33 | 20.35 |
| **SVHN**     | 2.49  | 2.34  | 2.30  | 2.27  |
### Experiment on Ablation study
**Architecture Search**
```
python train_search.py data --config configs/NAS_search.yaml  --p_max [0.1/0.2/0.3/0.4/0.5]  --epochs [50/400]  --grow_mode  [linear/exp/cosine/early/middle/late]
```
**Architecture Evaluation**

Utilize the NAS-Bench-201 API, to evaluate the performance of the discovered architectures on the CIFAR-10 dataset.

**Table5:** Under the NAS-Bench-201 framework, the accuracy (%) results obtained from experiments on the CIFAR-10 dataset with different strategies.
|  strategy | Exp1 | Exp2 | Exp3 | Exp4 | Mean|
|--------------|------|------|------|------|------|
| **DARTS** | 39.77 | 39.77 | 39.77 | 39.77 | 39.77 ± 0 |
| **p=0.1** | 90.40 | 90.57 | 91.5 | 91.12 |90.90 ± 0.44|
| **p=0.2** | 90.57 | 90.57 | 90.4 | 90.32 |90.47 ± 0.11|
| **early** | 91.12 | 90.12 | 89.81 | 90.5 | 90.39 ± 0.49|
| **middle** | 91.12 | 91.12 | 90.51 | 91.50 | 91.06 ± 0.35|
| **late** | 91.12 | 91.55 | 91.55 | 91.50 | 91.43 ± 0.18|
| **linear** | 91.55 | 91.55 | 91.55 | 91.55 | 91.55 ± 0|



**Table6:** Under the NAS-Bench-201 framework, the accuracy (%) results obtained from experiments on the CIFAR-10 dataset with different $p_{\max}$ values.
|  $p_{\max}$  | Exp1 | Exp2 | Exp3 | Exp4 |
|--------------|------|------|------|------|
| **0.1** | 91.12 | 91.55 | 91.12 | 91.12 |
| **0.2** | 91.55 | 91.55 | 91.55 | 91.55 |
| **0.3** | 90.57 | 91.55 | 90.51 | 91.55 |
| **0.4** | 90.57 | 90.40 | 91.55 | 90.24 |
| **0.5** | 90.57 | 90.24 | 90.57 | 90.40 |

**Table7:** Under the NAS-Bench-201 framework, the accuracy (%) results obtained from experiments on the CIFAR-10 dataset with different epochs values.
|  epochs | Exp1 | Exp2 | Exp3 | Exp4 |
|--------------|------|------|------|------|
| **50** | 91.55 | 91.55 | 91.55 | 91.55 |
| **100** | 91.55 | 91.55 | 91.55 | 91.55 |
| **200** | 91.50 | 91.55 | 90.51 | 91.55 |
| **300** | 91.34 | 91.42 | 91.50 | 91.50 |
| **400** | 90.79 | 90.88 | 90.79 | 90.88 |


**Table8:** Under the NAS-Bench-201 framework, the accuracy (%) results obtained from experiments on the CIFAR-10 dataset with different growth strategies.
|  strategy | Exp1 | Exp2 | Exp3 | Exp4 |
|--------------|------|------|------|------|
| **linear** | 91.55 | 91.55 | 91.55 | 91.55 |
| **exponential** | 91.53 | 91.50 | 90.50 | 91.55 |
| **cosine** | 91.55 | 91.55 | 91.50 | 91.50 |

### Experiment on S5
**Architecture Search**
```
python train_search.py  data --config configs/DARTS_search.yaml --p_max 0.125 --search_space s5 --dataset cifar10
```
**Architecture Evaluation**
```
python train.py  data  --config configs/DARTS_cifar10.yaml --arch  [emdarts/darts/random][1/2/3/4] 
```
**Table9:** Under S5, the accuracy (%) results obtained from experiments on the CIFAR-10 datasets.
|  strategy | Exp1 | Exp2                | Exp3         | Exp4 |
|--------------|------|---------------------|--------------|------|
| **Random Sampling** | 96.9 | 97.09 | 96.85 | 97.11 |
| **DARTS(1st)** | 97.33 | 97.36 | 97.37 | 97.11 |
| **EM-DARTS** | 97.44 | 97.35 | 97.49 | 97.40|
All experiment records can be found in the **results** folder. 

