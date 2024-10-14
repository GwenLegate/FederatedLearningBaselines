# Federated Learning Baselines

This repository implements baseline algorithms for Federated Learning. 

Currently, available baselines are:
* [FedAvg](https://arxiv.org/pdf/1602.05629.pdf) 
* [FedAvgM](https://arxiv.org/pdf/1909.06335.pdf)
* [FedADAM](https://arxiv.org/pdf/2003.00295.pdf)

Baselines are selected using the `--fed_type` option. The default is FedAvg

### Hyper-parameters and Results

Experiemntal conditions match the federated hyper-parameters provided in [Adaptive Federated Optimization](https://arxiv.org/pdf/2003.00295.pdf) for the 
Cifar-10 dataset. Learning rates were tuned by a grid search and each reported accuracy was obtained over four seeds.
* 4000 global rounds
* 500 training samples per client
* 1 local epoch
* batch size of 20
* Group normalization as per [Hsieh et. al.](http://proceedings.mlr.press/v119/hsieh20a.html).

Note: While the code has the option to use batch norm as well, training using batch norm and FedAvgM was found to be unstable and does not converge.

|Baseline  | Accuracy avg +/- std | Run Command to Obtain Results in Prev Column                                                                                                                                                                 |
|----------|----------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FedAvg   | 72.8 +/- 0.5         | `federated_main.py --epochs=4000 --client_lr=0.03 --num_clients=450 --frac=0.023 --local_ep=1 --local_bs=20 --num_workers=2`                                                                                 |
| FedAvgM  | 83.5 +/- 0.7         | `federated_main.py --fed_type=fedavgm --epochs=4000 --client_lr=0.003 --num_clients=450 --frac=0.023 --momentum=0.9 --local_ep=1 --local_bs=20 --num_workers=2`                                              |
| FedADAM  | 85.0 +/- 0.6         | `federated_main.py --epochs=4000 --fed_type=fedadam --global_lr=0.0001 --beta1=0.9 --beta2=0.999 --adam_eps=1e-8 --client_lr=0.01 --num_clients=450 --frac=0.023 --local_ep=1 --local_bs=20 --num_workers=2` | 

sample training curves and their final accuracies for each of the three baselines currently implemented
![alt text](https://github.com/GwenLegate/FederatedLearningBaselines/blob/master/figs/curves.png?raw=true)

