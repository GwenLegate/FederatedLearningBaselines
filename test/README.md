# Federated Learning Baselines

This repository implements baseline algorithms for Federated Learning. 

Currently, available baselines are:
* [FedAvg](https://arxiv.org/pdf/1602.05629.pdf) 
* [FedAvgM](https://arxiv.org/pdf/1909.06335.pdf)
* [FedADAM](https://arxiv.org/pdf/2003.00295.pdf)

Baselines are selected using the `--fed_type` option. The default is FedAvg

### Hyper-parameters and Results

Experiemnts were conducted matching as closely as possible to the federated hyper-parameters provided in
[Adaptive Federated Optimization](https://arxiv.org/pdf/2003.00295.pdf) using the Cifar-10 dataset.
* 4000 global rounds
* 500 training samples per client
* 1 local epoch
* batch size of 20

|Baseline  |Accuracy (avg) +/- std  |Run Command to Obtain Results in Prev Column |
|----------|------------------------|---------------------------------------------|
| FedAvg   |73.8 +/- 1.3            |`federated_main.py --epochs=4000 --client_lr=0.0316 --num_clients=450 --frac=0.023 --local_ep=1 --local_bs=20 --num_workers=16`                                           |
| FedAvgM  |72.6 +/- 1.9            |`federated_main.py --epochs=4000 --client_lr=0.0316 --num_clients=450 --frac=0.023 --momentum=0.95 --local_ep=1 --local_bs=20 --num_workers=16`                                             |
| FedADAM  |78.0 +/- 2.2            | `federated_main.py --epochs=4000 --fed_type=fedadam --global_lr=0.0316 --beta1=0.9 --beta2=0.999 --adam_eps=0.01 --client_lr=0.01 --num_clients=450 --frac=0.023 --local_ep=1 --local_bs=20 --num_workers=16`                                            |

training curves for each of the three baselines currently implemented
![alt text](https://github.com/GwenLegate/FederatedLearningBaseline/blob/main/figs/curves.png?raw=true)

