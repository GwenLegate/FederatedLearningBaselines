# Federated Learning Baselines

This repository implements baseline algorithms for Federated Learning. 

Currently, available baselines are:
* [FedAvg](https://arxiv.org/pdf/1602.05629.pdf) 
* [FedAvgM](https://arxiv.org/pdf/1909.06335.pdf)
* [FedADAM](https://arxiv.org/pdf/2003.00295.pdf)

All baselines can be used in combination with [WSM](https://proceedings.mlr.press/v232/legate23a/legate23a.pdf) by setting `--wsm=1`

Baselines are selected using the `--fed_type` option. The default is FedAvg.

### Data Heterogeneity

Label skew and quantity skew are controlled independently.

**Label skew** — which classes a client holds:

| Option | Partition |
|--------|-----------|
| `--iid=1` | every client sees the same class distribution |
| `--dirichlet=1 --alpha=<a>` | each client draws a multinomial over classes from a Dirichlet, per [Hsu et. al.](https://arxiv.org/abs/1909.06335). Lower `alpha` is more heterogeneous. This is the default (`alpha=0.1`) |
| `--dirichlet=0` | shards, per [McMahan et. al.](https://arxiv.org/pdf/1602.05629.pdf) |

**Quantity skew** — how *much* data a client holds. Off by default, so every client gets the same number of
samples. Setting `--quantity_skew=1` draws client sizes as `q ~ Dir(quantity_beta)` and gives client `j` a `q_j`
share of the dataset, the quantity skew of [NIID-Bench (Li et. al., ICDE 2022)](https://arxiv.org/abs/2102.02079).
Lower `quantity_beta` is more skewed; `--quantity_min_samples` (default 10) floors each client.

```
# NIID-Bench 'iid-diff-quantity': same label distribution, unequal amounts of data
federated_main.py --iid=1 --quantity_skew=1 --quantity_beta=0.5

# both skews at once: heterogeneous labels AND unequal amounts of data
federated_main.py --dirichlet=1 --alpha=0.1 --quantity_skew=1 --quantity_beta=0.5
```

Two caveats worth knowing:

* Composing quantity skew with `--dirichlet=1` means `alpha` and `quantity_beta` interact, so the partition is
  no longer the one described in Hsu et. al. Neither paper evaluates this combination.
* NIID-Bench enforces its minimum client size by rejecting and redrawing the Dirichlet. That works for the ~10
  parties it evaluates but effectively never terminates at the client counts used here (at 450 clients over
  CIFAR-10, a `Dir(0.5)` draw clears a floor of 10 with probability ~0). Instead each client is given
  `quantity_min_samples` up front and the Dirichlet shares out the remainder, which compresses the skew slightly
  but always terminates.

Because shards are a fixed size, `--quantity_skew=1` with `--dirichlet=0` skews the *number of shards* per
client, so sizes are quantised to a multiple of the shard size.

### Aggregation

Client updates are aggregated weighted by each client's share of the round's training samples (`n_k/n`), as
specified by [FedAvg](https://arxiv.org/pdf/1602.05629.pdf). Every partition above gives clients equal amounts of
data unless `--quantity_skew=1`, and with equal `n_k` the weighting reduces to the straight mean, so the results
below are unaffected. (The average is accumulated in float64 and cast back, so it agrees with a float32 straight
mean to about 1e-7 rather than bit-exactly.)

### Hyper-parameters and Results

Experiemntal conditions match the federated hyper-parameters provided in [Adaptive Federated Optimization](https://arxiv.org/pdf/2003.00295.pdf) for the 
Cifar-10 dataset. Learning rates were tuned by a grid search and each reported accuracy was obtained over four seeds.
* 4000 global rounds
* 500 training samples per client
* 1 local epoch
* batch size of 20
* Group normalization as per [Hsieh et. al.](http://proceedings.mlr.press/v119/hsieh20a.html).

Note: While the code has the option to use batch norm as well, training using batch norm and FedAvgM was found to be unstable and does not converge.

|Baseline  | Accuracy avg +/- std | Run Command to Obtain Results in Prev Column                                                                                                                                                               |
|----------|----------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| FedAvg   | 72.8 +/- 0.5         | `federated_main.py --epochs=4000 --client_lr=0.03 --num_clients=450 --frac=0.023 --local_ep=1 --local_bs=20 --num_workers=2`                                                                               |
| FedAvgM  | 83.5 +/- 0.7         | `federated_main.py --fed_type=fedavgm --epochs=4000 --client_lr=0.003 --num_clients=450 --frac=0.023 --momentum=0.9 --local_ep=1 --local_bs=20 --num_workers=2`                                            |
| FedADAM  | 85.0 +/- 0.6         | `federated_main.py --epochs=4000 --fed_type=fedadam --server_lr=0.001 --beta1=0.9 --beta2=0.999 --adam_eps=1e-8 --client_lr=0.1 --num_clients=450 --frac=0.023 --local_ep=1 --local_bs=20 --num_workers=2` | 

sample training curves and their final accuracies for each of the three baselines currently implemented
![alt text](https://github.com/GwenLegate/FederatedLearningBaselines/blob/master/figs/curves.png?raw=true)

