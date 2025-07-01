# MGCP-CL
Code for *[Multi-Modal Graph-Based Sentiment Analysis via Hybrid Contrastive Learning](https://ieeexplore.ieee.org/abstract/document/10729439)*
![model](./pipeline-img/MGCP-CL.png)

## Environment
Python==3.11

Pytorch==2.2.0

cuda==12.1

Transformer==4.37.2

More dependencies can follow the [requirements.txt](./requirements.txt).

## Dataset
NGCP-CL is evaluated on MOSI and MOSEI datasets.

First you need to create necessary folds for the datasets.

* `./data/`
* `./save/`
* `./save/mosi`
* `./save/mosei`

After the preparation you can get the data from links.

| Dataset | Link                                                        |
| ------- | ------------------------------------------------------------ |
| MOSI    | *[GoogleDrive](https://drive.google.com/file/d/172iNTfiJq4ChN8XyrwIW6NFHVgtYgOVt/view?usp=sharing)* |
| MOSEI   | *[GoogleDrive](https://drive.google.com/file/d/119n_beAYaMImWmNNF7vstK3ckxhRgrvc/view?usp=sharing)* |

Download the data of MOSI and MOSEI and place them to `./data/`.

## Train and Test
Take MOSI as the example, run the `./run.py` with the following configuration.

```bash

--dataset
mosi
--batch_size
24
--max_len
128
--embed_type
bert_word
--seeds
24
--do_train
--save_path
./save/mosi/
--device_ids
0
--epoch
5
--lr_bert
1e-5
--lr_other
5e-4
--weight_decay_bert
1e-5
--weight_decay_other
5e-4
--hidden_size
128
--num_lstm_layers
1
--num_gnn_layers
2
--num_gnn_heads
1
--dropout
0.1
--dropout_gnn
0.1
--aug_ratio
0.1
--sup_cl_weight
0.1
--self_cl_weight
0.1
--nce_weight
1
--clustering_rate
0.3
--knn_neighbors
4
```
Of course you can change and fine-tune the hyperparameters for more explorations.
