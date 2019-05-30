# On the Power of Curriculum Learning in Training Deep Networks

Code implementing the basic results ofthe paper [On the Power of Curriculum Learning in Training Deep Networks](https://arxiv.org/abs/1904.03626).
This project demonstrate how to train simple keras model using curriculum by transfer, and fixed exponential pacing, reproducing the results depicted in the paper.

## Getting Started
### Prerequisites

All prerequiesites are listed in the requirements.txt, and can be installed by running:

```
pip3 install -r requirements.txt
```

The project assumes python 3.5 or higher.

## Running the expriments

the file main_train_networks.py controls the entire pipeline of the project. It can train the model used in the paper, on various test cases, using the following flags:

* --dataset - any superclass of CIFAR-100. Options are: cifar100_subset_0/cifar100_subset_1/.../cifar100_subset_20. defualt is cifar100_subset_16, which is the "small-mammals" dataset which was used in the paper
* --curriculum - Which test case (as defined in the paper) to use. Can choose from: curriculum, vanilla, random and anti (corresponding to anti-curriculum)
* --output_path - location to save the output
* --repeats - number of times to repeat the experiment
* --batch_size - detemine the batchsize
* --num_epochs - number of epochs to train the model

### learning rate parameters:

* --learning_rate - initial learning
* --lr_decay_rate - factor by which we drop learning rate exponentially
* --minimal_lr - min learning rate we allow
* --lr_batch_size - interval of batches in which we drop the learning rate   

### curriculum parameters:

* --batch_increase - interval of batches to increase the amount of data we sample from
* --increase_amount - factor by which we increase the amount of data we sample from
* --starting_percent - percent of data to sample from in the inital batch
* --order - determine the network from which we do transfer learning. options: inception, vgg16, vgg19, xception, resnet

An example of running each test case, including the resulting graphs, can be seen by running: main_reproduce_paper.py


## Authors

* **Guy Hacohen** - *Initial work* - [GuyHacohen](https://github.com/GuyHacohen)

## License

This project is licensed under the GNU general public License - see the [LICENSE.md](LICENSE.md) file for details
