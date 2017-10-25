# PyTorch

## Using group GPUs

We have a number of group machines with GPUs for deep learning: Myrtle, Ned, Athena and Minerva. 

Today we will use Athena and Minerva which each have 4 GPUs. To connect, use ssh on the vlc shared account:

```
$ ssh vlc@athena.ecs
```

You will need to be connected to the VPN or on a machine which is physically connected.

To connect using windows, use Putty.

You can use the command:

```
tpl1g12@minerva:~$ nvidia-smi
```

to view current GPU usage and htop to view cpu and other resource usage.

When using the vlc account, please make a directory in home for your code etc. 

To run code, decide which GPU to use and run any python program like this:

```
$ CUDA_VISIBLE_DEVICES=X python train.py 
```

## Fundamentals + Autograd Basics
yan
regression example

tensors etc
numpy equivalent
gpu

## NN Overview
tom
cifar10, connecting, set up

## Creating New Modules
tom + yan
replace Linear
### Randomly Drop Layers


## Inception
tom
alex too fill in gaps
easier in pytorch

## Ensembles
tom
maybe if it works
module == layer
mode(inc,vgg,alex)
freeze layers

## Creating Optimizer
yan
sgd copy + something


## Adding Noise to Gradients
both maybe never
maybe