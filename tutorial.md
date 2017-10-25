# PyTorch

## Fundamentals + Autograd Basics
At the lowest level of abstraction, PyTorch is just a tensor library.
You can do things just like you can in Matlab or with the Python `numpy` library.
We will start with a simple logistic regression example.
In this example, we will do most things manually so that everything is explicit and you know what goes on under the hood.
In later examples, there will be higher-level abstractions that make life easier than this.

Note: You should keep the documentation at http://pytorch.org/docs/master/torch.html open and make sure that you understand what each function call to the `torch` module is doing.

The task is to classify whether a sample came from distribution 1 (target label 0) or distribution 2 (target label 1).
Both of these distributions are simple bivariate Gaussians with different means and identity covariance matrix.
First, let's generate some data.

```
import torch


def generate_data(n):  # n is the number of samples to generate from each distribution
    mean1 = torch.Tensor([0, 0])
    mean2 = torch.Tensor([2, 1])

    data1 = torch.randn(n, 2) + torch.unsqueeze(mean1, dim=0)
    data2 = torch.randn(n, 2) + torch.unsqueeze(mean2, dim=0)
    labels1 = torch.zeros(n)
    labels2 = torch.ones(n)

    all_data = torch.cat([data1, data2], dim=0)
    all_labels = torch.cat([labels1, labels2], dim=0)

    permutation = torch.randperm(2 * n)
    shuffled_data = all_data[permutation]
    shuffled_labels = all_labels[permutation]

    return shuffled_data, shuffled_labels


data, labels = generate_data(100)
print('data', data)
print('labels', labels)
```

Run the program to see some data being printed out.

We can also go ahead and define a logistic regression model: a linear projection followed by a sigmoid.
For now, we just initialise the weights and the bias to small random numbers.

Fill in the initialisation for `weights` and `bias`.
`weights` should be a matrix of size 2x1 and `bias` should be a vector of size 1, both drawn from a normal distribution with mean 0 and standard deviation 0.01.

```
weights = ...
bias = ...

def sigmoid(x):
    return 1 / (1 + (torch.exp(-x)))

def run(x):
    x = torch.mm(x.unsqueeze(0), weights) + bias
    x = sigmoid(x)
    return x


# do the evaluation
number_correct = 0
for sample, target in zip(data, labels):
    output = run(sample)

    # check whether output is the same as target
    correct = (torch.squeeze(output) > 0.5).data == target
    if correct:
        number_correct += 1

print(number_correct)
```

Run the program to see what the untrained model is currently outputting.
You can also put more print statements in the `run` function to see what the intermediate values are.

Now the magic of PyTorch comes in.
So far, we've only been using the CPU to do this computation.
It's really easy to use the GPU instead!
Insert the following before the `for` loop:

```
data = data.cuda()
labels = labels.cuda()
weights = weights.cuda()
bias = bias.cuda()
```

Now everything runs on the GPU, without having to change anything else! You can see that it worked by the `cuda.FloatTensor` type and the `(GPU 0)` telling you on which GPU device it is.
While this model is so small that it's easily run on CPUs, that won't be the case for models in Deep Learning.
There will also be a simpler way of putting parameters on the GPU; you don't need to manually send every parameter to it in the future.

Next, we would like to actually train this model using gradient descent.
While we could compute the gradients manually, there is a much simpler solution that will also scale to massive models.
We will wrap the tensors in the `Variable` type, which will keep track of gradients for us.
You can always access the wrapped tensor with `.data`, but any operations you do with that won't have a tracked gradient.
Tensors wrapped in `Variable`s support almost the exact same operations as the tensors themselves: you can use all the familiar `torch.*` functions with them, add them, matrix-multiply them, and so on.
During these operations, the `Variable` will keep track of which operations were performed on the tensor, which allows you to backpropagate them.
When using `Variable`, tensors that don't need a gradient but are involved in the computation need to be wrapped in `Variables` too.
For the parameters that should be trained, pass the `requires_grad=True` argument to it to tell torch that these gradients should be tracked.

We will use a binary cross-entropy as loss function and let torch do the automatic differentiation.
Add this before the `for` loop:

```
from torch.autograd import Variable

# wrap the tensors in Variables
data = Variable(data)
labels = Variable(labels)
weights = Variable(weights, requires_grad=True)
bias = Variable(bias, requires_grad=True)

def binary_crossentropy(prediction, true):
    return - true * torch.log(prediction) - (1 - true) * torch.log(1 - prediction)

def train(x, target):
    predicted = run(x)
    loss = binary_crossentropy(predicted, target)
    loss = torch.mean(loss)

    # compute the gradients
    loss.backward()

    # stochastic gradient descent with step size of 0.01
    # accessing .data is ok here because we don't want to differentiate through the SGD update anyway
    weights.data -= 0.1 * weights.grad.data
    bias.data -= 0.1 * bias.grad.data

    # clear the gradients for the next gradient update
    weights.grad.data.fill_(0)
    bias.grad.data.fill_(0)


# do the training
for epoch in range(10):
    for sample, target in zip(data, labels):
        train(sample, target)
```

Notice how we never had to tell it how to differentiate the loss explicitly.
Try to replace the binary cross-entropy loss with a mean squared error loss.


this is a bit of effort, replace with optim


we're done with the basics!


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
