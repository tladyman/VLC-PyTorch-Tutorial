import torch


def generate_data(n):  # n is the number of samples to generate from each distribution
    mean1 = torch.Tensor([0, 0, 0])
    mean2 = torch.Tensor([1, 1, 1])

    data1 = torch.randn(n, 3) + torch.unsqueeze(mean1, dim=0)
    data2 = torch.randn(n, 3) + torch.unsqueeze(mean2, dim=0)
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




weights = torch.randn(3, 1) * 0.01
bias = torch.randn(1) * 0.01

def sigmoid(x):
    return 1 / (1 + (torch.exp(-x)))

def run(x):
    x = torch.mm(x.unsqueeze(0), weights) + bias
    x = sigmoid(x)
    return x



data = data.cuda()
labels = labels.cuda()
weights = weights.cuda()
bias = bias.cuda()



from torch.autograd import Variable

# wrap the tensors in Variables
data = Variable(data)
labels = Variable(labels)
weights = Variable(weights, requires_grad=True)
bias = Variable(bias, requires_grad=True)

import torch.optim as optim

optimiser = optim.SGD([weights, bias], lr=0.1)

def binary_crossentropy(prediction, true):
    return - true * torch.log(prediction) - (1 - true) * torch.log(1 - prediction)

def train(x, target):
    predicted = run(x)
    loss = binary_crossentropy(predicted, target).mean()

    # compute the gradients and accumulate them in the Variables that have requires_grad=True
    loss.backward()

    # stochastic gradient descent with step size of 0.1
    # accessing .data is ok here because we don't want to differentiate through the SGD update anyway
    optimiser.step()

    # clear the gradients for the next gradient update
    optimiser.zero_grad()


# do the training
for epoch in range(2):
    for sample, target in zip(data, labels):
        train(sample, target)




# do the evaluation
number_correct = 0
for sample, target in zip(data, labels):
    output = run(sample)
    output = torch.squeeze(output)

    if not isinstance(target, float):  # this is here to make a later section work without changing stuff...
        target = target.data[0]  # Variable -> float
        output = output.data  # Variable -> Tensor

    # check whether output is the same as target
    correct = (output > 0.5)[0] == target
    if correct:
        number_correct += 1


print('accuracy:', number_correct / data.size(0))

