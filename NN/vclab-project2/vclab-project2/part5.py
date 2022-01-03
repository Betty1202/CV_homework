import torch
import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F  # useful stateless functions
from copy import deepcopy
from part1 import device, dtype, loader_train, print_every, loader_val, loader_test
from models.LeNet5 import LeNet5
from models.ResNet import resnet18, resnet34, resnet50, resnet101, resnet152

'''
------------------------------------------------------------------------------------------------------------------------
Part V. CIFAR-10 open-ended challenge
In this section, you can experiment with whatever ConvNet architecture you'd like on CIFAR-10.

Now it's your job to experiment with architectures, hyperparameters, loss functions, and optimizers to train a 
model that achieves at least 70% accuracy on the CIFAR-10 validation set within 10 epochs. You can use the 
check_accuracy and train functions from above. You can use either nn.Module or nn.Sequential API.

Describe what you did at the end of this notebook.

Here are the official API documentation for each component. One note: what we call in the class 
"spatial batch norm" is called "BatchNorm2D" in PyTorch.

Layers in torch.nn package: http://pytorch.org/docs/stable/nn.html
Activations: http://pytorch.org/docs/stable/nn.html#non-linear-activations
Loss functions: http://pytorch.org/docs/stable/nn.html#loss-functions
Optimizers: http://pytorch.org/docs/stable/optim.html

Things you might try:
Filter size: Above we used 5x5; would smaller filters be more efficient?
Number of filters: Above we used 32 filters. Do more or fewer do better?
Pooling vs Strided Convolution: Do you use max pooling or just stride convolutions?
Batch normalization: Try adding spatial batch normalization after convolution layers and vanilla batch normalization 
after affine layers. Do your networks train faster?

Network architecture: The network above has two layers of trainable parameters. Can you do better with a 
deep network? Good architectures to try include:
[conv-relu-pool]xN -> [affine]xM -> [softmax or SVM]
[conv-relu-conv-relu-pool]xN -> [affine]xM -> [softmax or SVM]
[batchnorm-relu-conv]xN -> [affine]xM -> [softmax or SVM]

Global Average Pooling: Instead of flattening and then having multiple affine layers, perform convolutions until 
your image gets small (7x7 or so) and then perform an average pooling operation to get to a 1x1 image picture 
(1, 1 , Filter#), which is then reshaped into a (Filter#) vector. This is used in Google's Inception Network 
(See Table 1 for their architecture).

Regularization: Add l2 weight regularization, or perhaps use Dropout.

Tips for training
For each network architecture that you try, you should tune the learning rate and other hyperparameters. 
When doing this there are a couple important things to keep in mind:

-If the parameters are working well, you should see improvement within a few hundred iterations
-Remember the coarse-to-fine approach for hyperparameter tuning: start by testing a large range of 
hyperparameters for just a few training iterations to find the combinations of parameters that are working at all.
-Once you have found some sets of parameters that seem to work, search more finely around these parameters. 
You may need to train for more epochs.
-You should use the validation set for hyperparameter search, and save your test set for evaluating 
your architecture on the best parameters as selected by the validation set.


Going above and beyond
If you are feeling adventurous there are many other features you can implement to try and 
improve your performance. You are not required to implement any of these, but don't miss the fun if you have time!

-Alternative optimizers: you can try Adam, Adagrad, RMSprop, etc.
-Alternative activation functions such as leaky ReLU, parametric ReLU, ELU, or MaxOut.
-Model ensembles
-Data augmentation
-New Architectures
-ResNets where the input from the previous layer is added to the output.
-DenseNets where inputs into previous layers are concatenated together.
-This blog has an in-depth overview

Have fun and happy training!
'''


def check_accuracy_part34(loader, model):
    if loader.dataset.train:
        print('Checking accuracy on validation set')
    else:
        print('Checking accuracy on test set')
    num_correct = 0
    num_samples = 0
    model.eval()  # set model to evaluation mode
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)
            scores = model(x)
            _, preds = scores.max(1)
            num_correct += (preds == y).sum()
            num_samples += preds.size(0)
        acc = float(num_correct) / num_samples
        print('Got %d / %d correct (%.2f)' % (num_correct, num_samples, 100 * acc))
    return acc


def flatten(x):
    N = x.shape[0]  # read in N, C, H, W
    return x.view(N, -1)  # "flatten" the C * H * W values into a single vector per image


def train_part34(model, optimizer, epochs=1, log=True):
    """
    Train a model on CIFAR-10 using the PyTorch Module API.

    Inputs:
    - model: A PyTorch Module giving the model to train.
    - optimizer: An Optimizer object we will use to train the model
    - epochs: (Optional) A Python integer giving the number of epochs to train for

    Returns: Nothing, but prints model accuracies during training.
    """
    model = model.to(device=device)  # move the model parameters to CPU/GPU
    for e in range(epochs):
        for t, (x, y) in enumerate(loader_train):
            model.train()  # put model to training mode
            x = x.to(device=device, dtype=dtype)  # move to device, e.g. GPU
            y = y.to(device=device, dtype=torch.long)

            scores = model(x)
            loss = F.cross_entropy(scores, y)

            # Zero out all of the gradients for the variables which the optimizer
            # will update.
            optimizer.zero_grad()

            # This is the backwards pass: compute the gradient of the loss with
            # respect to each  parameter of the model.
            loss.backward()

            # Actually update the parameters of the model using the gradients
            # computed by the backwards pass.
            optimizer.step()

            if t % print_every == 0 and log:
                print('Iteration %d, loss = %.4f' % (t, loss.item()))
                check_accuracy_part34(loader_val, model)
                print()


################################################################################
# TODO:                                                                        #
# Experiment with any architectures, optimizers, and hyperparameters.          #
# Achieve AT LEAST 70% accuracy on the *validation set* within 10 epochs.      #
#                                                                              #
# Note that you can use the check_accuracy function to evaluate on either      #
# the test set or the validation set, by passing either loader_test or         #
# loader_val as the second argument to check_accuracy. You should not touch    #
# the test set until you have finished your architecture and  hyperparameter   #
# tuning, and only run the test set once at the end to report a final value.   #
################################################################################
model = None
optimizer = None

# ***** START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

learning_rates = [1e-3, 1e-2, 1e-1]

models = {"LeNet5": LeNet5,  # 70+
          "ResNet18": resnet18(False),  # 80.60
          "ResNet34": resnet34(False),
          "ResNet50": resnet50(False),
          "ResNet101": resnet101(False),
          "ResNet152": resnet152(False)}

best_model = None
best_model_name = None
best_val_acc = 0
best_opt = None

for lr in learning_rates:
    for model_name, model in models.items():
        model = deepcopy(model)
        model.to(device)
        optimizers = {"Nesterov": optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True),  # validation 70+
                      "Adam": optim.Adam(model.parameters(), lr=lr),  # Poor results
                      "Adagrad": optim.Adagrad(model.parameters(), lr=lr),  # validation 72.5
                      "RMSprop": optim.RMSprop(model.parameters(), lr=lr)}  # Poor results
        for opt_name, optimizer in optimizers.items():
            print(f"==========lr: {lr}, model: {model_name}, opt: {opt_name}==========")
            train_part34(model, optimizer, epochs=10, log=False)
            acc = check_accuracy_part34(loader_val, model)
            if acc > best_val_acc:
                best_val_acc = acc
                best_model = model
                best_model_name = model_name
                best_opt = opt_name

# *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

################################################################################
#                                 END OF YOUR CODE
################################################################################

'''
Describe what you did
In the cell below you should write an explanation of what you did, any additional features 
that you implemented, and/or any graphs that you made in the process of training and 
evaluating your network.

First built a small basic residual block with 3X3 conv followed by BatchNorm and again 
3X3 conv followed by BatchNorm. In the main architecture first conv followed by BatchNorm 
and then 3 resiual blocks as described above which is followed by maxpooling and then fc.
Adam optimiser is used. No scheduler used.

Test set -- run this only once
Now that we've gotten a result we're happy with, we test our final model on the test set 
(which you should store in best_model). Think about how this compares to your validation set accuracy.
'''

best_model = model
check_accuracy_part34(loader_test, best_model)
