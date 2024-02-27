import torch
from loss import loss_function
import numpy as np
import matplotlib.pyplot as plt

def minimize(D,C,depths_uniform,dx):
    optimizer = torch.optim.SGD([C,D], lr=0.0001)
    C.requires_grad = True
    D.requires_grad = True
    num_iters = 10000
    hist = [] #np.zeros(num_iters)
    n = 0
    loss0 = loss_function(depths_uniform, dx, D, C)
    while loss0.item() > 1.0:
    # for n in range(num_iters):
        optimizer.zero_grad()
        loss0 = loss_function(depths_uniform, dx, D, C)
        print(n,loss0.item())
        hist.append(loss0.item())
        n = n+1
        loss0.backward()
        optimizer.step()


    print(C,D)
    plt.figure()
    hist = np.array(hist)
    plt.plot(hist)
    plt.title('loss function')
    plt.xlabel('epoch')
    plt.savefig('loss_function_history.png')
