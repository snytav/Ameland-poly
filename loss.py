from get_loss_one_year import get_loss_one_year
import torch


def loss_function(depths_uniform,dx,D,C):

    num_years = depths_uniform.shape[1]
    sum = 0
    for n in range(num_years-1):
        t0 = get_loss_one_year(D, C, torch.from_numpy(depths_uniform[:, n]),
                                  torch.from_numpy(depths_uniform[:, n+1]), 1.0, dx)
        sum += t0

    return sum

