import torch


def time_derivative(un1,un, dt):
    return (un1 - un) / dt

def space_second_derivative(un,dx):
    return 1.0 / dx ** 2 * (un[2:] - 2 * un[1:-1] + un[0:-2])

def space_first_derivative(un,dx):
    return 1.0 / dx * (un[ 2:] + un[ 0:-2])


def convection_diffusion(un1,un,dt,dx,C,D):
    dudt = time_derivative(un1, un, dt)
    du2dx2 = space_second_derivative(un,dx)
    dudx = space_first_derivative(un,dx)
    diff_term = torch.multiply(D, du2dx2)
    conv_term = torch.multiply(C, dudx)

    t = torch.add(dudt[1:-1], diff_term)
    t = torch.add(t, conv_term)
    return t

def get_loss_one_year(D, C, un, un1, dt, dx):
    t = convection_diffusion(un1, un, dt, dx, C, D)

    res = torch.sum(t)
    res = res ** 2

    return res




