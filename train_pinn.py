import torch
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
import pinn
from burger_model import Burger
#import smplotlib

n_sample = 2000
nu = 0.01/np.pi

pinn_model = pinn.Model(2,1,20,5)

sample_space = np.random.uniform(low=-1, high=1, size=(n_sample,))
sample_time = np.random.uniform(low=0, high=1, size=(n_sample,))

sample_space = torch.tensor(sample_space, requires_grad=True).float().view(-1,1)
#torch.linspace(-1, 1, n_sample, requires_grad=True).view(-1,1)
sample_time = torch.tensor(sample_time, requires_grad=True).float().view(-1,1)
#torch.linspace(0, 1, n_sample, requires_grad=True).view(-1,1)

optimiser = torch.optim.Adam(pinn_model.parameters(), lr=1e-3)

u_init_true = torch.sin(-sample_space*np.pi)

sample_init_time = torch.zeros(n_sample, requires_grad=True).view(-1,1)
sample_init_space_lower = -1*torch.ones(n_sample, requires_grad=True).view(-1,1)
sample_init_space_upper = torch.ones(n_sample, requires_grad=True).view(-1,1)

for i in range(10000):
    optimiser.zero_grad()

    lambda1, lambda2, lambda3 = 1, 1, 1

    input = torch.hstack([sample_space,sample_time])
    input_init = torch.hstack([sample_space, sample_init_time])
    input_bc_lower = torch.hstack([sample_init_space_lower, sample_time])
    input_bc_upper = torch.hstack([sample_init_space_upper, sample_time])

    u_pred = pinn_model(input)
    u_init = pinn_model(input_init)
    u_bc_lower = pinn_model(input_bc_lower)
    u_bc_upper = pinn_model(input_bc_upper)

    loss_bc = torch.mean((u_bc_lower)**2) + torch.mean((u_bc_upper)**2)

    loss1 = torch.mean((u_init - u_init_true)**2)
    
    '''

    loss is just u_t + u*u_x - nu*u_xx + b.c.

    want u(x, t=0) = sin(x), that's what loss1 does (mean because needs scalar value)

    '''

    dudt_pred = torch.autograd.grad(u_pred, sample_time, torch.ones_like(u_pred), create_graph=True, allow_unused=True)[0]

    dudx_pred = torch.autograd.grad(u_pred, sample_space, torch.ones_like(u_pred), create_graph=True, allow_unused=True)[0]
    d2udx2_pred = torch.autograd.grad(dudx_pred, sample_space, torch.ones_like(u_pred), create_graph=True, allow_unused=True)[0]

    loss2 = torch.mean((dudt_pred + u_pred*dudx_pred - nu*d2udx2_pred)**2)

    loss = lambda1*loss1 + lambda2*loss2 + lambda3*loss_bc
    loss.backward(retain_graph=True) # need to retain graph (doesn't work otherwise)
    optimiser.step()
    if i % 250 == 0:
        print("loss", i, loss)
        
n_plot = 1001

x, t = torch.linspace(-1, 1, n_plot), torch.linspace(0,1,n_plot)

X, T = torch.meshgrid(x, t)

xcol = X.reshape(-1, 1)
tcol = T.reshape(-1, 1)

inputplot = torch.hstack([xcol,tcol])

u = pinn_model(inputplot)
u = u.reshape(x.numel(), t.numel())

plt.figure(figsize=(10,4))
plt.contourf(T, X, u.detach().numpy(), cmap=cm.jet, levels=300)
#plt.scatter(sample_time.detach().numpy(), sample_init_space_lower.detach().numpy())
#plt.scatter(sample_time.detach().numpy(), sample_init_space_upper.detach().numpy())
#plt.scatter(sample_init_time.detach().numpy(), sample_space.detach().numpy())
#plt.scatter(sample_time.detach().numpy(), sample_space.detach().numpy())
plt.xlabel("t")
plt.ylabel("x")
plt.savefig("burger.png", dpi=1000)

    

 