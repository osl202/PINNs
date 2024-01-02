**Physics Informed Neural Networks - Burgers' Equation**

The viscous Burgers' equation in one spatial dimension is given by
$$\frac{\partial u}{\partial t} + u\frac{\partial u}{\partial x} = \nu\frac{\partial^2 u}{\partial x^2}.$$

In this project we use a Physics Informed Neural Network (PINN) to find the solution to this equation.  Here we minimise the loss functional without any training data (unsupervised learning).  We construct the loss function using the PDE described above and the boundary conditions $u(x,t=0) = \text{sin}(x)$ and $u(x=0,1;t) = 0$ as

$$\mathcal{L}(\theta) = \frac{1}{N}\sum^N(\frac{\partial NN}{\partial t} + u\frac{\partial NN}{\partial x} - \nu\frac{\partial^2 NN}{\partial x^2})^2 + \frac{1}{N}\sum^N(NN(\theta;x,t=0)-\text{sin}(x))^2 + frac{1}{N}\sum^N(NN(\theta;x=0,1;t))^2$$.

