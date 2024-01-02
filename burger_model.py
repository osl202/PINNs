import numpy as np

class Burger():
    def __init__(self,nx, nu):

        self.nx = nx
        self.nu = nu
        self.dx = 2*np.pi/nx
        self.dt = nu*self.dx
        self.nt = int(1/self.dt)

    def solve(self):

        x = np.linspace(0, 2*np.pi, self.nx)
        t = np.linspace(0, self.dt*self.nt, self.nt)

        u = np.sin(x)

        for i in range(self.nt):
            if i%1000 == 0:
                print(100 * i/self.nt, "%")
            for j in range(1, self.nx - 1):

                u[j] = u[j] - u[j]*(self.dt/self.dx)*(u[j] - u[j-1]) + self.nu*(self.dt/self.dx**2)*(u[j+1] - 2*u[j] + u[j-1])

                u[0] = u[0] - u[0]*(self.dt/self.dx)*(u[0] - u[-2]) + self.nu*(self.dt/self.dx**2)*(u[1] - 2*u[0] + u[-2])

                u[-1] = u[0]

        return u








    



