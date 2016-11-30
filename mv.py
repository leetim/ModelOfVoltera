import matplotlib.pyplot as plt
# import scipy as sp
import numpy as np

G = [[0.02, 0.03], [0.07, 0.05]]
E = [0.19, 0.5]

def dx1(x1, x2):
    return (E[0] - G[0][0]*x1 - G[0][1]*x2)*x1

def dx2(x1, x2):
    return (E[1] - G[1][0]*x1 - G[1][1]*x2)*x2

Y, X = np.mgrid[0:10:200j, 0:10:200j]
U = (E[0] - G[0][0]*X - G[0][1]*Y)*X
V = (E[1] - G[1][0]*X - G[1][1]*Y)*Y
speed = np.sqrt(U*U + V*V)
print(X.shape)

fig0, ax0 = plt.subplots()
strm = ax0.streamplot(X, Y, U, V, color=speed, linewidth=1, cmap=plt.cm.autumn)
fig0.colorbar(strm.lines)

def ode(x0, y0, dx, dy, h = 0.1, n = 101, t0 = 0):
        # Y, X = np.mgrid[0:10:100j, 0:10:100j]
        X, Y = [[0 for i in range(100)], [0 for i in range(100)]]
        X[0], Y[0] = x0, y0
        for i in range(1, 100):
            x_tild = X[i-1] + h*dx(X[i-1], Y[i-1])
            y_tild = Y[i-1] + h*dy(X[i-1], Y[i-1])
            X[i] = X[i-1] + h*(dx(X[i-1], Y[i-1]) + dx(X[i-1], y_tild))/2
            Y[i] = Y[i-1] + h*(dy(X[i-1], Y[i-1]) + dy(X[i-1], y_tild))/2
        # X = [[X[i] for j in range(n)] for i in range(n)]
        # Y = [Y for i in range(n)]
        return [Y, X]

Y1, X1 = ode(10, 10, dx1, dx2)
plt.plot(X1, Y1, linewidth=3)

# fig1, (ax1, ax2) = plt.subplots(ncols=2)
# ax1.streamplot(X, Y, U, V, density=[0.5, 1])
#
# lw = 5*speed / speed.max()
# ax2.streamplot(X, Y, U, V, density=0.6, color='k', linewidth=lw)

plt.show()
