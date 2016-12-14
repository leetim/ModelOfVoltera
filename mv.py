import matplotlib.pyplot as plt
import scipy as sp
import numpy as np
import random as rnd
from scipy.optimize import leastsq

G = [[0.02, 0.03], [0.07, 0.05]]
E = [0.19, 0.5]

def dx1_res(x1, x2, E, G):
    return (E[0] - G[0][0]*x1 - G[0][1]*x2)*x1

def dx2_res(x1, x2, E, G):
    return (E[1] - G[1][0]*x1 - G[1][1]*x2)*x2

def Jak(x1, x2, E1, G1):
    a11 = E1[0] - 2*G1[0][0]*x1 - G1[0][1]*x2
    a12 = -G1[0][1]*x1
    a21 = -G1[1][0]*x2
    a22 = E1[1] - G1[1][0]*x1 - 2*G1[1][1]*x2
    b = a11 + a22
    c = a11*a22 - a12*a21
    if (b**2 - 4*c) > 0:
        l1 = (b + np.sqrt(b**2 - 4*c))/2
        l2 = (b - np.sqrt(b**2 - 4*c))/2
        f = False
    else:
        l1, l2 = b, b
        f = True
    return [l1, l2, f]

def solve(E, G):
    x2 = (E[1] - E[0]*G[1][0]/G[0][0])/(G[1][1] - G[0][1]*G[1][0]/G[0][0])
    x1 = (E[0] - G[0][1]*x2)/G[0][0]
    return [x1, x2]

def ust(E1, G1):
    x1, x2 = solve(E1, G1)
    print('Points:')
    print([0.0, 0.0])
    print([0.0, E1[1]/G1[1][1]])
    print([E1[0]/G1[0][0], 0.0])
    print([x1, x2])
    print(Jak(0.0, 0.0, E1, G1))
    print(Jak(0.0, E1[1]/G1[1][1], E1, G1))
    print(Jak(E1[0]/G1[0][0], 0.0, E1, G1))
    print(Jak(x1, x2, E1, G1))

print 'Original system'
ust(E, G)

def draw(E, G):
    Y, X = np.mgrid[0:15:400j, 0:15:400j]
    U = dx1_res(X, Y, E, G)
    V = dx2_res(X, Y, E, G)
    # V = (E[1] - G[1][0]*X - G[1][1]*Y)*Y
    speed = np.sqrt(U*U + V*V)
    print(X.shape)

    fig0, ax0 = plt.subplots()
    strm = ax0.streamplot(X, Y, U, V, color=speed, linewidth=1, cmap=plt.cm.autumn)
    ax0.grid(color='grey', linestyle='dotted', linewidth=1)
    ax0.set_xlabel('X(t)');
    ax0.set_ylabel('Y(t)');
    ax0.set_xlim(0, 15);
    ax0.set_ylim(0, 15);

    fig0.colorbar(strm.lines)

def ode(x0, y0, dx, dy, h = 0.1, n = 101, t0 = 0):
        # Y, X = np.mgrid[0:10:100j, 0:10:100j]
        X, Y = [[0 for i in range(n)], [0 for i in range(n)]]
        X[0], Y[0] = x0, y0
        for i in range(1, n):
            x_tild = X[i-1] + h*dx(X[i-1], Y[i-1])
            y_tild = Y[i-1] + h*dy(X[i-1], Y[i-1])
            X[i] = X[i-1] + h*(dx(X[i-1], Y[i-1]) + dx(X[i-1], y_tild))/2
            Y[i] = Y[i-1] + h*(dy(X[i-1], Y[i-1]) + dy(X[i-1], y_tild))/2
        # X = [[X[i] for j in range(n)] for i in range(n)]
        # Y = [Y for i in range(n)]
        return np.array([Y, X])

dx1 = lambda x1, x2: dx1_res(x1, x2, E, G)
dx2 = lambda x1, x2: dx2_res(x1, x2, E, G)

draw(E, G)
Y1, X1 = ode(1.0, 2.0, dx1, dx2)
dX1, dY1 = dx1(X1, Y1), dx2(X1, Y1)
mkError = lambda x: (1 + (2*rnd.random() - 1.)/2)*x
dY1 = list(map(mkError, dY1))
dX1 = list(map(mkError, dX1))
plt.plot(X1, Y1, linewidth=3)

# print(sum(dY12 - dY1))

funcX = lambda tpl, x, y: (tpl[0] - tpl[1]*x - tpl[2]*y)*x
erFuncX = lambda tpl, x, y, f: funcX(tpl, x, y) - f
tplFinalX, succes = leastsq(erFuncX, (0.3, 0.4, 0.5), args=(X1, Y1, dX1));

funcY = lambda tpl, x, y: (tpl[0] - tpl[1]*x - tpl[2]*y)*y
erFuncY = lambda tpl, x, y, f: funcY(tpl, x, y) - f
tplFinalY, succes = leastsq(erFuncY, (0.3, 0.4, 0.5), args=(X1, Y1, dY1));

# print(tplFinalX)
# print(tplFinalY)
print("New cofficients:")
print(E)
print(G)
E1 = [tplFinalX[0], tplFinalY[0]]
G1 = [[tplFinalX[1], tplFinalX[2]], [tplFinalY[1], tplFinalY[2]]]
print(E1)
print(G1)
draw(E1, G1)
print 'Droped system'
ust(E1, G1)


# fig1, (ax1, ax2) = plt.subplots(ncols=2)
# ax1.streamplot(X, Y, U, V, density=[0.5, 1])
#
# lw = 5*speed / speed.max()
# ax2.streamplot(X, Y, U, V, density=0.6, color='k', linewidth=lw)

plt.show()
