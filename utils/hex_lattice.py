# hex_lattice.py

import numpy as np
import random

def gen_hex_lattice(w, a=1.):
    """
    Fill a circle centered at the origin with radius w
        with a hexagonal lattice nodes separted by a
    """
    L = 2 * w / a + 1
    i, j = np.meshgrid(np.arange(-L, L), np.arange(-L, L))
    i, j = i.ravel()[None, :], j.ravel()[None, :]
    # Random rotation
    theta = np.random.rand(1)[0] * 2 * np.pi
    R = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    u, v = np.array([[1], [0]]), np.array([[0.5], [np.sqrt(3)/2]])
    u, v = np.dot(R, u), np.dot(R, v)
    XE, YE = a * (u * i + v * j)
    idx = XE ** 2 + YE ** 2 < w ** 2
    XE, YE = XE[idx], YE[idx]
    # Random center
    i0, j0 = (np.random.rand(2) - 0.5)
    XE, YE = XE + i0 * a, YE + i0 * a
    
    return XE, YE

def gen_color_lattice(r, g, b, w, a=1.):
    """
    Fill a circle centered at the origin with radius w
    with a hexagonal lattice nodes separted by a
    Lattice cone type ratio is dependent on r,g,b 
    values passed in; must add up to 1
    """
    L = 2 * w / a + 1
    i, j = np.meshgrid(np.arange(-L, L), np.arange(-L, L))
    i, j = i.ravel()[None, :], j.ravel()[None, :]
    # Random rotation
    theta = np.random.rand(1)[0] * 2 * np.pi
    R = np.array([[np.cos(theta), np.sin(theta)],
                  [-np.sin(theta), np.cos(theta)]])
    u, v = np.array([[1], [0]]), np.array([[0.5], [np.sqrt(3)/2]])
    u, v = np.dot(R, u), np.dot(R, v)
    XE, YE = a * (u * i + v * j)
    idx = XE ** 2 + YE ** 2 < w ** 2
    XE, YE = XE[idx], YE[idx]
    # Random center
    i0, j0 = (np.random.rand(2) - 0.5)
    XE, YE = XE + i0 * a, YE + i0 * a

    n = len(XE)
    ratio = [int(n*r), int(n*g)]
    r = random.sample(range(0, n), n)
    rb = (np.asarray(range(0, n, int(n/(b*n)))) + random.randint(0,20)) % n
    for i in range(0, len(rb)):
        r.remove(rb[i])

    cones = [r[0:ratio[0]], r[ratio[0]:len(r)], rb] 
    XE = [XE[cones[0]], XE[cones[1]], XE[cones[2]]]
    YE = [YE[cones[0]], YE[cones[1]], YE[cones[2]]]

    return XE, YE

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    #for i in range(10):i
    #XE, YE = gen_hex_lattice(8.00, 0.5)
    XE, YE = gen_color_lattice(0.47, 0.47, 0.06, 7.00, 0.5)
    
    # Trichromatic
    fig, ax = plt.subplots(2,2)
    ax[0,0].scatter(XE[0], YE[0], color='red')
    ax[0,1].scatter(XE[1], YE[1], color='green')
    ax[1,0].scatter(XE[2], YE[2], color='blue')
    ax[1,1].scatter(XE[0], YE[0], color='red')
    ax[1,1].scatter(XE[1], YE[1], color='green')
    ax[1,1].scatter(XE[2], YE[2], color='blue')
    ax[0,0].set_aspect('equal')
    ax[0,1].set_aspect('equal')
    ax[1,0].set_aspect('equal')
    ax[1,1].set_aspect('equal')
    plt.show()

    # Deuteranope
    #fig, ax = plt.subplots(1,3)
    #ax[0].scatter(XE[0], YE[0], color='red')
    #ax[1].scatter(XE[2], YE[2], color='blue')
    #ax[2].scatter(XE[0], YE[0], color='red')
    #ax[2].scatter(XE[2], YE[2], color='blue')
    #ax[0].set_aspect('equal')
    #ax[1].set_aspect('equal')
    #ax[2].set_aspect('equal')

    #plt.show()

    #plt.scatter(XE[0], YE[0], color='red')
    #plt.scatter(XE[1], YE[1], color='green')
    #plt.scatter(XE[2], YE[2], color='blue')
    #plt.axes().set_aspect('equal')
    #plt.show()

#    idx = (-w < XE) * (XE < w) * (-w < YE) * (YE < w)
