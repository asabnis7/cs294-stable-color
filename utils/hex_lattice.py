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
    
    n = len(XE)
    #ratio = [int(n*0.47), int(n*0.47), int(n*0.06)]
    ratio = [int(n*0.9), int(n*0), int(n*0.1)]
    print ratio
    color_ind = random.sample(range(0, n), n)
    #print color_ind
    
    ind = 0
    redXE = XE[color_ind[ind:ratio[0]]]
    redYE = YE[color_ind[ind:ratio[0]]] 
    #print color_ind[ind:ratio[0]]
    
    ind += ratio[0]
    greenXE = XE[color_ind[ind:ind+ratio[1]]]
    greenYE = YE[color_ind[ind:ind+ratio[1]]] 
    #print color_ind[ind:ind+ratio[1]]
    
    ind += ratio[1]
    blueXE = XE[color_ind[ind:-1]]
    blueYE = YE[color_ind[ind:-1]] 
    #print color_ind[ind:-1]

    #XE = [[redXE], [greenXE], [blueXE]]
    #YE = [[redYE], [greenYE], [blueYE]]
    return XE, YE


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    #for i in range(10):i
    XE, YE = gen_hex_lattice(8.00, 0.5)
    
    # Trichromatic
    #fig, ax = plt.subplots(2,2)
    #ax[0,0].scatter(XE[0], YE[0], color='red')
    #ax[0,1].scatter(XE[1], YE[1], color='green')
    #ax[1,0].scatter(XE[2], YE[2], color='blue')
    #ax[1,1].scatter(XE[0], YE[0], color='red')
    #ax[1,1].scatter(XE[1], YE[1], color='green')
    #ax[1,1].scatter(XE[2], YE[2], color='blue')
    #ax[0,0].set_aspect('equal')
    #ax[0,1].set_aspect('equal')
    #ax[1,0].set_aspect('equal')
    #ax[1,1].set_aspect('equal')

    # Deuteranope
    fig, ax = plt.subplots(1,3)
    ax[0].scatter(XE[0], YE[0], color='red')
    ax[1].scatter(XE[2], YE[2], color='blue')
    ax[2].scatter(XE[0], YE[0], color='red')
    ax[2].scatter(XE[2], YE[2], color='blue')
    ax[0].set_aspect('equal')
    ax[1].set_aspect('equal')
    ax[2].set_aspect('equal')

    plt.show()

    #plt.scatter(XE[0], YE[0], color='red')
    #plt.scatter(XE[1], YE[1], color='green')
    #plt.scatter(XE[2], YE[2], color='blue')
    #plt.axes().set_aspect('equal')
    #plt.show()

#    idx = (-w < XE) * (XE < w) * (-w < YE) * (YE < w)
