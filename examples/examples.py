#!/usr/bin/env python3

"""This file includes some of the examples from 'kalmanfilter.net'.
"""

try:
    from kalman import Kalman
    rootdir = 'examples/'
except ModuleNotFoundError:
    import sys
    sys.path.append('../')
    from kalman import Kalman
    rootdir = './'

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl_fontpath = mpl.get_data_path() + '/fonts/ttf/STIXGeneral.ttf'
mpl_fontprop = mpl.font_manager.FontProperties(fname=mpl_fontpath)
plt.rc('font', family='STIXGeneral', weight='normal', size=10)
plt.rc('mathtext', fontset='stix')


def example6():
    """This example is the implementation of
    `Example 6 - Estimating the temperature of the liquid in a tank`
    from `https://www.kalmanfilter.net/kalman1d.html`.

    This is a 1-D example of the Kalman filter.

    In this example, system matrixes and initial values are:
        F = [[1]],
        G = None
        H = [[1]]
        x = [10]
        P = [[100]]

    The process noise and measurement uncertainty are constant:
        Q = [[0.0001]]
        R = [[0.1**2]]
        
    The measurements are:
        49.95, 49.967, 50.1, 50.106, 49.992,
        49.819, 49.933, 50.007, 50.023, 49.99
    and the true values are:
        49.979, 50.025, 50, 50.003, 49.994,
        50.002, 49.999, 50.006, 49.998, 49.991
    """
    ## Setup
    F = [[1.]]
    G = None
    H = [[1.]]
    x = [10.]
    P = [[10000]]
    Q = [[0.0001]]
    R = [[0.1**2]]
    zs = [49.95, 49.967, 50.1, 50.106, 49.992,
          49.819, 49.933, 50.007, 50.023, 49.99]
    
    ## Filt
    kf = Kalman(F, G, H)
    kf.x = x
    kf.P = P
    x_kalman = []
    P_kalman = []
    K_kalman = []
    for z in zs:
        kf.predict(Q=Q)
        kf.update([z], R)
        x_kalman.append(kf.x[0])
        P_kalman.append(kf.P[0, 0])
        K_kalman.append(kf.K[0, 0])
        
    ## Plot
    idx = range(1, len(zs)+1)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(idx, [49.979, 50.025, 50, 50.003, 49.994,
                  50.002, 49.999, 50.006, 49.998, 49.991],
            'gd-', label='True values')
    ax.plot(idx, zs, 'bs-', label='Measurements')
    ax.plot(idx, x_kalman, 'ro-', label='Estimates')
    ax.legend()
    ax.grid()
    ax.set_xlabel('Measurement number')
    ax.set_ylabel(r'Temperature ($\degree$ C)')
    fig.suptitle('Liquid Temperature')
    fig.savefig(f'{rootdir}example6-1.svg')

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(idx, P_kalman, 'ro-')
    ax.grid()
    ax.set_xlabel('Measurement number')
    ax.set_ylabel('Estimate uncertainty')
    fig.suptitle('Estimate uncertainty')
    fig.savefig(f'{rootdir}example6-2.svg')

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(idx, K_kalman, 'k')
    ax.grid()
    ax.set_xlabel('Measurement number')
    ax.set_ylabel('Kalman gain')
    fig.suptitle('Kalman gain')
    fig.savefig(f'{rootdir}example6-3.svg')
    

def example7():
    """This example is the implementation of
    `Example 7 - Estimating the temperature of a heating liquid Ⅰ`
    from `https://www.kalmanfilter.net/kalman1d.html`.

    This is a 1-D example of the Kalman filter. Different from
    example6, the liquid is heating at a rate of 0.1°C every second
    (the measurement is taken every 5 seconds).

    In this example, system matrixes and initial values are:
        F = [[1]],
        G = None
        H = [[1]]
        x = [10]
        P = [[100]]

    The process noise and measurement uncertainty are constant:
        Q = [[0.0001]]
        R = [[0.1**2]]
        
    The measurements are:
        50.45, 50.967, 51.6, 52.106, 52.492,
        52.819, 53.433, 54.007, 54.523, 54.99

    and the true values are:
        50.479, 51.025, 51.5, 52.003, 52.494,
        53.002, 53.499, 54.006, 54.498, 54.991
    """
    ## Setup
    F = [[1.]]
    G = None
    H = [[1.]]
    x = [10.]
    P = [[10000]]
    Q = [[0.0001]]
    R = [[0.1**2]]
    zs = [50.45, 50.967, 51.6, 52.106, 52.492,
          52.819, 53.433, 54.007, 54.523, 54.99]
    
    ## Filt
    kf = Kalman(F, G, H)
    kf.x = x
    kf.P = P
    x_kalman = []
    P_kalman = []
    K_kalman = []
    for z in zs:
        kf.predict(Q=Q)
        kf.update([z], R)
        x_kalman.append(kf.x[0])
        P_kalman.append(kf.P[0, 0])
        K_kalman.append(kf.K[0, 0])
        
    ## Plot
    idx = range(1, len(zs)+1)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(idx, [50.479, 51.025, 51.5, 52.003, 52.494,
                  53.002, 53.499, 54.006, 54.498, 54.991],
            'gd-', label='True values')
    ax.plot(idx, zs, 'bs-', label='Measurements')
    ax.plot(idx, x_kalman, 'ro-', label='Estimates')
    ax.legend()
    ax.grid()
    ax.set_xlabel('Measurement number')
    ax.set_ylabel(r'Temperature ($\degree$ C)')
    fig.suptitle('Liquid Temperature')
    fig.savefig(f'{rootdir}example7-1.svg')
    
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(idx, P_kalman, 'ro-')
    ax.grid()
    ax.set_xlabel('Measurement number')
    ax.set_ylabel('Estimate uncertainty')
    fig.suptitle('Estimate uncertainty')
    fig.savefig(f'{rootdir}example7-2.svg')

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(idx, K_kalman, 'k')
    ax.grid()
    ax.set_xlabel('Measurement number')
    ax.set_ylabel('Kalman gain')
    fig.suptitle('Kalman gain')
    fig.savefig(f'{rootdir}example7-3.svg')


    ## Extend the measurements, by assuming the true measurement
    ## variance is the assumed value.
    zs_true2 = np.arange(50, 100, .5)
    zs2 = zs_true2 + np.random.randn(100) * R[0][0] ** .5
    for z in zs2[len(zs):]:
        kf.predict(Q=Q)
        kf.update([z], R)
        x_kalman.append(kf.x[0])
        P_kalman.append(kf.P[0, 0])
        K_kalman.append(kf.K[0, 0])
    idx = range(1, len(zs2)+1)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(idx, zs_true2,
            'gd-', label='True values')
    ax.plot(idx, zs2, 'bs-', label='Measurements')
    ax.plot(idx, x_kalman, 'ro-', label='Estimates')
    ax.legend()
    ax.grid()
    ax.set_xlabel('Measurement number')
    ax.set_ylabel(r'Temperature ($\degree$ C)')
    fig.suptitle('Liquid Temperature')
    fig.savefig(f'{rootdir}example7-4.svg')


def example8():
    """This example is the implementation of
    `Example 8 - Estimating the temperature of a heating liquid ⅠⅠ`
    from `https://www.kalmanfilter.net/kalman1d.html`.

    This is a 1-D example of the Kalman filter. This example is
    similar to example7, with only one change where the process
    uncertainty is increased from 0.0001 to 0.15. 

    In this example, system matrixes and initial values are:
        F = [[1]],
        G = None
        H = [[1]]
        x = [10]
        P = [[100]]

    The process noise and measurement uncertainty are constant:
        Q = [[0.15]]
        R = [[0.1**2]]
        
    The measurements are:
        50.45, 50.967, 51.6, 52.106, 52.492,
        52.819, 53.433, 54.007, 54.523, 54.99

    and the true values are:
        50.479, 51.025, 51.5, 52.003, 52.494,
        53.002, 53.499, 54.006, 54.498, 54.991
    """
   ## Setup
    F = [[1.]]
    G = None
    H = [[1.]]
    x = [10.]
    P = [[10000]]
    Q = [[0.15]]
    R = [[0.1**2]]
    zs = [50.45, 50.967, 51.6, 52.106, 52.492,
          52.819, 53.433, 54.007, 54.523, 54.99]
    
    ## Filt
    kf = Kalman(F, G, H)
    kf.x = x
    kf.P = P
    x_kalman = []
    P_kalman = []
    K_kalman = []
    for z in zs:
        kf.predict(Q=Q)
        kf.update([z], R)
        x_kalman.append(kf.x[0])
        P_kalman.append(kf.P[0, 0])
        K_kalman.append(kf.K[0, 0])
        
    ## Plot
    idx = range(1, len(zs)+1)
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(idx, [50.479, 51.025, 51.5, 52.003, 52.494,
                  53.002, 53.499, 54.006, 54.498, 54.991],
            'gd-', label='True values')
    ax.plot(idx, zs, 'bs-', label='Measurements')
    ax.plot(idx, x_kalman, 'ro-', label='Estimates')
    ax.legend()
    ax.grid()
    ax.set_xlabel('Measurement number')
    ax.set_ylabel(r'Temperature ($\degree$ C)')
    fig.suptitle('Liquid Temperature')
    fig.savefig(f'{rootdir}example8-1.svg')

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(idx, K_kalman, 'k')
    ax.grid()
    ax.set_xlabel('Measurement number')
    ax.set_ylabel('Kalman gain')
    fig.suptitle('Kalman gain')
    fig.savefig(f'{rootdir}example8-2.svg')


def example9():
    """This example is the implementation of
    `Example 9 - vehicle location estimation`
    from `https://www.kalmanfilter.net/multiExamples.html`

    This is a multi-dimensional example of the Kalman filter.

    In this example, system matrixes and initial values are:
        F = [[1, 1, 0.5, 0, 0,   0],
             [0, 1,   1, 0, 0,   0],
             [0, 0,   1, 0, 0,   0],
             [0, 0,   0, 1, 1, 0.5],
             [0, 0,   0, 0, 1,   1],
             [0, 0,   0, 0, 0,   1]]
        G = None
        H = [[1, 0, 0, 0, 0, 0],
             [0, 0, 0, 1, 0, 0]]
        x = [0, 0, 0, 0, 0, 0]
        P = np.diag([500] * 6)

    The process noise and measurement uncertainty are constant:
        Q = (F @
             [[0, 0,      0, 0, 0,      0],
              [0, 0,      0, 0, 0,      0],
              [0, 0, 0.2**2, 0, 0,      0],
              [0, 0,      0, 0, 0,      0],
              [0, 0,      0, 0, 0,      0],
              [0, 0,      0, 0, 0, 0.2**2]]
             @ F.T)
        R = [[3**2,    0],
             [   0, 3**2]]
        
    The measurements are:
        xs = [-393.66, -375.93, -351.04, -328.96, -299.35,
              273.36, -245.89, -222.58, -198.03, -174.17,
              -146.32, -123.72, -103.47, -78.23, -52.63,
              -23.34, 25.96, 49.72, 76.94, 95.38,
              119.83, 144.01, 161.84, 180.56, 201.42,
              222.62, 239.4, 252.51, 266.26, 271.75,
              277.4, 294.12, 301.23, 291.8, 299.89]
        ys = [300.4, 301.78, 295.1, 305.19, 301.06,
              302.05, 300, 303.57, 296.33, 297.65,
              297.41, 299.61, 299.6, 302.39, 295.04,
              300.09, 294.72, 298.61, 294.64, 284.88,
              272.82, 264.93, 251.46, 241.27, 222.98,
              203.73, 184.1, 166.12, 138.71, 119.71,
              100.41, 79.76, 50.62, 32.99, 2.14]
    """
    # Measurements
    xs = [-393.66, -375.93, -351.04, -328.96, -299.35,
          -273.36, -245.89, -222.58, -198.03, -174.17,
          -146.32, -123.72, -103.47, -78.23, -52.63,
          -23.34, 25.96, 49.72, 76.94, 95.38,
          119.83, 144.01, 161.84, 180.56, 201.42,
          222.62, 239.4, 252.51, 266.26, 271.75,
          277.4, 294.12, 301.23, 291.8, 299.89]
    ys = [300.4, 301.78, 295.1, 305.19, 301.06,
          302.05, 300, 303.57, 296.33, 297.65,
          297.41, 299.61, 299.6, 302.39, 295.04,
          300.09, 294.72, 298.61, 294.64, 284.88,
          272.82, 264.93, 251.46, 241.27, 222.98,
          203.73, 184.1, 166.12, 138.71, 119.71,
          100.41, 79.76, 50.62, 32.99, 2.14]
    zs = zip(xs, ys)
    
    ## Setup
    kf = Kalman()
    kf.F = [[1, 1, 0.5, 0, 0,   0],
            [0, 1,   1, 0, 0,   0],
            [0, 0,   1, 0, 0,   0],
            [0, 0,   0, 1, 1, 0.5],
            [0, 0,   0, 0, 1,   1],
            [0, 0,   0, 0, 0,   1]]
    kf.H = [[1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0]]
    kf.x = [0, 0, 0, 0, 0, 0]
    kf.P = np.diag([500] * 6)

    R = [[3**2,    0],
         [   0, 3**2]]
    Q = (kf.F @
         [[0, 0,      0, 0, 0,      0],
          [0, 0,      0, 0, 0,      0],
          [0, 0, 0.2**2, 0, 0,      0],
          [0, 0,      0, 0, 0,      0],
          [0, 0,      0, 0, 0,      0],
          [0, 0,      0, 0, 0, 0.2**2]]
         @ kf.F.T)
    ## Filt
    x_kalman = []
    for z in zs:
        kf.predict(Q=Q)
        kf.update(z, R)
        x_kalman.append(kf.x)

    # Plot
    t = np.arange(35)
    x_true = [-400 + 25 * i if i < 16 else
              300 * np.sin((25 * i - 400) / 300) for i in t]
    y_true = [300 if i < 16 else
              300 * np.cos((25 * i - 400) / 300) for i in t]
    fig = plt.figure(figsize=(7, 3))
    ax = fig.add_subplot(111)
    ax.plot(x_true, y_true, 'gd-',label='True values')
    ax.plot(xs, ys, 'bs-',label='Measurements')
    ax.plot([i[0] for i in x_kalman],
            [i[3] for i in x_kalman],
            'ro-', label='Estimates')
    ax.set_xlabel('$x$ (m)')
    ax.set_ylabel('$y$ (m)')
    ax.legend()
    ax.grid()
    ax.set_aspect('equal')
    fig.suptitle('Vehicle position')
    fig.savefig(f'{rootdir}example9.svg')
    

def example10():
    """This example is the implementation of
    `Example 10 - rocket altitude estimation`
    from `https://www.kalmanfilter.net/multiExamples.html`

    This is a multi-dimensional example of the Kalman filter
    with control input.

    In this example, system matrixes and initial values are:
        F = [[1, dt],
             [0,  1]]
        G = [[dt ** 2 / 2], [dt]]
        H = [[1, 0]]
        x = [0, 0]
        P = np.diag([500] * 2)
    where dt = 0.25 is the sampling frequency.
    The process noise and measurement uncertainty are constant:
        Q = G @ G.T * 0.1 ** 2
        R = [[20 ** 2]]
        
    The measurements are:
        zs = (-32.4, -11.1, 18, 22.9, 19.5,
              28.5, 46.5, 68.9, 48.2, 56.1,
              90.5, 104.9, 140.9, 148, 187.6,
              209.2, 244.6, 276.4, 323.5, 357.3,
              357.4, 398.3, 446.7, 465.1, 529.4,
              570.4, 636.8, 693.3, 707.3, 748.5)
        us = (39.72, 40.02, 39.97, 39.81, 39.75,
              39.6, 39.77, 39.83, 39.73, 39.87,
              39.81, 39.92, 39.78, 39.98, 39.76,
              39.86, 39.61, 39.86, 39.74, 39.87,
              39.63, 39.67, 39.96, 39.8, 39.89,
              39.85, 39.9, 39.81, 39.81, 39.68)
    """
    ## Setup
    g = -9.8
    dt = 0.25
    F = np.array([[1, dt],
                  [0,  1]])
    G = np.array([[dt ** 2 / 2], [dt]])
    H = np.array([[1, 0]])
    x = np.array([0, 0])
    P = np.diag([500] * 2)
    Q = G @ G.T * 0.1 ** 2
    R = np.array([[20 ** 2]])
    zs = (-32.4, -11.1, 18, 22.9, 19.5,
          28.5, 46.5, 68.9, 48.2, 56.1,
          90.5, 104.9, 140.9, 148, 187.6,
          209.2, 244.6, 276.4, 323.5, 357.3,
          357.4, 398.3, 446.7, 465.1, 529.4,
          570.4, 636.8, 693.3, 707.3, 748.5)
    us = (39.72, 40.02, 39.97, 39.81, 39.75,
          39.6, 39.77, 39.83, 39.73, 39.87,
          39.81, 39.92, 39.78, 39.98, 39.76,
          39.86, 39.61, 39.86, 39.74, 39.87,
          39.63, 39.67, 39.96, 39.8, 39.89,
          39.85, 39.9, 39.81, 39.81, 39.68)
    
    ## Filt
    kf = Kalman(F, G, H)
    kf.x = x
    kf.P = P
    z_kalman = []
    ks = []
    kf.predict([g], Q)
    for u, z in zip(us, zs):
        kf.update([z], R)
        z_kalman.append(kf.x[0])
        ks.append(kf.K[0, 0])
        kf.predict([u + g], Q)
        
    ## Plot
    t = np.arange(len(zs)) * dt
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(t, 0.5 * 30 * t ** 2, 'gd-', label='True value')
    ax.plot(t, zs, 'bs-', label='Measurements')
    ax.plot(t, z_kalman, 'ro-', label='Estimates')
    ax.grid()
    ax.legend()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Altitude (m)')
    fig.suptitle('Rocket altitude')
    fig.savefig(f'{rootdir}example10-1.svg')

    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    ax.plot(t, ks, 'k')
    ax.grid()
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Kalman gain')
    fig.suptitle('Kalman gain')
    fig.savefig(f'{rootdir}example10-2.svg')
    

if __name__ == '__main__':
    example6()
    plt.show()
    example7()
    plt.show()
    example8()
    plt.show()
    example9()
    plt.show()
    example10()
    plt.show()
