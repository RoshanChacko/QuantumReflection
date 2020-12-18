import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import complex_ode
from sklearn.metrics import mean_squared_error as mse
#%%
pi=np.pi
hbar = 1
sigma = 1
m = 1
k_0 = 2
x_0 = -5

nu = hbar*k_0/m
kernel=np.array([1,-2,1])
H = 1j*hbar/(2*m)

def f(t, phi):
    d2phi = np.convolve(phi, kernel, mode='same') / (delta_x ** 2)
    return H * d2phi


def analytic(t):
    del_t = (hbar / (2 * hbar * (sigma ** 2))) * (t)
    expo = np.exp(-((x - x_0 - nu * t) ** 2) / (2 * (sigma ** 2) * (1 + del_t ** 2)))
    numphi = (1 / (sigma * np.sqrt(2 * pi * (1 + del_t ** 2)))) * expo
    return numphi

#%% Finding optimal N
optim_N = 0
err=10
maxdiff = 1000
plt.figure()
#The following loop will go through each value of N and get the mean squared error between the
#analytic and numerical calculations
for N in range(100,5000,100):
    #N=100
    xb = 30
    x = np.linspace(-xb,xb,N)
    delta_x=x[1]-x[0]
    e_term = np.exp(-((x-x_0)**2)/(4*(sigma**2)))
    phi_t0=(e_term/(np.sqrt(sigma)*((2*pi)**0.25)))*np.exp(1j*k_0*x)
    

    r = complex_ode(f)
    r.set_initial_value(phi_t0, 0)
    #r.integrate(8)
    #phisq = np.abs(r.integrate(8)) ** 2
    while r.successful() and r.t < 8:
        r.integrate(r.t + 7)
        phisq = (np.abs(r.y)) ** 2
        anfunc = analytic(r.t)
        plt.plot(N,mse(phisq,anfunc),'rx-')
        diff=(err-mse(phisq,anfunc))**2 #We take the difference between the previous and current MSE
        if diff<maxdiff: #smallest difference gives optimal N
            maxdiff=diff
            optim_N = N
        err=mse(phisq,anfunc)
