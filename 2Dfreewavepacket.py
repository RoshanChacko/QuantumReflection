import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import complex_ode

#%% Defining constants
pi=np.pi
hbar = 1
sigma = 1
m = 1
k_0x,k_0y = 2,10
x_0, y_0 = -5,-5
N, xb = 100, 20
xv = np.linspace(-xb,xb,N)
yv = np.linspace(-xb,xb,N)
x,y = np.meshgrid(xv,yv, sparse=True) #Here, we add our y values and then do a meshgrid where x is a transpose of y
delta_x=xv[1]-xv[0]
e_term = np.exp(-((x-x_0)**2+(y-y_0)**2)/(4*(sigma**2)))

phi_t0=(e_term/(np.sqrt(sigma)*((2*pi)**0.25)))*np.exp(1j*(k_0x*x+k_0y*y))


kernel=np.array([1,-2,1])

H = 1j*hbar/(2*m)

def f(t,phi):
    phi = np.ravel(phi) #We need to flatten the array to apply our 1D convolution
    d2phi = np.convolve(phi, kernel, mode='same') / (delta_x ** 2)
    return H*d2phi

r = complex_ode(f)
r.set_initial_value(phi_t0.ravel(),0)
dat = np.reshape(r.y,(N,N)) #After doing our convolution and calculations, we reshape it back to the original N by N shape
phisq=np.abs(dat)*np.abs(dat)
plt.figure()
plt.imshow(phisq)
ax = plt.gca()
while r.successful() and r.t < 8:
    r.integrate(r.t+0.05)
    dat = np.reshape(r.y,(N,N))
    phisq=np.abs(dat)*np.abs(dat)
    ax.clear()
    ax.imshow(phisq)
    plt.pause(0.1)

