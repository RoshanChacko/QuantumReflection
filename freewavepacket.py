import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import complex_ode

#%% Defining constants
pi=np.pi
hbar = 1
sigma = 1
m = 1
k_0 = 2
x_0 = -5
N, xb = 1000, 30
x = np.linspace(-xb,xb,N)
delta_x=x[1]-x[0]

e_term = np.exp(-((x-x_0)**2)/(4*(sigma**2))) #the exponential term of the gaussian wavepacket equation

phi_t0=(e_term/(np.sqrt(sigma)*((2*pi)**0.25)))*np.exp(1j*k_0*x) #wavepacket at time t=0

nu = hbar*k_0/m #nu for analytic value

#Instead of using spdiags, using a convolution via this kernel almost halves the amount of the time of computation.
#This optimises the code.
kernel=np.array([1,-2,1]) #The values of the kernel are the coefficients of finite differences formula


H = 1j*hbar/(2*m) #Hamiltonian

#This functions convolves the kernel with the wavefunction and is used in the ODE solver
def f(t,phi):
    d2phi = np.convolve(phi, kernel, mode='same') / (delta_x ** 2)
    return H*d2phi

def analytic(t): #calcuates the analytic value
    del_t = (hbar / (2 * hbar * (sigma ** 2))) * (t)
    expo = np.exp(-((x - x_0 - nu * t) ** 2)/(2*(sigma**2)*(1+del_t**2)))
    numphi = (1 / (sigma * np.sqrt(2 * pi*(1 + del_t ** 2))))*expo
    return numphi

r = complex_ode(f)

r.set_initial_value(phi_t0,0)
phisq=np.abs(phi_t0)**2
plt.figure()
plt.plot(x,np.abs(r.y)**2)
ax = plt.gca()
plt.ylim([0,max(np.abs(phi_t0)**2)*1.2])
while r.successful() and r.t < 8:
    r.integrate(r.t+0.05)
    # This stacks the value phi squared every time it is calculated. You can use this to do imshow plotting
    phisq=np.vstack((phisq,(np.abs(r.y))**2))
    anfunc = analytic(r.t)
    #Code for the animation
    ax.clear()
    ax.plot(x,np.abs(r.y)**2)
    ax.plot(x, anfunc, 'r.')
    plt.ylim([0, max(np.abs(phi_t0) ** 2) * 1.2])
    ax.legend(['Numerical calculation', 'Analytic calculation'])
    plt.ylabel('Probability')
    plt.xlabel('Displacement (units of distance)')
    plt.pause(0.1)

