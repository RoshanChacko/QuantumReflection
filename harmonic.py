import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import complex_ode

#%% Defining constants
pi=np.pi
hbar = 1
sigma = 1
m = 1
k_0 = 5
x_0 = -5
N, xb = 1000, 30
x = np.linspace(-xb,xb,N)
delta_x=x[1]-x[0]
e_term = np.exp(-((x-x_0)**2)/(4*(sigma**2)))

phi_t0=(e_term/(np.sqrt(sigma)*((2*pi)**0.25)))*np.exp(1j*k_0*x)

nu = hbar*k_0/m

kernel=np.array([1,-2,1])



V = 0.1*x**2 #defining our harmonic potential, value of k is 0.1

H = 1j*hbar/(2*m)

def f(t,phi):
    d2phi = np.convolve(phi, kernel, mode='same') / (delta_x ** 2)
    return H*d2phi +V*phi/(1j*hbar)


r = complex_ode(f)


r.set_initial_value(phi_t0,0)
phisq=np.abs(phi_t0)**2
plt.figure()
plt.plot(x,np.abs(r.y)**2)
ax = plt.gca()
plt.ylim([0,max(np.abs(phi_t0)**2)*1.2])
while r.successful() and r.t < 10:
    r.integrate(r.t+0.05)
    P = np.abs(r.y)**2
    phisq=np.vstack((phisq,P))
    ax.clear()
    ax.plot(x,P)
    ax.plot(x, V)
    ax.legend(['Wavefunction squared', 'Potential barrier'])
    plt.ylabel('Probability')
    plt.xlabel('Displacement (units of distance)')
    plt.ylim([0, max(np.abs(phi_t0) ** 2) * 3])
    plt.pause(0.1)

plt.figure()
plt.imshow(phisq,extent=[0, 1, 0, 1])

