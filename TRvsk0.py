import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import complex_ode

#%% Defining constants
pi=np.pi
hbar = 1
sigma = 1
m = 1

x_0 = -5
N, xb = 1000, 30
x = np.linspace(-xb,xb,N)
delta_x=x[1]-x[0]
e_term = np.exp(-((x-x_0)**2)/(4*(sigma**2)))


kernel=np.array([1,-2,1])

d, U=1, 2.5

xl, xr, xc = x<-d, x>d, np.logical_and(x>-d, x<d)

V = np.piecewise(x, [np.abs(x)<d/2, np.abs(x)>d/2], [U,0]) #,0.4 height set d= 1

H = 1j*hbar/(2*m)
def f(t,phi):
    d2phi = np.convolve(phi, kernel, mode='same') / (delta_x ** 2)
    return H*d2phi +V*phi/(1j*hbar)



R, T, M, tot_prob=[],[],[],[]

k_0 = np.arange(1,10,0.1)

#We use a similar routine as in discretisation_error.py to get our probabilities vs. k_0

for k in k_0:
    r = complex_ode(f)
    phi_t0=(e_term/(np.sqrt(sigma)*((2*pi)**0.25)))*np.exp(1j*k*x)
    r.set_initial_value(phi_t0,0)

    while r.successful() and r.t < 2:
        r.integrate(r.t+0.05)

    P = np.abs(r.y)**2
    R_area, L_area, C_area = np.trapz(P[xl],x[xl]), np.trapz(P[xr], x[xr]), np.trapz(P[xc], x[xc])
    R = np.hstack((R, R_area))
    T = np.hstack((T, L_area))
    M = np.hstack((M, C_area))

plt.figure() #probabilities plotting
plt.plot(R)
plt.plot(T)
plt.plot(M)
plt.plot(R+T+M)
plt.legend(['Reflected probability','Transmitted probability','Barrier probability','Total probability'])
plt.xlabel('k_0')
plt.ylabel('Probability')

