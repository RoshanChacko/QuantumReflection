import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import complex_ode

#%% Defining constants
pi=np.pi
hbar = 1
sigma = 1
m = 1
k_0 = 2.5
x_0 = -5
N, xb = 1000, 30
x = np.linspace(-xb,xb,N)
delta_x=x[1]-x[0]
e_term = np.exp(-((x-x_0)**2)/(4*(sigma**2)))

phi_t0=(e_term/(np.sqrt(sigma)*((2*pi)**0.25)))*np.exp(1j*k_0*x)

nu = hbar*k_0/m

kernel=np.array([1,-2,1])

d, U=1, 2.5

#Here we define the boundaries of the barrier to do our probability calculations
xl, xr, xc = x<-d, x>d, np.logical_and(x>-d, x<d)

#No we define the potential V
V = np.piecewise(x, [np.abs(x)<d/2, np.abs(x)>d/2], [U,0]) #,0.4 height set d= 1

H = 1j*hbar/(2*m)

def f(t,phi):
    d2phi = np.convolve(phi, kernel, mode='same') / (delta_x ** 2)
    return H*d2phi +V*phi/(1j*hbar) #Potential term added


r = complex_ode(f)

#These empty arrays are for storing the probabilites after each loop
R, T, M, tot_prob=[],[],[],[]

r.set_initial_value(phi_t0,0)
phisq=np.abs(phi_t0)**2
plt.figure()
plt.plot(x,np.abs(r.y)**2)
ax = plt.gca()
plt.ylim([0,max(np.abs(phi_t0)**2)*1.2])
while r.successful() and r.t < 8:
    r.integrate(r.t+0.05)
    P = np.abs(r.y)**2
    phisq=np.vstack((phisq,P))
    ax.clear()
    ax.plot(x,P)
    ax.plot(x, V)
    ax.legend(['Wavefunction squared', 'Potential barrier'])
    plt.ylabel('Probability')
    plt.xlabel('Displacement (units of distance)')
    R_area, L_area, C_area = np.trapz(P[xl],x[xl]), np.trapz(P[xr], x[xr]), np.trapz(P[xc], x[xc])
    R = np.hstack((R, R_area)) #Reflection probability
    T = np.hstack((T, L_area)) #Transmission probability
    M = np.hstack((M, C_area)) #Probability in barrier
    tot_prob = np.hstack((tot_prob, R_area+L_area+C_area))
    plt.ylim([0, max(np.abs(phi_t0) ** 2) * 1.2])
    plt.pause(0.1)

plt.figure()
plt.imshow(phisq,extent=[0, 1, 0, 1]) #imshow plotting

plt.figure() #probabilities plotting
plt.plot(R)
plt.plot(T)
plt.plot(M)
plt.plot(R+T+M)
plt.legend(['Reflected probability','Transmitted probability','Barrier probability','Total probability'],loc='upper right')
plt.xlabel('time')
plt.ylabel('Probability')
plt.xticks(np.arange(0, 161, step=20),[0,1,2,3,4,5,6,7,8])