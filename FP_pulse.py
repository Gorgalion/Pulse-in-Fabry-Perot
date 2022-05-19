from math import pi
from math import sqrt
from numpy.core.function_base import linspace
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import complex_ode as ode

# Units in SI
C=299792458

wavelength=600e-9

diamond_length=6e-3

# Initialise diamond with unfolded method
diamond, step=linspace(0, 2*diamond_length, 10, retstep=True)

r1, r2=0.999, 0.999

# Initialize pulse
m=50e-9

sigma=10e-9

energy=50e-6

def pulse(t):
    return energy*np.exp(-1/2*((t-m)/sigma)**2)/sqrt(2*pi)/sigma

# Equations
def dfuncdt(t, F):

    retval=[0]*(len(diamond))

    fourier=np.fft.fft(F)

    freq=np.fft.fftfreq(diamond.shape[-1])

    for i in range(len(freq)):
        freq[i]=freq[i]+step/wavelength

    future=0

    for m in range(len(freq)):
        future=future+fourier[m]*np.exp(1j*2*pi*2*diamond_length*freq[m]/step)

    f0=pulse(t)+r1*r2*np.exp(1j*np.angle(future))*abs(F[-1])

    retval[0]=-(F[0]-f0)*C/step

    for i in range(len(diamond)-1):
        retval[i+1]=-(F[i+1]-F[i])*C/step

    return retval

t0=0
F0=[0]*(len(diamond))

solver=ode(dfuncdt)

solver.set_integrator('dopri5')

solver.set_initial_value(F0, t0)

t=np.linspace(0, 1e-6, 1000)

sol=[]

sol.append(0)

k=1

while solver.successful() and solver.t <t[-1]:
    solver.integrate(t[k])
    sol.append((1-r1*r2)*solver.y[-1])
    solver.set_initial_value(solver.y, solver.t)
    k=k+1

# f=open('output_pulse.txt', 'w')

# f.write('Time\tInput Pump\tOutput'+'\n')

# for i in range(len(t)):
#     f.write(str(t[i])+'\t'+str(pulse(t[i]))+'\t'+str(sol[i])+'\n')

# f.close()

print(sum(np.real(sol))+sum(np.imag(sol)))
print(sum(pulse(t)))

fig, ax=plt.subplots(1, 2)

ax[0].plot(t, pulse(t))
ax[0].plot(t, np.abs(sol))

fourier=np.fft.fft(pulse(t))
freq=np.fft.fftfreq(t.shape[-1])

ax[1].plot(freq, np.abs(fourier))

fourier=np.fft.fft(np.abs(sol))
freq=np.fft.fftfreq(t.shape[-1])

ax[1].plot(freq, np.abs(fourier))

plt.show()