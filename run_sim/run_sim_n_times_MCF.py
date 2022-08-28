import sys, os, numpy as np, matplotlib.pyplot as plt
from aotools import circle

sys.path.append(os.getcwd().replace("run_sim",""))

import simulation

DIR = 'sim_conf.yaml'

def eval_MCF_plane_src(k, c2n, L, rho):

    arg = -1.46*c2n*(k**2)*L*rho**(5/3)
    return np.exp(arg)

class SimEngine():

     def run(self):
        
        sim = simulation.Sim(DIR)

        sim.aoinit()

        sim.aoloop()
        
        return sim.sciIdx

wvl = 1.55*1e-6
k   = 2*np.pi/wvl
c2n = 1e-14
L   = 700000

run_sim = SimEngine()
MCDOC2 = run_sim.run() 
N = np.shape(MCDOC2)[0]
x = np.linspace(0, 0.004,N)
n = 100

MCF_PLANE = eval_MCF_plane_src(k,c2n,L,x)
means = np.zeros(n)
for i in range(1,n):
    
    mask  = circle(i,N).astype(bool)
    irrad = np.where(mask, MCDOC2, 0)
    means[i-1] = np.mean(irrad, where=mask)

plt.title("Scaled MCF at an altitude of 21 km", fontsize = 25)

plt.plot(1e-3*np.arange(1,n+1),means,'k.', label = 'Simulated')
plt.plot(x, MCF_PLANE, 'b',label = 'Plane wave')
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)
plt.legend(fontsize = 30)
plt.xlabel(r'$\rho$ (m)', fontsize = 30)
plt.ylabel(r'$\Gamma_{2}(\rho,z=21 km)$', fontsize = 30)
plt.show()
