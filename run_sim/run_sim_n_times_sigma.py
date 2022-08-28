import sys, os, numpy as np, matplotlib.pyplot as plt
from aotools import circle

sys.path.append(os.getcwd().replace("run_sim",""))

import simulation

DIR = 'sim_conf.yaml'

class SimEngine():

     def run(self, h):
        
        sim = simulation.Sim(DIR)

        sim.aoinit(h)

        sim.aoloop()
        
        return sim.sciIdx


N   = 10
wvl = 1.55*1e-6
k   = 2*np.pi/wvl
c2n = 1e-14
L   = 1000
z   = np.linspace(1,L,N)

sci_arr = np.zeros(N)
run_sim = SimEngine()

for i in range(N):
    print("z =", z[i])
    sci_arr[i] = run_sim.run(z[i])/5 
    
#plt.title("Scaled MCF at an altitude of 21 km", fontsize = 25)

plt.plot(z,sci_arr,'k.')
'''
plt.plot(x, MCF_PLANE, 'b',label = 'Plane wave')
plt.xticks(fontsize = 30)
plt.yticks(fontsize = 30)
plt.legend(fontsize = 30)
plt.xlabel(r'$\rho$ (m)', fontsize = 30)
plt.ylabel(r'$\Gamma_{2}(\rho,z=21 km)$', fontsize = 30)
'''
plt.show()
