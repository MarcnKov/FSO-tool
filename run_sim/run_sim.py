import sys, os

sys.path.append(os.getcwd().replace("run_sim",""))

import simulation

DIR = 'sim_conf.yaml'

class SimEngine():

     def run(self):
        
        sim = simulation.Sim(DIR)

        sim.aoinit()

        sim.aoloop()
        
run_sim = SimEngine()
run_sim.run()

