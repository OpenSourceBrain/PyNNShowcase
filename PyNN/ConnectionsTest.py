"""
Simple network for testing export to NeuroML v1 & v2

"""
import logging
logging.basicConfig(format='%(levelname)s - %(name)s: %(message)s', level=logging.DEBUG)

import sys
import os
from importlib import import_module
import numpy as np

from pyNN.utility import get_script_args
import pyNN.space as space

simulator_name = get_script_args(1)[0]  
sim = import_module("pyNN.%s" % simulator_name)

tstop = 500.0
time_step = 0.005

sim.setup(timestep=time_step, debug=True, reference='ConnectionsTest')


sphere1 = space.Sphere(radius=100.0)
struct1 = space.RandomStructure(sphere1, origin=(0.0, 0.0, 0.0))

cell_params = {'tau_refrac':5,'v_thresh':-50.0, 'v_reset':-65.0, 'i_offset': 0.9, 'tau_syn_E'  : 2.0, 'tau_syn_I': 5.0}
pop_pre = sim.Population(5, sim.IF_cond_alpha(**cell_params), label="pop_pre",structure=struct1)
pop_pre.record('v')
pop_pre.annotate(radius=5)
pop_pre.annotate(color='0 0.6 0')

sphere2 = space.Sphere(radius=100.0)
struct2 = space.RandomStructure(sphere2, origin=(0.0, 200.0, 0.0))
pop_post = sim.Population(5, sim.IF_cond_alpha(**cell_params), label="pop_post",structure=struct2)
pop_post.record('v')
pop_post.annotate(radius=5)
pop_post.annotate(color='0 0.2 0.6')

pre_selection = sim.PopulationView(pop_pre, np.array([1,3]),label='pre_selection')
print("Creating view:  %s"%pre_selection)
post_selection = sim.PopulationView(pop_post, np.array([0,2,4]),label='post_selection')
print("Creating view:  %s"%post_selection)

###################################################################
# Projection less connection
connE = sim.connect(pop_pre, pop_post, weight=0.01, receptor_type='excitatory', delay=10)


###################################################################
# AllToAllConnector connection
proj1 = sim.Projection(pop_pre, pop_post, sim.AllToAllConnector(),
                                    sim.StaticSynapse(weight=0.05, delay=5),label="AllToAllConnectorProj")

###################################################################
# OneToOneConnector connection
proj2 = sim.Projection(pop_pre, pop_post, sim.OneToOneConnector(),
                                    sim.StaticSynapse(weight=0.02, delay=4),label="OneToOneConnector")

###################################################################
# FixedProbabilityConnector connection
proj3 = sim.Projection(pop_pre, pop_post, sim.FixedProbabilityConnector(p_connect=0.2),
                                    sim.StaticSynapse(weight=0.1, delay=1),label="FixedProbabilityConnector")
                                

###################################################################
# AllToAllConnector between pre_selection & post_selection
proj1 = sim.Projection(pre_selection, post_selection, sim.AllToAllConnector(),
                                    sim.StaticSynapse(weight=0.05, delay=5),label="AllToAllConnectorProjSelection")

sim.run(tstop)

use_hdf5 = False

if use_hdf5:
    from neo.io import NeoHdf5IO as NeoIO
    suffix = 'h5'

    results_file = "Results/ConnectionTest_%s.%s" % (simulator_name, suffix)
    if os.path.exists(results_file):
        os.remove(results_file)
    io = NeoIO(results_file)
    pop_IF_curr_alpha.write_data(io)
    pop_IF_curr_exp.write_data(io)
    pop_IF_cond_alpha.write_data(io)
    pop_IF_cond_exp.write_data(io)
    pop_EIF_cond_exp_isfa_ista.write_data(io)
    pop_HH_cond_exp.write_data(io)
    pop_post1.write_data(io)
    pop_post2.write_data(io)

else:
    #from neo.io import AsciiSignalIO as NeoIO
    #suffix = 'txt'
    #results_file = "Results/NeuroMLTest_%s.%s" % (simulator_name, suffix)

    for pop in [pop_pre, pop_post]:
        data =  pop.get_data('v', gather=False)
        filename = "%s_v.dat"%(pop.label)
        print("Writing data for %s"%pop)
        for segment in data.segments :
            vm = segment.analogsignals[0].transpose()[0]
            tt = np.array([t*time_step/1000. for t in range(len(vm))])
            times_vm = np.array([tt, vm/1000.]).transpose()
            np.savetxt(filename, times_vm , delimiter = '\t', fmt='%s')


sim.end()

if '-gui' in sys.argv:
    if simulator_name in ['neuron', 'nest', 'brian']:
        import matplotlib.pyplot as plt
        
        print("Plotting results of simulation in %s"%simulator_name)

        plt.figure("Voltages for IaF cells")
        for pop in [pop_pre, pop_post]:
            data = pop.get_data()
            vm = data.segments[0].analogsignals[0]
            plt.plot(vm, '-', label='%s: v'%pop.label)
            
        plt.legend()
        



        plt.show()



