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
from neo.io import PyNNTextIO

simulator_name = get_script_args(1)[0]  
sim = import_module("pyNN.%s" % simulator_name)

tstop = 500.0
time_step = 0.005

sim.setup(timestep=time_step, debug=True)
    
cell_params1 = {'tau_refrac':10,'v_thresh':-52.0, 'v_reset':-62.0, 'i_offset': 0.9, 'tau_syn_E'  : 2.0, 'tau_syn_I': 5.0}
pop_IF_curr_alpha = sim.Population(1, sim.IF_curr_alpha(**cell_params1), label="pop_IF_curr_alpha")
pop_IF_curr_alpha.record('v')
pop_IF_curr_alpha.record('spikes')
#pop_IF_curr_alpha.initialize('v', -67)

cell_params2 = {'tau_refrac':8,'v_thresh':-50.0, 'v_reset':-70.0, 'i_offset': 1, 'tau_syn_E'  : 2.0, 'tau_syn_I': 5.0}
pop_IF_curr_exp = sim.Population(1, sim.IF_curr_exp(**cell_params2), label="pop_IF_curr_exp")
pop_IF_curr_exp.record('v')
#pop_IF_curr_exp.initialize('v', -68)

cell_params3 = {'tau_refrac':5,'v_thresh':-50.0, 'v_reset':-65.0, 'i_offset': 0.9, 'tau_syn_E'  : 2.0, 'tau_syn_I': 5.0}
pop_IF_cond_alpha = sim.Population(1, sim.IF_cond_alpha(**cell_params3), label="pop_IF_cond_alpha")
pop_IF_cond_alpha.record('v')
#pop_IF_cond_alpha.initialize('v', -69)

cell_params4 = {'tau_refrac':5,'v_thresh':-52.0, 'v_reset':-68.0, 'i_offset': 1, 'tau_syn_E'  : 2.0, 'tau_syn_I': 5.0}
pop_IF_cond_exp = sim.Population(1, sim.IF_cond_exp(**cell_params4), label="pop_IF_cond_exp")
pop_IF_cond_exp.record('v')
pop_IF_cond_exp.record('spikes')
#pop_IF_cond_exp.initialize('v', -70)

##TODO: Test a>0!!
cell_params5 = {'tau_refrac':0,'v_thresh':-52.0, 'v_reset':-68.0, 'i_offset': 0.6, 'v_spike': -40, 'a': 0.0, 'b':0.0805}
pop_EIF_cond_exp_isfa_ista = sim.Population(1, sim.EIF_cond_exp_isfa_ista(**cell_params5), label="pop_EIF_cond_exp_isfa_ista")
pop_EIF_cond_exp_isfa_ista.record('v')
pop_EIF_cond_exp_isfa_ista.record('spikes')


cell_params6 = {'i_offset': 0.2, 'gbar_K':6.0, 'gbar_Na':20.0}
pop_HH_cond_exp = sim.Population(1, sim.HH_cond_exp(**cell_params6), label="pop_HH_cond_exp")
pop_HH_cond_exp.record('v')
#pop_HH_cond_exp.record('spikes')

# Post synaptic cells
cell_params_post1 = {'tau_refrac':10,'v_thresh':-52.0, 'v_reset':-62.0, 'i_offset': 0, 'tau_syn_E'  : 5.0}
pop_post1 = sim.Population(1, sim.IF_cond_exp(**cell_params_post1), label="pop_post1")
pop_post1.record('v')
pop_post1.record('gsyn_exc')
cell_params_post2 = {'tau_refrac':10,'v_thresh':-52.0, 'v_reset':-62.0, 'i_offset': 0, 'tau_syn_E'  : 5.0}
pop_post2 = sim.Population(1, sim.IF_cond_alpha(**cell_params_post2), label="pop_post2")
pop_post2.record('v')
pop_post2.record('gsyn_exc')

connE = sim.connect(pop_EIF_cond_exp_isfa_ista, pop_post1, weight=0.01, receptor_type='excitatory',delay=10)
connE = sim.connect(pop_EIF_cond_exp_isfa_ista, pop_post2, weight=0.005, receptor_type='excitatory',delay=20)

sim.run(tstop)

use_hdf5 = False

if use_hdf5:
    from neo.io import NeoHdf5IO as NeoIO
    suffix = 'h5'

    results_file = "Results/NeuroMLTest_%s.%s" % (simulator_name, suffix)
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

    for pop in [pop_IF_curr_alpha, pop_IF_curr_exp, pop_IF_cond_exp, pop_IF_cond_alpha,pop_EIF_cond_exp_isfa_ista, pop_HH_cond_exp, pop_post1,pop_post2]:
        data =  pop.get_data('v', gather=False)
        filename = "%s_v.dat"%(pop.label)
        print("Writing data for %s"%pop)
        for segment in data.segments:
            vm = segment.analogsignalarrays[0].transpose()[0]
            tt = np.array([t*time_step/1000. for t in range(len(vm))])
            times_vm = np.array([tt, vm/1000.]).transpose()
            np.savetxt(filename, times_vm , delimiter = '\t', fmt='%s')
        filename = "%s.spikes"%(pop.label)
        io = PyNNTextIO(filename=filename)
        spikes =  pop.get_data('spikes', gather=False)
        if len(spikes.segments[0].spiketrains)>0:
            print("Writing data for %s"%pop)
            for segment in spikes.segments:
                io.write_segment(segment)


sim.end()

if '-gui' in sys.argv:
    if simulator_name in ['neuron', 'nest', 'brian']:
        import matplotlib.pyplot as plt
        
        print("Plotting results of simulation in %s"%simulator_name)

        plt.figure("Voltages for IaF cells")
        for pop in [pop_IF_curr_alpha, pop_IF_curr_exp, pop_IF_cond_exp, pop_IF_cond_alpha]:
            data = pop.get_data()
            vm = data.segments[0].analogsignalarrays[0]
            plt.plot(vm, '-', label='%s: v'%pop.label)
            
        plt.legend()
        
        plt.figure("Voltages for EIF & HH cells")
        for pop in [pop_EIF_cond_exp_isfa_ista, pop_HH_cond_exp]:
            data = pop.get_data()
            vm = data.segments[0].analogsignalarrays[0]
            plt.plot(vm, '-', label='%s: v'%pop.label)

        plt.legend()

        plt.figure("Voltages for postsynaptic cells")
        for pop in [pop_post1, pop_post2]:
            data = pop.get_data()
            vm = data.segments[0].analogsignalarrays[1]
            plt.plot(vm, '-', label='%s: v'%pop.label)
            
        plt.legend()

        plt.figure("Conductances for syns on postsynaptic cells")
        for pop in [pop_post1, pop_post2]:
            data = pop.get_data()
            gsyn = data.segments[0].analogsignalarrays[0]
            plt.plot(gsyn, '-', label='%s: gsyn'%pop.label)
            
        plt.legend()


        plt.show()
        
else:

    print("Simulation completed.\nRun this script with flag: -gui to plot activity of a number of cells")


