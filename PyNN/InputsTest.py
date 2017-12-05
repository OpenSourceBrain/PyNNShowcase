"""
Simple network for testing export to NeuroML v2

"""
import logging
logging.basicConfig(format='%(levelname)s - %(name)s: %(message)s', level=logging.DEBUG)

import sys
from importlib import import_module
import numpy as np

from pyNN.utility import get_script_args

simulator_name = get_script_args(1)[0]  
sim = import_module("pyNN.%s" % simulator_name)

tstop = 500.0
time_step = 0.005

sim.setup(timestep=time_step, debug=True,reference="Inputs",save_format='xml')

pop_size = 2

cell_params = {'tau_refrac':5,'v_thresh':-50.0, 'v_reset':-65.0, 'i_offset': 0.1, 'tau_syn_E'  : 2.0, 'tau_syn_I': 5.0}

all_pops = []
pop_pre = sim.Population(pop_size, sim.IF_cond_alpha(**cell_params), label="pop_pre")
pop_pre.record('v')
all_pops.append(pop_pre)

pulse = sim.DCSource(amplitude=0.9, start=100, stop=300.0)
#pulse.inject_into(pop_pre)
pop_pre[0].inject(pulse)

'''
pop_post = sim.Population(pop_size, sim.IF_cond_alpha(**cell_params), label="pop_post",structure=struct)
pop_post.record('v')
all_pops.append(pop_post)'''


sim.run(tstop)


for pop in all_pops:
    for i in range(len(pop)):
        filename = "%s_%s_v.dat"%(pop.label,i)
        print("Writing data for %s[%s]"%(pop,i))
        data =  pop.get_data('v', gather=False)
        for segment in data.segments:
            vm = segment.analogsignals[0].transpose()[i]
            tt = np.array([t*sim.get_time_step()/1000. for t in range(len(vm))])
            times_vm = np.array([tt, vm/1000.]).transpose()
            np.savetxt(filename, times_vm , delimiter = '\t', fmt='%s')


sim.end()

if '-gui' in sys.argv:
    if simulator_name in ['neuron', 'nest', 'brian']:
        import matplotlib.pyplot as plt
        
        print("Plotting results of simulation in %s"%simulator_name)

        plt.figure("Voltages for IaF cells")
        for pop in all_pops:
            data = pop.get_data()
            vm = data.segments[0].analogsignals[0]
            plt.plot(vm, '-', label='%s: v'%pop.label)
            
        plt.legend()

        plt.show()


