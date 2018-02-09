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

sim.setup(timestep=time_step, debug=True,reference="Positions",save_format='hdf5')

pop_size = 400

cell_params = {'tau_refrac':5,'v_thresh':-50.0, 'v_reset':-65.0, 'i_offset': 0.9, 'tau_syn_E'  : 2.0, 'tau_syn_I': 5.0}

sphere = space.Sphere(radius=100.0)
struct = space.RandomStructure(sphere, origin=(0.0, 100.0, 0.0))

pop_pre = sim.Population(pop_size, sim.IF_cond_alpha(**cell_params), label="pop_pre",structure=struct)
pop_pre.record('v')
pop_pre.annotate(radius=5)
pop_pre.annotate(color='0 0.6 0')

cuboid = space.Cuboid(30,40,50)
struct = space.RandomStructure(cuboid, origin=(-200.0, 0.0, -200.0))

pop_post = sim.Population(pop_size, sim.IF_cond_alpha(**cell_params), label="pop_post",structure=struct)
pop_post.record('v')
pop_post.annotate(radius=5)
pop_post.annotate(color='0.2 0.2 1')



sim.run(tstop)

for pop in [pop_pre, pop_post]:
    print("Positions of %s: %s"%(pop.label,pop.positions))


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



