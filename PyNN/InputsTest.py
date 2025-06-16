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
time_step = 0.05

sim.setup(timestep=time_step, debug=True,reference="Inputs",save_format='xml')

pop_size = 4

cell_params = {'tau_refrac':5,'v_thresh':-50.0, 'v_reset':-65.0, 'i_offset': 0.1, 'tau_syn_E'  : 2.0, 'tau_syn_I': 5.0}

all_pops = []
pop_pre = sim.Population(pop_size, sim.IF_cond_alpha(**cell_params), label="pop_pre")
pop_pre.record('v')
pop_pre.record('spikes')
all_pops.append(pop_pre)

dcs = sim.DCSource(amplitude=0.9, start=50, stop=400.0)
pop_pre[0].inject(dcs)

acs = sim.ACSource(start=50.0, stop=450.0, amplitude=1.0, offset=1.0,
                   frequency=10.0, phase=180.0)
pop_pre[1].inject(acs)

scs = sim.StepCurrentSource(times=[50.0, 110.0, 350.0, 410.0],
                        amplitudes=[0.4, 0.9, -0.2, 0.2])
pop_pre[2].inject(scs)

#noise = sim.NoisyCurrentSource(mean=0.9, stdev=1.0, start=50.0, stop=450.0, dt=1.0)
#pop_pre[3].inject(noise)
                        
    

sim.run(tstop)

for pop in all_pops:
    data =  pop.get_data('v', gather=False)
    analogsignal = data.segments[0].analogsignals[0]
    name = analogsignal.name

    def get_source_ids_as(analogsignal):
        if 'source_ids' in analogsignal.annotations:
            return analogsignal.annotations['source_ids']
        elif 'channel_ids' in analogsignal.annotations: # See https://github.com/NeuralEnsemble/PyNN/pull/762
            return analogsignal.annotations['channel_ids']
            
    source_ids = get_source_ids_as(analogsignal)
    filename = "%s_%s.dat"%(pop.label,name)
    print('Saving data recorded for %s in pop %s, global ids: %s to %s'%(name, pop.label, source_ids, filename))
    times_vm_a = []
    for i in range(len(source_ids)):
        glob_id = source_ids[i]
        index_in_pop = pop.id_to_index(glob_id)
        print("Writing data for cell %i = %s[%s] (gid: %i) to %s "%(i, pop.label,index_in_pop, glob_id, filename))
        vm = analogsignal.transpose()[i]
        if len(times_vm_a)==0:
            tt = np.array([t*sim.get_time_step()/1000. for t in range(len(vm))])
            times_vm_a.append(tt)
        times_vm_a.append(vm/1000.)
        
    times_vm = np.array(times_vm_a).transpose()
    np.savetxt(filename, times_vm , delimiter = '\t', fmt='%s')
            
    data =  pop.get_data('spikes', gather=False)
    spiketrains = data.segments[0].spiketrains
        
    filename = "%s.spikes"%(pop.label)
    ff = open(filename, 'w')

    def get_source_id(spiketrain):
        if 'source_id' in spiketrain.annotations:
            return spiketrain.annotations['source_id']
            
        elif 'channel_id' in spiketrain.annotations: # See https://github.com/NeuralEnsemble/PyNN/pull/762
            return spiketrain.annotations['channel_id']

    print('Saving data recorded for spikes in pop %s, indices: %s to %s'%(pop.label, [get_source_id(s) for s in spiketrains], filename))
    for spiketrain in spiketrains:
        source_id = get_source_id(spiketrain)
        source_index = pop.id_to_index(source_id)
        print("Writing spike data for cell %s[%s] (gid: %i): %s "%(pop.label,source_index, source_id, spiketrain))
        for t in spiketrain:
            ff.write('%s\t%i\n'%(t.magnitude/1000.,source_index))
    ff.close()


sim.end()

if '-gui' in sys.argv:
    if simulator_name in ['neuron', 'nest', 'brian2']:
        import matplotlib.pyplot as plt
        
        print("Plotting results of simulation in %s"%simulator_name)

        for pop in all_pops:
            data = pop.get_data()
            vms = data.segments[0].analogsignals[0].transpose()
            for i, vm in enumerate(vms):
                plt.figure("Voltages for cell %i in %s, simulator: %s"%(i, pop.label,simulator_name))
                plt.plot(vm, '-', label='%s[%i]: v'%(pop.label,i))
                plt.legend()

        plt.show()

else:

    print("Simulation completed.\nRun this script with flag: -gui to plot activity of a number of cells")


