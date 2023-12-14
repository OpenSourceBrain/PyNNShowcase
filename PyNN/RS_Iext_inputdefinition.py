'''
PyNN simulator compliant export for:

-- Component, id = RS_Iext, type: pulseGenerator ----------
    Parameters:
        delay = 0.0 (SI time)
        duration = 0.52 (SI time)
        amplitude = 1.0E-10 (SI current)
---------------------------------------------

    This PyNN file has been generated by org.neuroml.export (see https://github.com/NeuroML/org.neuroml.export)
         org.neuroml.export  v1.10.0
         org.neuroml.model   v1.10.0
         jLEMS               v0.11.0

'''
# Input file for Component: RS_Iext

print("Loading input file for RS_Iext")

#from pyNN.neuron import NativeCellType
from neuron import h


class RS_Iext(object):
     
    all_inputs = {}
    parameters = {}

    def __init__(self, **parameters):
        print("Created instance of RS_Iext with params: %s"%parameters)
        self.parameters = parameters

    def inject_into(self, cells):
        print("Injecting RS_Iext into: %s"%cells)
        for id in cells:
            if id.local:
                self.all_inputs[id] = h.RS_Iext(0.5, sec=id._cell.source_section)
                #h('forall psection()')