'''
PyNN simulator compliant export for:

-- Component, id = RS, type: izhikevich2007Cell ----------
    Parameters:
        v0 = -0.06 (SI voltage)
        k = 7.0E-7 (SI conductance_per_voltage)
        vr = -0.06 (SI voltage)
        vt = -0.04 (SI voltage)
        vpeak = 0.035 (SI voltage)
        a = 30.0 (SI per_time)
        b = -2.0E-9 (SI conductance)
        c = -0.05 (SI voltage)
        d = 1.0E-10 (SI current)
        C = 1.0E-10 (SI capacitance)
---------------------------------------------

    This PyNN file has been generated by org.neuroml.export (see https://github.com/NeuroML/org.neuroml.export)
         org.neuroml.export  v1.5.2
         org.neuroml.model   v1.5.2
         jLEMS               v0.9.8.9

'''
# Cell file for Component: RS

print("Loading cell file for RS")

from pyNN.neuron import NativeCellType
from neuron import h
from neuron import nrn


class RS(object):

    def __init__(self, **parameters):
        print("Created instance of RS with params: %s"%parameters)

        self.soma = nrn.Section('sec_RS')
        #h(" forall psection() ")

        self.soma.L = 10.0
        self.soma(0.5).diam = 10.0
        # Todo: work this out here from area etc.
        self.soma(0.5).cm = (318309*parameters['C'] if parameters.has_key('C') else 318.31927 )
        self.soma.push()
        self.mechanism = h.RS(0.5, sec=self.soma)

        self.mechanism.v0 = parameters['v0']
        self.mechanism.k = parameters['k']
        self.mechanism.vr = parameters['vr']
        self.mechanism.vt = parameters['vt']
        self.mechanism.vpeak = parameters['vpeak']
        self.mechanism.a = parameters['a']
        self.mechanism.b = parameters['b']
        self.mechanism.c = parameters['c']
        self.mechanism.d = parameters['d']
        self.mechanism.C = parameters['C']

        self.source = self.soma(0.5)._ref_v

        # needed for PyNN
        self.source_section = self.soma
        self.parameter_names = ( 'v0',  'k',  'vr',  'vt',  'vpeak',  'a',  'b',  'c',  'd',  'C', )
        self.traces = {}
        self.recording_time = False

    # Getter/setter for v0
    def _set_v0(self, value):
        print("Setting v0 in RS to %s"%value)
        self.mechanism.v0 = value
    def _get_v0(self):
        return self.mechanism.v0
    v0 = property(fget=_get_v0, fset=_set_v0)

    # Getter/setter for k
    def _set_k(self, value):
        print("Setting k in RS to %s"%value)
        self.mechanism.k = value
    def _get_k(self):
        return self.mechanism.k
    k = property(fget=_get_k, fset=_set_k)

    # Getter/setter for vr
    def _set_vr(self, value):
        print("Setting vr in RS to %s"%value)
        self.mechanism.vr = value
    def _get_vr(self):
        return self.mechanism.vr
    vr = property(fget=_get_vr, fset=_set_vr)

    # Getter/setter for vt
    def _set_vt(self, value):
        print("Setting vt in RS to %s"%value)
        self.mechanism.vt = value
    def _get_vt(self):
        return self.mechanism.vt
    vt = property(fget=_get_vt, fset=_set_vt)

    # Getter/setter for vpeak
    def _set_vpeak(self, value):
        print("Setting vpeak in RS to %s"%value)
        self.mechanism.vpeak = value
    def _get_vpeak(self):
        return self.mechanism.vpeak
    vpeak = property(fget=_get_vpeak, fset=_set_vpeak)

    # Getter/setter for a
    def _set_a(self, value):
        print("Setting a in RS to %s"%value)
        self.mechanism.a = value
    def _get_a(self):
        return self.mechanism.a
    a = property(fget=_get_a, fset=_set_a)

    # Getter/setter for b
    def _set_b(self, value):
        print("Setting b in RS to %s"%value)
        self.mechanism.b = value
    def _get_b(self):
        return self.mechanism.b
    b = property(fget=_get_b, fset=_set_b)

    # Getter/setter for c
    def _set_c(self, value):
        print("Setting c in RS to %s"%value)
        self.mechanism.c = value
    def _get_c(self):
        return self.mechanism.c
    c = property(fget=_get_c, fset=_set_c)

    # Getter/setter for d
    def _set_d(self, value):
        print("Setting d in RS to %s"%value)
        self.mechanism.d = value
    def _get_d(self):
        return self.mechanism.d
    d = property(fget=_get_d, fset=_set_d)

    # Getter/setter for C
    def _set_C(self, value):
        print("Setting C in RS to %s"%value)
        self.mechanism.C = value
    def _get_C(self):
        return self.mechanism.C
    C = property(fget=_get_C, fset=_set_C)


    def memb_init(self):

        # Initialising state v
        for seg in self.soma:
            seg.v = self.get_value('v0')
        # Initialising state u
        self.mechanism.u = self.get_value('0')


    def get_value(self, val_string):
        if val_string=='v':
            return self.soma(0.5)._ref_v
        if val_string.isdigit():
            return float(val_string) 
        else:
            return getattr(self.mechanism,val_string)


class RSType(NativeCellType):

    default_parameters = { 'v0':-60.0,  'k':7.0E-4,  'vr':-60.0,  'vt':-40.0,  'vpeak':35.0,  'a':0.030000001,  'b':-0.002,  'c':-50.0,  'd':0.1,  'C':1.0E-4, }
    default_initial_values = {}    
    recordable = [ 'soma(0.5).v',  'soma(0.5).u', ]
    units = {'soma(0.5).v' : 'mV'}
    # Synapses allowed: $synapses_allowed
    receptor_types = []
    model = RS

