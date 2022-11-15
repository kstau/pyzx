from fractions import Fraction
from .base import VT, ET
from typing import Tuple, Dict, Set, Any

from .graph_s import GraphS

from ..utils import MeasurementType

class GraphMBQC(GraphS):
    backend = 'mbqc'

    def __init__(self):
        GraphS.__init__(self)
        self.measurements: Dict[int, MeasurementType.Type] = dict()
    
    def mtype(self, vertex):
        if vertex in self.measurements:
            return self.measurements[vertex]
        else:
            return MeasurementType.XY #default
    
    def set_mtype(self, vertex, measurement):
        self.measurements[vertex] = measurement
    
    def mneighbors(self, vertex):
        return [n for n in self.neighbors(vertex) if self.mtype(n) != MeasurementType.EFFECT]
    
    def effect(self, v):
        for n in self.neighbors(v):
            if self.mtype(n) == MeasurementType.EFFECT:
                return n
        return None

    def mtypes(self):
        return self.measurements

    def effects(self):
        return [v for v,m in self.measurements.items() if m == MeasurementType.EFFECT]

    def non_inputs(self):
        return set(self.vertices()).difference(set(self.inputs()))

    def non_outputs(self):
        return set(self.vertices()).difference(set(self.outputs()))