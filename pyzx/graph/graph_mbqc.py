from fractions import Fraction
from typing import Tuple, Dict, Set, Any

from .graph_s import GraphS

from ..utils import MeasurementType

class GraphMBQC(GraphS[int,Tuple[int,int]]):
    backend = 'mbqc'

    def __init__(self):
        GraphS.__init__(self)
        self.measurements: Dict[int, MeasurementType.Type] = dict()
    
    def mtype(self, vertex):
        return self.measurements[vertex]
    
    def set_mtype(self, vertex, measurement):
        self.measurements[vertex] = measurement

    def mtypes(self):
        return self.measurements