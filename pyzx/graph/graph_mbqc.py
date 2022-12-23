from .base import VT
from typing import Dict

from .graph_s import GraphS

from ..utils import MeasurementType, VertexType

#TODO: Think about making measurement plane a vdata attribute of GraphS instead of creating a new class
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
        return [n for n in self.neighbors(vertex) if self.mtype(n) != MeasurementType.EFFECT and self.type(n) != VertexType.BOUNDARY]
    
    def effect(self, v):
        for n in self.neighbors(v):
            if self.mtype(n) == MeasurementType.EFFECT:
                return n
        return None

    def mtypes(self):
        return self.measurements

    def effects(self):
        return [v for v,m in self.measurements.items() if m == MeasurementType.EFFECT]

    def mvertices(self):
        return set(self.vertices()).difference(set(self.inputs())).difference(set(self.outputs())).difference(set(self.effects()))

    def minputs(self):
        return set([list(self.neighbors(input))[0] for input in self.inputs()])

    def moutputs(self):
        return set([list(self.neighbors(output))[0] for output in self.outputs()])

    def non_inputs(self):
        return set(self.mvertices()).difference(set(self.minputs()))

    def non_outputs(self):
        return set(self.mvertices()).difference(set(self.moutputs()))

    def is_pauli(self, v):
        return self.mtype(v) in [MeasurementType.X, MeasurementType.Y, MeasurementType.Z]

    def copy(self, adjoint:bool=False):
        g = GraphMBQC()
        g.track_phases = self.track_phases
        g.scalar = self.scalar.copy()
        g.merge_vdata = self.merge_vdata
        mult:int = 1
        if adjoint: mult = -1

        #g.add_vertices(self.num_vertices())
        ty = self.types()
        ph = self.phases()
        qs = self.qubits()
        rs = self.rows()
        maxr = self.depth()
        mp = self.mtypes()
        vtab = dict()
        for v in self.vertices():
            i = g.add_vertex(ty[v],phase=mult*ph[v])
            if v in qs: g.set_qubit(i,qs[v])
            if v in rs: 
                if adjoint: g.set_row(i, maxr-rs[v])
                else: g.set_row(i, rs[v])
            vtab[v] = i
            for k in self.vdata_keys(v):
                g.set_vdata(i, k, self.vdata(v, k))

            if v in mp:
                g.set_mtype(i,mp[v])
            
        for v in self.grounds():
            g.set_ground(vtab[v], True)

        new_inputs = tuple(vtab[i] for i in self.inputs())
        new_outputs = tuple(vtab[i] for i in self.outputs())
        if not adjoint:
            g.set_inputs(new_inputs)
            g.set_outputs(new_outputs)
        else:
            g.set_inputs(new_outputs)
            g.set_outputs(new_inputs)
        
        etab = {e:g.edge(vtab[self.edge_s(e)],vtab[self.edge_t(e)]) for e in self.edges()}
        g.add_edges(etab.values())
        for e,f in etab.items():
            g.set_edge_type(f, self.edge_type(e))
        return g
    
    def remove_vertex(self, vertex: VT) -> None:
        self.remove_vertices([vertex])
        if vertex in self.measurements:
            del self.measurements[vertex]