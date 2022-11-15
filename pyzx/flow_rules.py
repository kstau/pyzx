
from .graph.graph_mbqc import GraphMBQC
from .graph.base import VT
from .utils import MeasurementType, VertexType, EdgeType
from fractions import Fraction

"""
Graph theoretic local complementation on graph-like diagram
This method does not eliminate Clifford spiders, but introduces XZ and YZ measurement effects as in https://arxiv.org/pdf/2003.01664.pdf
g: A MBQCGraph instance
v: The vertex to complement
inverse: Whether to use angle +pi/2 or -pi/2 for complementing v.
"""
def lcomp(g: GraphMBQC, v: VT, inverse = None):
    vn = list(g.mneighbors(v))
    vn.sort()
    # default use -pi/2 for every angle > pi, thus we are more likely to get angles cancel out to 0
    if not inverse:
        inverse = g.phase(v) >= 1 

    for n in vn:
        # flip edges
        for n2 in vn[vn.index(n)+1:]:
            if g.connected(n,n2):
                g.remove_edge(g.edge(n,n2))
            else:
                g.add_edge(g.edge(n,n2), EdgeType.HADAMARD)
        
        # set new measurement plane and angle for neighbors
        g.set_mtype(n, {
            MeasurementType.YZ: MeasurementType.XZ,
            MeasurementType.XZ: MeasurementType.YZ,
        }.get(g.mtype(n), MeasurementType.XY))

        g.set_phase(n, g.phase(n)+Fraction(1,2) if inverse else g.phase(n)-Fraction(1,2))
    
    # set new measurement plane of v
    g.set_mtype(v, {
        MeasurementType.XY: MeasurementType.XZ,
        MeasurementType.XZ: MeasurementType.XY,
    }.get(g.mtype(v), MeasurementType.YZ))
    
    # update phases of v (and its effect) according to measurement plane
    if g.mtype(v) == MeasurementType.XZ:
        # add new effect spider
        newv = g.add_vertex(VertexType.Z, -1, g.row(v), g.phase(v)+Fraction(1,2) if inverse else g.phase(v)-Fraction(1,2))
        g.set_mtype(newv, MeasurementType.EFFECT)
        g.add_edge(g.edge(v, newv), EdgeType.HADAMARD)
        g.set_phase(v, Fraction(1,2) if inverse else -Fraction(1,2))

    elif g.mtype(v) == MeasurementType.XY:
        e = g.effect(v)
        g.set_phase(v, g.phase(e)+Fraction(1,2) if inverse else g.phase(e)-Fraction(1,2))
        g.remove_vertex(e)

    else:
        e = g.effect(v)
        g.set_phase(e, g.phase(e)+Fraction(1,2))

"""
Graph theoretic pivot on graph-like diagram
"""
def pivot(g: GraphMBQC, u: VT, v: VT):
    lcomp(g, u)
    lcomp(g, v)
    lcomp(g, u)
