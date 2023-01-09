
from .graph.graph_mbqc import GraphMBQC
from typing import List
from .graph.base import VT
from .utils import MeasurementType, VertexType, EdgeType, insert_identity, toggle_edge
from fractions import Fraction
from .drawing import draw

from .tensor import compare_tensors

"""Pauli flow preserving rewrite rules for graph-like diagrams"""


def lcomp(g: GraphMBQC, v: VT):
    """
    Graph theoretic local complementation on graph-like diagram
    This method does not eliminate Clifford spiders, but introduces XZ and YZ measurement effects as in https://arxiv.org/pdf/2003.01664.pdf
    Similar to the classic rewrite rule this rule would also work with every phase addition inverted, however we ommitted that due to simplicity
    g: A MBQCGraph instance
    v: The vertex to complement
    """
    vn = list(g.mneighbors(v))
    if g.type(v) != VertexType.Z or g.mtype(v) == MeasurementType.EFFECT or any([n for n in g.neighbors(v) if g.type(n) != VertexType.Z]):
        return False
    vn.sort()

    for n in vn:
        # flip edges
        for n2 in vn[vn.index(n)+1:]:
            if g.connected(n,n2):
                g.remove_edge(g.edge(n,n2))
            else:
                g.add_edge(g.edge(n,n2), EdgeType.HADAMARD)
        
        lcomp_update_neighbor(g, v, n)
    
    lcomp_update_vertex(g, v)

    return True

def lcomp_update_vertex(g: GraphMBQC, v: VT):
    """Updates phase and measurement type of a complemented vertex"""
    phase_to_add = -Fraction(1,2)

    if g.mtype(v) == MeasurementType.XZ and g.phase(v) == Fraction(3,2):
        phase_to_add = Fraction(1,2)

    elif g.mtype(v) == MeasurementType.YZ and g.phase(v) == 0:
        phase_to_add = Fraction(1,2)

    if g.mtype(v) in [MeasurementType.XY, MeasurementType.Y]:
        #create effect spider
        newv = g.add_vertex(VertexType.Z, -1, g.row(v), g.phase(v) + phase_to_add)
        g.add_edge(g.edge(v, newv), EdgeType.HADAMARD)
        g.set_mtype(newv, MeasurementType.EFFECT)

        if g.mtype(v) == MeasurementType.XY:
            g.set_phase(v, Fraction(1,2))
            g.set_phase(newv, -g.phase(newv))
        else:
            g.set_phase(v, 0)
    
    elif g.mtype(v) in [MeasurementType.XZ, MeasurementType.Z]:
        # remove effect spider
        g.set_phase(v, g.phase(g.effect(v))+phase_to_add)
        g.remove_vertex(g.effect(v))
    
    elif g.mtype(v) == MeasurementType.YZ: 
        g.set_phase(g.effect(v), g.phase(g.effect(v))+phase_to_add)

    #set new measurement plane/axis
    g.set_mtype(v, {
        MeasurementType.XY: MeasurementType.XZ,
        MeasurementType.XZ: MeasurementType.XY,
        MeasurementType.X: MeasurementType.X,
        MeasurementType.Y: MeasurementType.Z,
        MeasurementType.Z: MeasurementType.Y
    }.get(g.mtype(v), MeasurementType.YZ))


def lcomp_update_neighbor(g: GraphMBQC, v: VT, n: VT):
    """Updates phase and measurement type of a neighbor of a complemented vertex"""
    if g.mtype(v) == MeasurementType.XZ and g.phase(v) == Fraction(3,2):
        g.add_to_phase(n,Fraction(1,2))
    else:
        g.add_to_phase(n,-Fraction(1,2))
    g.set_mtype(n, {
        MeasurementType.YZ: MeasurementType.XZ,
        MeasurementType.XZ: MeasurementType.YZ,
        MeasurementType.X: MeasurementType.Y,
        MeasurementType.Y: MeasurementType.X,
        MeasurementType.Z: MeasurementType.Z
    }.get(g.mtype(n), MeasurementType.XY))


def pivot(g: GraphMBQC, u: VT, v: VT) -> bool:
    """Graph theoretic pivot on graph-like diagram
    g: A MBQCGraph instance
    u: First vertex
    v: Second vertex"""
    if not lcomp(g, u):
        return False
    success = lcomp(g, v)
    lcomp(g, u)
    return success

def z_delete(g: GraphMBQC, vertex: VT) -> bool:
    """Deletes a Z measured vertex from graph-like diagram according to Lemma 5.4 in https://arxiv.org/pdf/2109.05654.pdf"""
    if g.mtype(vertex) == MeasurementType.Z or (g.mtype(vertex) in [MeasurementType.XZ, MeasurementType.YZ] and g.phase(vertex) in [0,1]):
        if not any([n for n in g.neighbors(vertex) if g.type(n) != VertexType.Z]):
            e = g.effect(vertex)
            if g.phase(e) == 1:
                for n in g.neighbors(vertex):
                    g.add_to_phase(n,1)
            g.remove_vertex(e)
            g.remove_vertex(vertex)
            return True
    return False

def z_insert(g: GraphMBQC, neighbors: List[VT]):
    """Inserts a Z measured vertex anywhere in the diagram"""
    if any([n for n in neighbors if g.type(n) != VertexType.Z]):
        return False
    position = int(sum([g.row(n) for n in neighbors])/len(neighbors))
    v = g.add_vertex(VertexType.Z, -1, position, 0)
    e = g.add_vertex(VertexType.Z, -2, position, 0)
    g.add_edge(g.edge(v,e), EdgeType.HADAMARD)
    g.set_mtype(v, MeasurementType.Z)
    g.set_mtype(e, MeasurementType.EFFECT)
    for n in neighbors:
        g.add_edge(g.edge(v,n), EdgeType.HADAMARD)
    return v

def z_delete_all(g: GraphMBQC):
    candidates = []
    for v in g.vertices():
        if g.mtype(v) == MeasurementType.Z or (g.mtype(v) in [MeasurementType.XZ, MeasurementType.YZ] and g.phase(v) in [0,1]):
            candidates.append(v)
    for candidate in candidates:
        z_delete(g, candidate)

def yz_fusion(g: GraphMBQC, vertex: VT):
    """Fuses YZ spider with single neighbor.
    preserves flow due to Lemma 5.4. of https://arxiv.org/pdf/2109.05654.pdf
    preserves standard interpretation due to id and fusion rule"""
    if not g.mtype(vertex) == MeasurementType.YZ:
        return False
    effect = None
    neighbor = None
    for i,n in enumerate(g.neighbors()):
        if i > 2:
            return False
        if g.effect(vertex) == n:
            effect = n
        else:
            neighbor = n

    g.set_phase(neighbor, g.phase(neighbor) + g.phase(effect))
    g.remove_vertices([vertex,effect])
    return True

def neighbor_unfusion(g: GraphMBQC, vertex: VT, neighbor: VT):
    """neighbor unfusion rule as in https://elib.dlr.de/188470/1/QPL_2022_paper_10.pdf"""
    nv = list(set(g.neighbors(vertex)).difference(set([neighbor])))
    z1 = z_insert(g, [vertex] + nv)
    lcomp(g, z1)
    z2 = z_insert(g, [z1] + nv)
    lcomp(g, z2)

def spider_split(g: GraphMBQC, vertex: VT, split_neighbors: List[VT]):
    """Removes an arbitrary number of neighbors from a spider by using some sort of generalized neighbor unfusion rule"""
    z1 = z_insert(g, [vertex] + split_neighbors)
    lcomp(g, z1)
    z2 = z_insert(g, [z1] + split_neighbors)
    lcomp(g, z2)
    return z2

def boundary_insert(g: GraphMBQC, vertex: VT) -> VT:
    """Inserts an X spider at the before an input or after an output vertex.
    According to Lemma 3.7. and 3.8. of https://arxiv.org/pdf/2003.01664.pdf we can insert XY-spiders at the boundaries
    and according to Lemma 5.1. of https://arxiv.org/pdf/2109.05654.pdf we can relabel XY-spiders with angle 0 or pi as X measurements.
    Returns inserted vertex."""
    #check inputs
    boundary_vertex = set(g.neighbors(vertex)).intersection(set(g.inputs()))
    if boundary_vertex:
        newv = insert_identity(g, vertex, boundary_vertex.pop())
        g.set_mtype(newv, MeasurementType.X)
        return newv
    #check outputs
    boundary_vertex = set(g.neighbors(vertex)).intersection(set(g.outputs()))
    if boundary_vertex:
        newv = insert_identity(g, boundary_vertex.pop(), vertex)
        g.set_mtype(newv, MeasurementType.X)
        return newv
    return False