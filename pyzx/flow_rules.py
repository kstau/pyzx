
from .graph.graph_mbqc import GraphMBQC
from typing import List
from .graph.base import VT
from .utils import MeasurementType, VertexType, EdgeType, insert_identity, toggle_edge
from fractions import Fraction
from .drawing import draw

"""Pauli flow preserving rewrite rules for graph-like diagrams"""

def lcomp(g: GraphMBQC, v: VT) -> bool:
    """
    Graph theoretic local complementation on graph-like diagram
    This method does not eliminate Clifford spiders, but introduces XZ and YZ measurement effects as in https://arxiv.org/pdf/2003.01664.pdf
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
        
        # set new measurement plane and angle for neighbors
        g.set_mtype(n, {
            MeasurementType.YZ: MeasurementType.XZ,
            MeasurementType.XZ: MeasurementType.YZ,
            MeasurementType.X: MeasurementType.Y,
            MeasurementType.Y: MeasurementType.X,
            MeasurementType.Z: MeasurementType.Z
        }.get(g.mtype(n), MeasurementType.XY))
        n_phase = g.phase(n)
        g.set_phase(n, {
            MeasurementType.XY: n_phase + Fraction(1,2),
            MeasurementType.XZ: Fraction(1,2),
            MeasurementType.X: n_phase + Fraction(1,2),
            MeasurementType.Y: n_phase + Fraction(1,2),
        }.get(g.mtype(n), 0))
        if g.mtype(n) in [MeasurementType.YZ]:
            e = g.effect(n)
            g.set_phase(e, -g.phase(e))
    
    # set new measurement plane of v
    g.set_mtype(v, {
        MeasurementType.XY: MeasurementType.XZ,
        MeasurementType.XZ: MeasurementType.XY,
        MeasurementType.X: MeasurementType.X,
        MeasurementType.Y: MeasurementType.Z,
        MeasurementType.Z: MeasurementType.Y
    }.get(g.mtype(v), MeasurementType.YZ))

    if g.mtype(v) in [MeasurementType.XZ, MeasurementType.Z]:
        phase_add = Fraction(1,2) #if g.mtype(v) == MeasurementType.Z else 
        # create effect spider
        newv = None
        # check if we can write effect on output
        for n in g.neighbors(v): 
            if n in g.outputs():
                newv = insert_identity(g, v, n)
                # lcomp introduces Hadamard on output wire
                g.set_edge_type(g.edge(newv, n), toggle_edge(g.edge_type(g.edge(newv, n))))

                g.set_phase(newv, g.phase(v)+phase_add)
                break
        # if not: add effect spider
        if not newv:
            newv = g.add_vertex(VertexType.Z, -1, g.row(v), g.phase(v)+phase_add)
            g.add_edge(g.edge(v, newv), EdgeType.HADAMARD)
            
        g.set_mtype(newv, MeasurementType.EFFECT)

        # update phase of v
        g.set_phase(v, Fraction(1,2) if g.mtype(v) == MeasurementType.XZ else 0)
    
    elif g.mtype(v) in [MeasurementType.XY, MeasurementType.Y]:
        e = g.effect(v)
        g.set_phase(v, - g.phase(e) + Fraction(1,2))
        g.remove_vertex(e)
    
    elif g.mtype(v) == MeasurementType.YZ:
        e = g.effect(v)
        g.set_phase(e, g.phase(e)-Fraction(1,2))
    
    return True


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
    """Deletes a Z measured vertex from graph-like diagram"""
    if g.mtype(vertex) == MeasurementType.Z and not any([n for n in g.neighbors(vertex) if g.type(n) != VertexType.Z]):
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
        if g.mtype(v) == MeasurementType.Z:
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

