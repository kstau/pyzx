from .graph.graph_mbqc import GraphMBQC
from .graph.base import BaseGraph, VT, ET
from .simplify import spider_simp, to_gh, id_simp
from .flow_extract import relabel_pauli_measurements
from .flow_rules import lcomp, pivot, z_delete, spider_split
from .utils import MeasurementType, VertexType, EdgeType, insert_identity
import random
from fractions import Fraction

from .drawing import draw
from .tensor import compare_tensors

"""
TODO: implement an id preserving copy method
"""

def to_graph_like(g: BaseGraph):
    spider_simp(g)
    to_gh(g)
    id_simp(g)
    spider_simp(g)
    res = g.copy(backend='mbqc')
    res.normalize()
    relabel_pauli_measurements(res)
    return res

def apply_rule(g: GraphMBQC, match):
    rule, vertices, heuristic = match
    orig_num_wires = g.num_edges()
    if rule == 'lc':
        lcomp(g, vertices[0])
    elif rule == 'pv':
        pivot(g, vertices[0], vertices[1])
    for vertex in vertices:
        if g.mtype(vertex) == MeasurementType.Z:
            z_delete(g, vertex)
    print("heuristic said",heuristic,"really got",orig_num_wires-g.num_edges(),"match",match)

def simplify_hadamard_wires(g: GraphMBQC, threshold=0, strategy='greedy'):
    watchlist = []
    while True:
        rules = collect_rule_applications(g)
        filtered_rules = [(s,r,h) for s,r,h in rules if h >= threshold and not r in watchlist]
        # debug, interior clifford simp only, aka. only apply rules on paulis
        filtered_rules = [(s,r,h) for s,r,h in filtered_rules if (s == 'lc' and g.is_pauli(r[0])) or (s == 'pv' and g.is_pauli(r[0]) and g.is_pauli(r[1]))]
        if not filtered_rules:
            break
        if strategy == 'greedy':
            filtered_rules.sort(key=lambda t: t[2])
            apply_rule(g, filtered_rules[0])
        else:
            apply_rule(g, random.choice(filtered_rules))
        draw(g, labels=True)
        watchlist.append(filtered_rules[0][1])


def collect_rule_applications(g: GraphMBQC):
    rules = []
    interior_vertices = g.interior_vertices()
    for v in interior_vertices:
        rules.append(['lc',(v,),lc_heuristic(g, v)+z_lcomp(g, v)])
    for e in g.edges():
        u,v = g.edge_st(e)
        if u in interior_vertices and v in interior_vertices:
            rules.append(['pv',(u,v),pv_heuristic(g, u, v) + z_pivot(g, u, v)])
    return rules        

def lc_heuristic(g: GraphMBQC, v: VT):
    neighbors = list(g.mneighbors(v))
    m = 0
    n = len(neighbors)
    # get number of connected neighbors m
    for i, w in enumerate(neighbors):
        for x in neighbors[i:]:
            if g.connected(w,x):
                m += 1
    #LCH without consideration of measurement effects
    return 2*m - n*(n-1)/2

def pv_heuristic(g: GraphMBQC, u: VT, v: VT):
    nu = set(g.mneighbors(u)).difference(set([v]))
    nv = set(g.mneighbors(v)).difference(set([u]))
    nsets = [nu.difference(nv), nv.difference(nu), nu.intersection(nv)]
    cmax = len(nsets[0])*len(nsets[1]) + len(nsets[0])*len(nsets[2]) + len(nsets[1])*len(nsets[2])
    m = 0
    for i, nset in enumerate(nsets):
        for compareset in nsets[i+1:]:
            for v1 in nset:
                for v2 in compareset:
                    if g.connected(v1,v2):
                        m += 1

    return 2*m - cmax

def z_lcomp(g: GraphMBQC, v: VT):
    return len(g.mneighbors(v)) if g.mtype(v) == MeasurementType.Y else 0

def z_pivot(g: GraphMBQC, u: VT, v: VT):
    res = 0
    if g.mtype(u) == MeasurementType.X:
        res += len(g.mneighbors(v))
    if g.mtype(v) == MeasurementType.X:
        res += len(g.mneighbors(u))
    return res


def recursive_split(g: GraphMBQC, vertex: VT, maximal_neighbors: int, is_boundary: bool):
    print("vertex",vertex,"is_boundary",is_boundary)
    vn = list(g.mneighbors(vertex))
    if len(vn) > (maximal_neighbors - 1 if is_boundary else maximal_neighbors):
        new_y = spider_split(g, vertex, vn[:len(vn)//2])

        recursive_split(g, vertex, maximal_neighbors, is_boundary)
        recursive_split(g, new_y, maximal_neighbors, False)


def to_grid(g: GraphMBQC):
    for vertex in g.mvertices():
        recursive_split(g, vertex, 4, not vertex in g.interior_vertices())