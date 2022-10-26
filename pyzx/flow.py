from .graph.base import BaseGraph, VT, ET
from .graph.graph_mbqc import GraphMBQC
from typing import Literal, Tuple, Dict, Set, Optional
from .utils import MeasurementType, VertexType
from .extract import bi_adj
from .linalg import Mat2, CNOTMaker

class FlowType:
    Type = Literal[0,1,2,3]
    CAUSAL = 0
    XYGFLOW = 1
    GFLOW = 2
    PAULIFLOW = 3

"""Generic Flow type = Tuple of correction sets and vertex depth"""
Flow = (Dict[VT, Set[VT]], Dict[VT,int])

"""
Function for calculating the maximally delayed causal flow in a graph-like ZX-Diagram
Nearly identical to tket version, but with reversed ordering and a minor bugfix
Algorithm is taken from https://arxiv.org/pdf/0709.2670.pdf
"""
def identify_causal_flow(g: BaseGraph[VT,ET]) -> Flow:
    solved = set()
    correctors = set()
    past: Dict[VT, int] = dict()
    res: Flow = (dict(),dict())

    for o in g.outputs():
        n = list(g.neighbors(o))[0]
        past[n] = len(g.neighbors(n)) - 1
        solved.update(set([n,o]))
        res[0][n] = set()
        res[1][n] = 0
        if not n in g.inputs():
            correctors.add(n)
    
    depth = 1

    while True:
        new_correctors = set()
        new_solved = solved.copy()
        for v in correctors:
            u = []
            for n in g.neighbors(v):
                if not n in solved:
                    u.append(n)
            if len(u) == 1:
                u = u[0]
            else:
                continue

            res[0][u] = set([v])
            res[1][u] = depth
            new_solved.add(u)
            
            n_found = 0
            input_found = False
            for n in g.neighbors(u):
                if n in g.inputs():
                    input_found = True
                    new_solved.add(n)
                    continue
                if not n in solved:
                    n_found += 1
                if n in past and past[n] > 0:
                    past[n] -= 1
                    if past[n] == 1:
                        new_correctors.add(n)
            if not input_found:
                past[u] = n_found
                if n_found == 1:
                    new_correctors.add(u)
            
        correctors = new_correctors
        solved = new_solved
        depth += 1
        if len(correctors) == 0:
            break
    
    if len(solved) != g.num_vertices():
        return None
    
    inv_depth = inverse_depth(res[1])
    return (res[0],inv_depth)


"""Compute the maximally delayed gflow of a diagram in graph-like form where every spider is measured in XY plane.

Based on algorithm by Perdrix and Mhalla.
See dx.doi.org/10.1007/978-3-540-70575-8_70
"""
def identify_xy_gflow(g: BaseGraph[VT, ET]) -> Flow:

    res: Flow = (dict(),dict())

    inputs: Set[VT] = set(g.inputs())
    processed: Set[VT] = set(g.outputs()) | g.grounds()
    vertices: Set[VT] = set(g.vertices())
    pattern_inputs: Set[VT] = set()
    for inp in inputs:
        if g.type(inp) == VertexType.BOUNDARY:
            pattern_inputs |= set(g.neighbors(inp))
        else:
            pattern_inputs.add(inp)
    k: int = 1

    for v in processed:
        res[1][v] = 0

    while True:
        correct = set()
        processed_prime = [v for v in processed.difference(pattern_inputs) if any(w not in processed for w in g.neighbors(v))]
        candidates = [v for v in vertices.difference(processed) if any(w in processed_prime for w in g.neighbors(v))]

        zerovec = Mat2([[0] for _ in range(len(candidates))])
        m = bi_adj(g, processed_prime, candidates)
        cnot_maker = CNOTMaker()
        m.gauss(x=cnot_maker, full_reduce=True)

        for u in candidates:
            vu = zerovec.copy()
            vu.data[candidates.index(u)] = [1]
            x = get_gauss_solution(m, vu, cnot_maker.cnots)
            if x:
                correct.add(u)
                res[0][u] = {processed_prime[i] for i in range(x.rows()) if x.data[i][0]}
                res[1][u] = k

        if not correct:
            if not candidates:
                if len(processed) != g.num_vertices():
                    return None
                inv_depth = inverse_depth(res[1])
                return (res[0], inv_depth)
            return None
        else:
            processed.update(correct)
            k += 1

"""helper function for solving M * x = b if additions (cnots) for getting M in echelon form are already calculated"""
def get_gauss_solution(gauss: Mat2, vec: Mat2, cnots: CNOTMaker):
    for cnot in cnots:
        vec.row_add(cnot.target,cnot.control)
    x = Mat2.zeros(gauss.cols(),1)
    for i,row in enumerate(gauss.data):
        got_pivot = False
        for j,v in enumerate(row):
            if v != 0:
                got_pivot = True
                x.data[j][0] = vec.data[i][0]
                break
        if not got_pivot and vec.data[i][0] != 0:
            return None
    return x

"""helper function for inversing the depth dictionary of the flow algorithms"""
def inverse_depth(depth: Dict) -> Dict:
    # reverse ordering of vertices
    max_depth = depth[max(depth, key=depth.get)]
    inv_depth = dict()
    for k,v in depth.items():
        inv_depth[k] = max_depth-v
    return inv_depth

# {{0,1,0,1,0,1},{1,0,1,0,1,0},{0,1,0,1,0,1},{1,0,1,0,1,0},{0,1,0,1,0,1},{1,0,1,0,1,0}}*{{x1},{x2},{1},{x4},{x5},{x6}}={{0},{0},{0},{0},{0},{0}}

def identify_gflow(g: GraphMBQC) -> Flow:
    res: Flow = (dict(), dict())
    processed = set(g.outputs())
    vertices: Set[VT] = set(g.vertices()).difference(set(g.effects()))
    inputs: Set[VT] = set(g.inputs())
    depth: int = 1
    
    for v in processed:
        res[1][v] = 0
    
    while True:
        correct = set()
        processed_prime = [v for v in processed.difference(inputs) if any(w not in processed for w in g.neighbors(v))]
        candidates = [v for v in vertices.difference(processed) if any(w in processed_prime for w in g.neighbors(v))]

        zerovec = Mat2([[0] for _ in range(len(candidates))])
        m = bi_adj(g, processed_prime, candidates)
        cnot_maker = CNOTMaker()
        m.gauss(x=cnot_maker, full_reduce=True)

        for u in candidates:
            vu = zerovec.copy()
            mtype = g.mtype(u)
            if mtype == MeasurementType.XY:
                vu.data[candidates.index(u)] = [1]
            else:
                vu = Mat2([[1] if candidates[i] in g.neighbors(u) else [0] for i in range(len(candidates))])
                if mtype == MeasurementType.XZ:
                    vu.data[candidates.index(u)] = [1] if vu.data[candidates.index(u)] == 0 else [0]


            x = get_gauss_solution(m, vu, cnot_maker.cnots)

            if x:
                correct.add(u)
                res[0][u] = {processed_prime[i] for i in range(x.rows()) if x.data[i][0]}
                if mtype != MeasurementType.XY:
                    res[0][u].add(u)
                res[1][u] = depth

        if not correct:
            if not candidates:
                inv_depth = inverse_depth(res[1])
                return (res[0], inv_depth)
            return None
        else:
            processed.update(correct)
            depth += 1

## Testing purposes

def get_odd_neighbourhood(g: BaseGraph[VT,ET], vertex_set):
  all_neighbors = set()
  for vertex in vertex_set:
    all_neighbors.update(set(g.neighbors(vertex)).difference(g.effects()))
  odd_neighbors = []
  for neighbor in all_neighbors:
    if len(set(g.neighbors(neighbor)).difference(g.effects()).intersection(vertex_set)) % 2 == 1:
      odd_neighbors.append(neighbor)
  return odd_neighbors

def check_gflow_condition1(g: BaseGraph[VT,ET], gflow: Flow):
  for v in set(g.vertices()).difference(set(g.outputs())).difference(g.effects()):
    for w in gflow[0][v]:
      if v!=w and gflow[1][v] > gflow[1][w]:
        print("gflow violates condition 1 because vertex: ",v," has higher or equal ordering than vertex ",w," which is in the correction set of vertex ",v)
        return False
  return True

def check_gflow_condition2(g: BaseGraph[VT,ET], gflow: Flow):
  for v in set(g.vertices()).difference(set(g.outputs())).difference(g.effects()):
    for w in get_odd_neighbourhood(g,gflow[0][v]):
      if v!=w and gflow[1][v] > gflow[1][w]:
        print("gflow violates condition 2 because vertex: ",v," has higher or equal ordering than vertex ",w," which is in the odd neighborhood of the correction set of vertex ",v)
        return False
  return True

def check_gflow_condition3(g: BaseGraph[VT,ET], gflow: Flow):
    for v in set(g.vertices()).difference(set(g.outputs())).difference(g.effects()):
        odd_n = get_odd_neighbourhood(g,gflow[0][v])
        if g.mtype(v) == MeasurementType.XY:
            if v in gflow[0][v] or not v in odd_n:
                print("gflow violates condition 3 for XY measured vertex ",v)
                return False
        elif g.mtype(v) == MeasurementType.XZ:
            if not v in gflow[0][v] or not v in odd_n:
                print("gflow violates condition 3 for XZ measured vertex ",v)
                return False
        else:
            if not v in gflow[0][v] or v in odd_n:
                print("gflow violates condition 3 for YZ measured vertex ",v)
                return False
    return True