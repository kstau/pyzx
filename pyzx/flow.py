from .graph.base import BaseGraph, VT, ET
from typing import Literal, Tuple, List, Dict, Set
from .utils import VertexType, EdgeType

class FlowType:
    Type = Literal[0,1,2,3]
    CAUSAL = 0
    XYGFLOW = 1
    GFLOW = 2
    PAULIFLOW = 3

Flow = (Dict[VT, Set[VT]], Dict[VT,int])

"""
Function for identifying causal flow in a graph-like ZX-Diagram
Nearly identical to tket version, but with reversed ordering and bugfix
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
    
    # reverse ordering of vertices
    max_depth = res[1][max(res[1], key=res[1].get)]
    inv_depth = dict()
    for k,v in res[1].items():
        inv_depth[k] = max_depth-v
    return (res[0],inv_depth)
