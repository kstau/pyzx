from .graph.base import BaseGraph, VT, ET
from .graph.graph_mbqc import GraphMBQC
from typing import Literal, Tuple, Dict, Set, Optional, List
from .utils import MeasurementType, VertexType, EdgeType
from .circuit import Circuit
from .extract import clean_frontier, graph_to_swaps

"""Extracts a circuit from graph-like diagrams with causal flow"""
def extract_from_causal_flow(g: BaseGraph[VT, ET]) -> Circuit:
    res = Circuit(g.qubit_count())
    inputs = g.inputs()
    processed = set(g.outputs())
    qubit_map: Dict[VT,int] = dict()
    frontier = set()

    #create frontier
    for i, o in enumerate(processed):
        v = list(g.neighbors(o))[0]
        if not v in inputs:
            frontier.add(v)
            qubit_map[v] = i
            if g.edge_type(g.edge(v,o)) == EdgeType.HADAMARD:
                res.add_gate("HAD", i)
                g.set_edge_type(g.edge(v,o),EdgeType.SIMPLE)

    # extract CZs + RZ + Hadamard until no spiders left in diagram            
    while True:
        #RZs
        for v in frontier:
            phase = g.phase(v)
            if phase != 0:
                g.set_phase(v,0)
                res.add_gate("ZPhase", qubit_map[v], phase)

        #CZs
        for v in frontier:
            for w in set(g.neighbors(v)).intersection(frontier):
                g.remove_edge(g.edge(v,w))
                res.add_gate("CZ", qubit_map[v], qubit_map[w])
        
        new_frontier = set()
        #Hadamards
        for v in frontier:
            if len(g.neighbors(v)) > 2: #process later
                continue
            output = list(set(g.neighbors(v)).intersection(processed))[0]

            # extract (chains of empty spiders and) hadamards 
            neighbors, hcount = process_hadamards(g, inputs, processed, v)
            if hcount % 2 == 1:
                res.add_gate("HAD", qubit_map[v])

            #update diagram
            for n in neighbors[:-1]:
                g.remove_vertex(n)
            edge_type = g.edge_type(g.edge(v,output))
            g.remove_vertex(v)
            processed.add(v)
            g.add_edge(g.edge(neighbors[-1],output),edge_type)

            #get new frontier vertices
            if not neighbors[-1] in inputs:
                new_frontier.add(neighbors[-1])
                qubit_map[neighbors[-1]] = qubit_map[v]
        
        if len(new_frontier) == 0:
            break
        else:
            #update frontier
            frontier.difference_update(processed)
            frontier.update(new_frontier)

    # reverse circuit 
    res.gates = list(reversed(res.gates))

    # add swaps if necessary (?)
    return graph_to_swaps(g, False) + res

"""helper function for circuit extraction: Finds chains of Hadamard + empty 2-ary Z spiders starting from a frontier vertex
Returns: list of 2-ary spiders + number of Hadamard wires in the chain"""
def process_hadamards(g: BaseGraph[VT, ET], inputs, processed, v):
    neighbors = []
    hcount = 0
    while True:
        n = list(set(g.neighbors(v)).difference(processed))[0]
        neighbors.append(n)
        if g.edge_type(g.edge(n,v)) == EdgeType.HADAMARD:
            hcount += 1 
        if g.phase(n) != 0 or len(g.neighbors(n)) > 2 or n in inputs:
            return neighbors, hcount
        else:
            processed = set([v])
            v = n 