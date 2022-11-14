from .graph.base import BaseGraph, VT, ET
from .graph.graph_mbqc import GraphMBQC
from typing import Literal, Tuple, Dict, Set, Optional, List
from .utils import MeasurementType, VertexType, EdgeType
from .circuit import Circuit
from .extract import graph_to_swaps, bi_adj, neighbors_of_frontier, extract_circuit
from .linalg import CNOTMaker
from .drawing import draw

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
    import pdb
    pdb.set_trace()

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

def extract_from_xy_gflow(g: BaseGraph[VT, ET]) -> Circuit:
    res = Circuit(g.qubit_count())
    inputs = g.inputs()
    outputs = set(g.outputs())
    frontier: Dict[int,VT] = dict()

    #create frontier
    for i, o in enumerate(g.outputs()):
        v = list(g.neighbors(o))[0]
        if not v in inputs:
            frontier[i] = v
            if g.edge_type(g.edge(v,o)) == EdgeType.HADAMARD:
                res.add_gate("HAD", i)
                g.set_edge_type(g.edge(v,o),EdgeType.SIMPLE)

    # extract CZs + RZ + Hadamard until no spiders left in diagram            
    while True:
        #RZs
        for qubit,v in frontier.items():
            phase = g.phase(v)
            if phase != 0:
                g.set_phase(v,0)
                res.add_gate("ZPhase", qubit, phase)
                print("extract ZPhase on ", qubit, phase)

        #CZs
        for qubit, v in frontier.items():
            for w in set(g.neighbors(v)).intersection(set(frontier.values())):
                g.remove_edge(g.edge(v,w))
                res.add_gate("CZ", qubit, list(frontier.keys())[list(frontier.values()).index(w)])
                print("extract CZ on ", qubit, list(frontier.keys())[list(frontier.values()).index(w)])

        # If we cannot proceed with H,RZ,CZ gate extractions remove Hadamard Wires via CNOT row additions using gaussian elimination
        if all([len(g.neighbors(v)) > 2 for v in frontier.values()]):
            
            #Get all neighbors of frontier vertices
            frontier_neighbors = set()
            for v in frontier.values():
                frontier_neighbors.update(set(g.neighbors(v)).difference(outputs))
            
            # Compute row echelon form of adjacency matrix and save row operations as CNOTs 
            # -> because of gflow the resulting matrix has a row with only a single 1
            m = bi_adj(g, list(frontier_neighbors), frontier.values())
            cnot_maker = CNOTMaker()
            m.gauss(x=cnot_maker, full_reduce=True)

            for cnot in cnot_maker.cnots:
                control_qubit = list(frontier)[cnot.control]
                target_qubit = list(frontier)[cnot.target]
                res.add_gate("CNOT", control_qubit, target_qubit)
                print("extract CNOT on ", control_qubit, target_qubit)

                # Add or remove Hadamard wires according to CNOT
                ftarg = frontier[control_qubit]
                fcont = frontier[target_qubit]
                for v in set(g.neighbors(fcont)).difference(outputs):
                    # remove wire
                    if g.connected(ftarg,v):
                        g.remove_edge(g.edge(ftarg,v))
                    # add wire
                    else:
                        # special case: neighbor of "control" spider is an input, therefore we need to insert a spider between input and control spider
                        if v in g.inputs():
                            new_v = insert_identity(g, fcont, v) 
                            print("insert identity between ",v, fcont)
                            g.add_edge(g.edge(ftarg,new_v), EdgeType.HADAMARD)
                        else:
                            g.add_edge(g.edge(ftarg,v), EdgeType.HADAMARD)

        #Hadamards
        new_frontier: Dict[int,VT] = dict()

        for qubit,v in frontier.items():
            if len(g.neighbors(v)) > 2 or v in g.inputs(): #process later
                continue
            output = list(set(g.neighbors(v)).intersection(outputs))[0]

            # extract (chains of empty spiders and) hadamards 
            neighbors, hcount = process_hadamards(g, inputs, outputs, v)
            if hcount % 2 == 1:
                res.add_gate("HAD", qubit)
                print("HAD on ", qubit)

            #update diagram
            for n in neighbors[:-1]:
                g.remove_vertex(n)
            edge_type = g.edge_type(g.edge(v,output))
            g.remove_vertex(v)

            g.add_edge(g.edge(neighbors[-1],output),edge_type)

            #get new frontier vertices
            if not neighbors[-1] in inputs:
                new_frontier[qubit] = neighbors[-1]
            else:
                new_frontier[qubit] = -1 #remove from frontier
        
        if len(new_frontier) == 0:
            break
        else:
            #update frontier
            for qubit,v in new_frontier.items():
                if v == -1:
                    frontier.pop(qubit)
                else:
                    frontier[qubit] = v

    # reverse circuit 
    res.gates = list(reversed(res.gates))

    # add swaps if necessary (?)
    return graph_to_swaps(g, False) + res


def insert_identity(g, v1, v2) -> int:
    orig_type = g.edge_type(g.edge(v1, v2))
    if g.connected(v1, v2):
        g.remove_edge(g.edge(v1, v2))
    vmid = g.add_vertex(VertexType.Z,g.qubits()[v1],g.rows()[v1] -1)
    g.add_edge((v1,vmid), EdgeType.HADAMARD)
    if orig_type == EdgeType.HADAMARD:
        g.add_edge((vmid,v2), EdgeType.SIMPLE)
    else:
        g.add_edge((vmid,v2), EdgeType.HADAMARD)
    return vmid