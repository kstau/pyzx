from .graph.base import BaseGraph, VT, ET
from .graph.graph_mbqc import GraphMBQC
from typing import Dict, Set
from .utils import MeasurementType, EdgeType, insert_identity
from .circuit import Circuit
from .extract import graph_to_swaps, bi_adj
from .linalg import CNOTMaker
from .flow_rules import lcomp, pivot

"""Extracts a circuit from graph-like diagrams with causal flow"""
def extract_from_causal_flow(g: BaseGraph[VT, ET]) -> Circuit:
    circuit = Circuit(g.qubit_count())
    frontier = init_frontier(g, circuit)
    # extract CZs + RZ + Hadamard until no spiders left in diagram            
    while True:
        #RZs
        extract_rzs(g, frontier, circuit)
        #CZs
        extract_czs(g, frontier, circuit)
        #Hadamards
        new_frontier = process_frontier(g, frontier, circuit)
        #Update frontier
        if new_frontier: 
            for qubit,v in new_frontier.items():
                if v == -1:
                    frontier.pop(qubit)
                else:
                    frontier[qubit] = v
        else:
            break
    # reverse circuit 
    circuit.gates = list(reversed(circuit.gates))
    # add swaps if necessary
    return graph_to_swaps(g, False) + circuit

"""Extracts a circuit from graph-like diagrams with XY-gflow, i.e. all spiders are measured in XY-plane"""
def extract_from_xy_gflow(g: BaseGraph[VT, ET]) -> Circuit:
    circuit = Circuit(g.qubit_count())
    frontier = init_frontier(g, circuit)
    # extract CZs + RZ + Hadamard until no spiders left in diagram            
    while True:
        #RZs
        extract_rzs(g, frontier, circuit)
        #CZs
        extract_czs(g, frontier, circuit)
        # If we cannot proceed with H,RZ,CZ gate extractions remove Hadamard Wires via CNOT row additions using gaussian elimination
        if all([len(g.neighbors(v)) > 2 for v in frontier.values()]):
            frontier_neighbors = get_frontier_neighbors(g, frontier)
            # Compute row echelon form of adjacency matrix and save row operations as CNOTs 
            # -> because of gflow the resulting matrix has a row with only a single 1
            m = bi_adj(g, list(frontier_neighbors), frontier.values())
            cnot_maker = CNOTMaker()
            m.gauss(x=cnot_maker, full_reduce=True)
            extract_cnots(g, frontier, circuit, cnot_maker)

        new_frontier = process_frontier(g, frontier, circuit)
        #Update frontier
        if new_frontier: 
            for qubit,v in new_frontier.items():
                if v == -1:
                    frontier.pop(qubit)
                else:
                    frontier[qubit] = v
        else:
            break
    # reverse circuit
    circuit.gates = list(reversed(circuit.gates))
    # add swaps if necessary
    return graph_to_swaps(g, False) + circuit

"""Extracts a circuit from graph-like diagrams with gflow, i.e. all spiders are measured in XY, XZ or YZ plane"""
def extract_from_gflow(g: GraphMBQC) -> Circuit:
    circuit = Circuit(g.qubit_count())
    # Transform all XZ spiders to XY spiders via local complementation
    eliminate_xz_spiders(g)
    frontier = init_frontier(g, circuit)

    # extract CZs + RZ + Hadamard until no spiders left in diagram
    while True:
        #RZs
        extract_rzs(g, frontier, circuit)
        #CZs
        extract_czs(g, frontier, circuit)
        # If we cannot proceed with H,RZ,CZ gate extractions remove Hadamard Wires via CNOT row additions using gaussian elimination
        if frontier and all([len(g.neighbors(v)) > 2 for v in frontier.values()]):
            # Get all neighbors of frontier vertices
            frontier_neighbors = get_frontier_neighbors(g, frontier)
            # Compute row echelon form of adjacency matrix and save row operations as CNOTs 
            # -> because of gflow the resulting matrix has a row with only a single 1
            m = bi_adj(g, list(frontier_neighbors), frontier.values())
            cnot_maker = CNOTMaker()
            m.gauss(x=cnot_maker, full_reduce=True)
            # If there is no row with a single 1 there has to be a YZ measured spider in frontier neighbors which we can eliminate
            if not any([sum(row) == 1 for row in m.data]):
                eliminate_yz_spider(g, frontier, frontier_neighbors, circuit)
                continue
            extract_cnots(g, frontier, circuit, cnot_maker)

        #Hadamards
        new_frontier = process_frontier(g, frontier, circuit)
        #Update frontier
        if new_frontier: 
            for qubit,v in new_frontier.items():
                if v == -1:
                    frontier.pop(qubit)
                else:
                    frontier[qubit] = v
        else:
            break
    # reverse circuit 
    circuit.gates = list(reversed(circuit.gates))
    # add swaps if necessary
    return graph_to_swaps(g, False) + circuit

"""Inits the frontier of a ZX-diagram with the spiders adjacent to the outputs. Extracts Hadamard wires between outputs and frontier"""
def init_frontier(g: BaseGraph[VT, ET], circuit: Circuit) -> Dict[int,VT]:
    frontier: Dict[int,VT] = dict()

    for i, o in enumerate(g.outputs()):
        v = list(g.neighbors(o))[0]
        if not v in g.inputs():
            frontier[i] = v
            if g.edge_type(g.edge(v,o)) == EdgeType.HADAMARD:
                circuit.add_gate("HAD", i)
                g.set_edge_type(g.edge(v,o),EdgeType.SIMPLE)
    
    return frontier

"""Extracts phases of frontier spiders as ZPhase gates and updates the diagram"""
def extract_rzs(g: BaseGraph[VT, ET], frontier: Dict[int,VT], circuit: Circuit):
    for qubit,v in frontier.items():
        phase = g.phase(v)
        if phase != 0:
            g.set_phase(v,0)
            circuit.add_gate("ZPhase", qubit, phase)

"""Extracts connected frontier spiders as controlled Z gates and updates the diagram"""
def extract_czs(g: BaseGraph[VT, ET], frontier: Dict[int,VT], circuit: Circuit):
    for qubit, v in frontier.items():
        for w in set(g.neighbors(v)).intersection(set(frontier.values())):
            g.remove_edge(g.edge(v,w))
            circuit.add_gate("CZ", qubit, list(frontier.keys())[list(frontier.values()).index(w)])

"""Processes all frontier spiders which are only connected to a single non-output spider in the diagram by extracting Hadamards.
Returns new frontier vertices"""
def process_frontier(g: BaseGraph[VT, ET], frontier: Dict[int,VT], circuit: Circuit):
    new_frontier: Dict[int,VT] = dict()
    inputs = g.inputs()
    outputs = g.outputs()

    for qubit,v in frontier.items():
        if len(g.neighbors(v)) > 2 or v in inputs: #process later
            continue
        output = list(set(g.neighbors(v)).intersection(outputs))[0]

        # extract (chains of empty spiders and) hadamards 
        neighbors, hcount = process_hadamards(g, inputs, outputs, v)
        if hcount % 2 == 1:
            circuit.add_gate("HAD", qubit)

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
    
    return new_frontier

"""Helper function for process_frontier: Finds chains of Hadamard + empty 2-ary Z spiders starting from a frontier vertex
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

"""Returns all non-output neighbors of a frontier"""
def get_frontier_neighbors(g: BaseGraph[VT, ET], frontier: Dict[int,VT]):
    frontier_neighbors = set()
    for v in frontier.values():
        frontier_neighbors.update(set(g.neighbors(v)).difference(set(g.outputs())))
    return frontier_neighbors

"""Extracts CNOT gates resulting from gaussian elimination to circuit and adds the Hadamard wires of the corresponding frontier vertices"""
def extract_cnots(g: BaseGraph[VT, ET], frontier: Dict[int,VT], circuit: Circuit, cnot_maker: CNOTMaker):
    for cnot in cnot_maker.cnots:
        # Add CNOT to circuit
        control_qubit = list(frontier)[cnot.control]
        target_qubit = list(frontier)[cnot.target]
        circuit.add_gate("CNOT", control_qubit, target_qubit)

        # Add or remove Hadamard wires in diagram according to CNOT addition
        ftarg = frontier[control_qubit]
        fcont = frontier[target_qubit]
        for v in set(g.neighbors(fcont)).difference(set(g.outputs())):
            # remove wire
            if g.connected(ftarg,v):
                g.remove_edge(g.edge(ftarg,v))
            # add wire
            else:
                if v in g.inputs():
                    # special case: neighbor of "control" spider is an input, therefore we need to insert a spider between input and control spider
                    new_v = insert_identity(g, fcont, v) 
                    g.add_edge(g.edge(ftarg,new_v), EdgeType.HADAMARD)
                else:
                    g.add_edge(g.edge(ftarg,v), EdgeType.HADAMARD)

"""Repeatedly applies local complementation on all XZ spiders in a diagram which thereby become XY spiders"""
def eliminate_xz_spiders(g: GraphMBQC):
    while True:
        candidates = []
        for v in g.vertices():
            if g.mtype(v) == MeasurementType.XZ:
                candidates.append(v)
    
        if candidates:
            for candidate in candidates:
                lcomp(g, candidate)
        else:
            break

"""Finds a YZ measured spider which is connected to a spider in frontier and applies a pivot on them. 
By that, the YZ spider transforms to a XY spider"""
def eliminate_yz_spider(g: GraphMBQC, frontier: Dict[int,VT], frontier_neighbors: Set, circuit: Circuit):
    for n in frontier_neighbors:
        if g.mtype(n) == MeasurementType.YZ:
            frontier_vertex = list(set(g.neighbors(n)).intersection(set(frontier.values())))[0]
            pivot(g, n, frontier_vertex)

            e = g.effect(frontier_vertex)
            e_n = list(g.neighbors(e))
            simple_wire = g.edge_type(g.edge(e,e_n[0])) == g.edge_type(g.edge(e,e_n[1]))
            if not simple_wire:
                qubit = list(frontier)[list(frontier.values()).index(frontier_vertex)]
                circuit.add_gate("HAD", qubit)
            
            g.remove_vertex(e)
            g.add_edge(g.edge(e_n[0],e_n[1]),EdgeType.SIMPLE)

            # no measurements in frontier, because in original paper those are outputs
            # from graph-theoretical perspective we can maybe also see the extraction of the H wire as an implicit conversion from YZ to XY?
            g.set_mtype(frontier_vertex, MeasurementType.XY) 

            break