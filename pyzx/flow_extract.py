from .graph.base import BaseGraph, VT, ET
from .graph.graph_mbqc import GraphMBQC
from typing import Dict, Set, List
from .utils import MeasurementType, EdgeType, insert_identity
from .circuit import Circuit
from .extract import graph_to_swaps, bi_adj
from .linalg import CNOTMaker
from .flow_rules import lcomp, pivot
from .flow import Flow, get_odd_nh, focus, identify_pauli_flow
from .simplify import clifford_simp
from fractions import Fraction

#debug
from .tensor import compare_tensors
from .drawing import draw

def extract_from_causal_flow(g: BaseGraph[VT, ET]) -> Circuit:
    """Extracts a circuit from graph-like diagrams with causal flow"""
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

def extract_from_xy_gflow(g: BaseGraph[VT, ET]) -> Circuit:
    """Extracts a circuit from graph-like diagrams with XY-gflow, i.e. all spiders are measured in XY-plane"""
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

def extract_from_gflow(g: GraphMBQC) -> Circuit:
    """Extracts a circuit from graph-like diagrams with gflow, i.e. all spiders are measured in XY, XZ or YZ plane"""
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

def extract_from_pauli_flow(g: GraphMBQC) -> Circuit:
    circuit = Circuit(len(g.outputs())) 
    pauli_flow = identify_pauli_flow(g)
    focus(g, pauli_flow)
    
    order = dict()
    for v, d in pauli_flow[1].items():
        if d in order.keys():
            order[d].append(v)
        else:
            order[d] = [v]
    
    for depth in range(0,len(order.keys())):
        for v in order[depth]:
            if g.mtype(v) in [MeasurementType.XY, MeasurementType.XZ, MeasurementType.YZ]:
                extraction_str = get_primary_extraction_string(g, v, pauli_flow)
                extract_pauli_gadget(g, v, extraction_str, circuit)
    
    circuit.gates = list(reversed(circuit.gates))
    draw(g, labels=True)
    g_orig = g.copy()
    clifford_simp(g) 

    # import pdb
    # pdb.set_trace()

    circuit2 = extract_from_xy_gflow(g) #TODO: Pauli flow allows for patterns where |I| != |O|
    assert(compare_tensors(circuit2, g_orig))

    return circuit2 + circuit


def init_frontier(g: BaseGraph[VT, ET], circuit: Circuit) -> Dict[int,VT]:
    """Inits the frontier of a ZX-diagram with the spiders adjacent to the outputs. Extracts Hadamard wires between outputs and frontier"""
    frontier: Dict[int,VT] = dict()

    for i, o in enumerate(g.outputs()):
        v = list(g.neighbors(o))[0]
        if not v in g.inputs():
            frontier[i] = v
            if g.edge_type(g.edge(v,o)) == EdgeType.HADAMARD:
                circuit.add_gate("HAD", i)
                g.set_edge_type(g.edge(v,o),EdgeType.SIMPLE)
    
    return frontier

def extract_rzs(g: BaseGraph[VT, ET], frontier: Dict[int,VT], circuit: Circuit):
    """Extracts phases of frontier spiders as ZPhase gates and updates the diagram"""
    for qubit,v in frontier.items():
        phase = g.phase(v)
        if phase != 0:
            g.set_phase(v,0)
            circuit.add_gate("ZPhase", qubit, phase)

def extract_czs(g: BaseGraph[VT, ET], frontier: Dict[int,VT], circuit: Circuit):
    """Extracts connected frontier spiders as controlled Z gates and updates the diagram"""
    for qubit, v in frontier.items():
        for w in set(g.neighbors(v)).intersection(set(frontier.values())):
            g.remove_edge(g.edge(v,w))
            circuit.add_gate("CZ", qubit, list(frontier.keys())[list(frontier.values()).index(w)])

def process_frontier(g: BaseGraph[VT, ET], frontier: Dict[int,VT], circuit: Circuit):
    """Processes all frontier spiders which are only connected to a single non-output spider in the diagram by extracting Hadamards.
    Returns new frontier vertices"""
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

def process_hadamards(g: BaseGraph[VT, ET], inputs, processed, v):
    """Helper function for process_frontier: Finds chains of Hadamard + empty 2-ary Z spiders starting from a frontier vertex
    Returns: list of 2-ary spiders + number of Hadamard wires in the chain"""
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

def get_frontier_neighbors(g: BaseGraph[VT, ET], frontier: Dict[int,VT]):
    """Returns all non-output neighbors of a frontier"""
    frontier_neighbors = set()
    for v in frontier.values():
        frontier_neighbors.update(set(g.neighbors(v)).difference(set(g.outputs())))
    return frontier_neighbors

def extract_cnots(g: BaseGraph[VT, ET], frontier: Dict[int,VT], circuit: Circuit, cnot_maker: CNOTMaker):
    """Extracts CNOT gates resulting from gaussian elimination to circuit and adds the Hadamard wires of the corresponding frontier vertices"""
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

def eliminate_xz_spiders(g: GraphMBQC):
    """Repeatedly applies local complementation on all XZ spiders in a diagram which thereby become XY spiders"""
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

def eliminate_yz_spider(g: GraphMBQC, frontier: Dict[int,VT], frontier_neighbors: Set, circuit: Circuit):
    """Finds a YZ measured spider which is connected to a spider in frontier and applies a pivot on them. 
    By that, the YZ spider transforms to a XY spider"""
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

def get_primary_extraction_string(g: GraphMBQC, v: VT, flow: Flow):
    """determines primary extraction string over the outputs according to Definition 4.2. of https://arxiv.org/pdf/2109.05654.pdf"""
    corrections = flow[0][v]
    odd_n = get_odd_nh(g, corrections)
    outputs = [v for v in flow[0].keys() if len(flow[0][v]) == 0]
    print(outputs)
    extraction_string = ""
    for output in outputs:
        if output in corrections:
            if output in odd_n:
                extraction_string += 'Y'
            else:
                extraction_string += 'X'
        elif output in odd_n:
            extraction_string += 'Z'
        else:
            extraction_string += 'I'

    return extraction_string

def extract_pauli_gadget(g: GraphMBQC, v: VT, extraction_string: str, circuit: Circuit):
    print(extraction_string)
    for qubit, extraction in enumerate(extraction_string):
        if extraction == 'X':
            circuit.add_gate("HAD", qubit)
        elif extraction == 'Y':
            circuit.add_gate("XPhase", qubit, -Fraction(1,2))

    last_non_identity = None
    for qubit, extraction in enumerate(extraction_string):
        if extraction != 'I':
            if last_non_identity != None:
                circuit.add_gate("CNOT",last_non_identity, qubit)
            last_non_identity = qubit
    
    phase_vertex = v if g.mtype(v) == MeasurementType.XY else g.effect(v)
    circuit.add_gate("ZPhase",last_non_identity, g.phase(phase_vertex) if g.mtype(v) == MeasurementType.YZ else g.phase(phase_vertex))
    print("add zphase on ",last_non_identity, g.phase(phase_vertex) if g.mtype(v) == MeasurementType.YZ else g.phase(phase_vertex))
    g.set_phase(phase_vertex,0)

    last_non_identity = None
    for qubit, extraction in reversed(list(enumerate(extraction_string))):
        if extraction != 'I':
            if last_non_identity != None:
                circuit.add_gate("CNOT",qubit, last_non_identity)
            last_non_identity = qubit
    
    for qubit, extraction in reversed(list(enumerate(extraction_string))):
        if extraction == 'X':
            circuit.add_gate("HAD", qubit)
        elif extraction == 'Y':
            circuit.add_gate("XPhase", qubit, Fraction(1,2))
    
