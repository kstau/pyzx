from .graph.base import BaseGraph, VT, ET
from .graph.graph_mbqc import GraphMBQC
from typing import Dict, Set
from .utils import MeasurementType, EdgeType, insert_identity, VertexType
from .circuit import Circuit
from .extract import graph_to_swaps, bi_adj
from .linalg import CNOTMaker
from .flow_rules import lcomp, pivot
from .flow import Flow, get_odd_nh, focus, identify_pauli_flow
from .simplify import clifford_simp
from fractions import Fraction
import itertools

#debug
from .tensor import compare_tensors
from .drawing import draw
from .flow import check_focussed_property, check_pauli_flow, check_focussed_2

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
    """Extracts a circuit from graph-like diagrams with pauli flow.
    Currently limited to diagrams where the number of inputs is equal to the number of outputs"""
    circuit = Circuit(len(g.outputs())) 
    pauli_flow = identify_pauli_flow(g)
    focus(g, pauli_flow)

    frontier: Dict[int,VT] = init_frontier(g, circuit)
    
    order = dict()
    for v, d in pauli_flow[1].items():
        if d in order.keys():
            order[d].append(v)
        else:
            order[d] = [v]

    for depth in range(0,len(order.keys())):
        for v in order[depth]:
            if g.mtype(v) in [MeasurementType.XY, MeasurementType.XZ, MeasurementType.YZ]:
                extract_pauli_gadget(g, v, pauli_flow, circuit, frontier)

    circuit.gates = list(reversed(circuit.gates))

    clifford_simp(g)

    circuit2 = extract_from_xy_gflow(g.copy()) #TODO: Pauli flow allows for patterns where |I| != |O|

    return circuit2 + circuit

def extract_from_pauli_flow_generic(g: BaseGraph) -> Circuit:
    """Extracts a circuit from graph-like diagrams with pauli flow.
    Currently limited to diagrams where the number of inputs is equal to the number of outputs.
    Works for any backend"""
    g_mbqc = g.copy(backend='mbqc')
    convert_to_extractable_graph(g_mbqc)
    relabel_pauli_measurements(g_mbqc)
    return extract_from_pauli_flow(g_mbqc)

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

def get_primary_extraction_string(g: GraphMBQC, v: VT, flow: Flow, frontier: Dict[int,VT]):
    """determines primary extraction string over the outputs according to Definition 4.2. of https://arxiv.org/pdf/2109.05654.pdf"""
    corrections = flow[0][v]
    odd_n = get_odd_nh(g, corrections)
    extraction_string = ['I' for _ in range(0,len(frontier.keys()))]
    for i, output in frontier.items():
        if output in corrections:
            if output in odd_n:
                extraction_string[i] = 'Y'
            else:
                extraction_string[i] = 'X'
        elif output in odd_n:
            extraction_string[i] = 'Z'

    return extraction_string

def calculate_extraction_string_sign(g: GraphMBQC, v: VT, flow: Flow):
    """Calculates extraction string sign as in Lemma C.2. of https://arxiv.org/pdf/2109.05654.pdf
    Two modifications are made compared to the paper:
    1: YZ vertices do not need an additional phase flip
    2. Edges between vertices in the correction set are only counted if they are not adjacent to an output vertex"""
    a = 0 # How many vertices in the correction set are connected?
    corrections = list(flow[0][v].difference(g.moutputs()))
    for i, w in enumerate(corrections):
        for x in corrections[i:]:
            if g.connected(w,x):
                a += 1
    b = len(flow[0][v].intersection(get_odd_nh(g,flow[0][v])))/2 #XZ corrections divided by 2
    c1 = flow[0][v].union(get_odd_nh(g,flow[0][v])) 
    c2 = [w for w in g.non_outputs() if g.mtype(w) in [MeasurementType.X, MeasurementType.Y, MeasurementType.Z] and get_measurement_angle(g,w) == 1]
    c = len(c1.intersection(set(c2))) # Pauli Pi vertices

    return 1 if (a+b+c) % 2 == 0 else -1


def get_measurement_angle(g: GraphMBQC, v: VT):
    """Helper function for determining measurement angle from ZX vertex"""
    mt = g.mtype(v)
    if mt in [MeasurementType.XY]:
        return -g.phase(v)
    elif mt == MeasurementType.XZ:
        return g.phase(g.effect(v))
    elif mt == MeasurementType.YZ:
        return -g.phase(g.effect(v))
    elif mt == MeasurementType.X:
        return 0 if g.phase(v) == 0 else 1
    elif mt == MeasurementType.Y:
        return 0 if (g.phase(v) - Fraction(1,2)) == 0 else 1
    elif mt == MeasurementType.Z:
        return 0 if g.phase(g.effect(v)) == 0 else 1
    else:
        print("Error in get_measurement_angle; vertex has no measurement plane")
        return None

def extract_pauli_gadget(g: GraphMBQC, v: VT, flow: Flow, circuit: Circuit, frontier: Dict[int,VT]):
    extraction_string = get_primary_extraction_string(g, v, flow, frontier)

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
    sign = calculate_extraction_string_sign(g,v,flow)

    circuit.add_gate("ZPhase",last_non_identity, g.phase(phase_vertex) if sign == 1 else -g.phase(phase_vertex))
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

def relabel_pauli_measurements(g: GraphMBQC):
    """assigns measurement type to all Z vertices of a graph-like diagram"""
    # first set all Z measurements (recognizable via 1-ary spiders)
    for v in g.vertices():
        if g.type(v) != VertexType.Z:
            continue #no assignment on boundary vertices
        neighbors = g.neighbors(v)
        if len(neighbors) == 1: #has to be an XZ,YZ or Z effect
            n = list(neighbors)[0]
            g.set_mtype(v, MeasurementType.EFFECT)
            if g.phase(n) == 0 or g.phase(n) == Fraction(1,1):
                if g.phase(v) == 0 or g.phase(v) == Fraction(1,1):
                    g.set_mtype(n, MeasurementType.Z)
                else:
                    g.set_mtype(n, MeasurementType.YZ)
            elif g.phase(n) == Fraction(1,2) or g.phase(n) == Fraction(3,2):
                g.set_mtype(n, MeasurementType.XZ)
            else:
                raise Exception("Found gadget like root without Clifford phase")
    # then set all X,Y measurements
    for v in g.vertices():
        if g.type(v) != VertexType.Z:
            continue #no assignment on boundary vertices
        if g.mtype(v) != MeasurementType.XY:
            continue #vertex already processed (Z measurements)
        if g.phase(v) == 0 or g.phase(v) == Fraction(1,1):
            g.set_mtype(v, MeasurementType.X)
        elif g.phase(v) == Fraction(1,2) or g.phase(v) == Fraction(3,2):
            g.set_mtype(v, MeasurementType.Y)
        else:
            g.set_mtype(v, MeasurementType.XY)

def convert_to_extractable_graph(g: GraphMBQC):
    """Converts diagram to equivalent one where all outputs are measured on the X axis"""
    for output in g.outputs():
        n = list(g.neighbors(output))[0]
        if g.edge_type(g.edge(output,n)) == EdgeType.SIMPLE and (g.phase(n) == 0 or g.phase(n) == Fraction(1,1)):
            continue #we can already use this as output
        x_vertex = g.add_vertex(VertexType.Z, g.qubit(n), g.row(n),0) #TODO: adjust row
        g.set_mtype(x_vertex, MeasurementType.X)
        if g.edge_type(g.edge(output,n)) == EdgeType.HADAMARD:
            g.add_edge(g.edge(x_vertex, output), EdgeType.SIMPLE)
        else:
            g.add_edge(g.edge(x_vertex, output), EdgeType.HADAMARD)
        
        g.add_edge(g.edge(n, x_vertex), EdgeType.HADAMARD)
        g.remove_edge(g.edge(output,n))    



#debugging stuff, may be outdated
    
def assign_pauli_xy_measurements(g: GraphMBQC):
    for v in g.vertices():
        if not v in g.inputs() and not v in g.outputs():
            if g.phase(v) == Fraction(1,1) or g.phase(v) == 0:
                g.set_mtype(v, MeasurementType.X)
            elif g.phase(v) == Fraction(1,2) or g.phase(v) == Fraction(3,2):
                g.set_mtype(v, MeasurementType.Y)
            else:
                g.set_mtype(v, MeasurementType.XY)


def test_extraction_string(g: GraphMBQC, v: VT, angle: Fraction, extraction_string):
    g_orig = g.copy()
    g_temp = g.copy()
    valid_extraction_sign = 1
    circuit = Circuit(len(g_temp.outputs()))
    for sign in [1,-1]:
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
        
        phase_vertex = v if g_temp.mtype(v) == MeasurementType.XY else g.effect(v)
        circuit.add_gate("ZPhase",last_non_identity, sign*angle)
        # print("add zphase on ",last_non_identity, g.phase(phase_vertex) if sign == -1 else -g.phase(phase_vertex))
        g_temp.set_phase(phase_vertex,0)

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

        test_graph = circuit.copy()
        test_graph.gates = list(reversed(test_graph.gates))
        test_graph = test_graph.to_graph()
        if compare_tensors(g_orig, g_temp+test_graph):
            print("tested extraction string",extraction_string,sign,angle)
            valid_extraction_sign = sign
            # import pdb
            # pdb.set_trace()
            # return True
        g_temp = g_orig.copy()
        circuit = Circuit(len(g_temp.outputs()))
    return valid_extraction_sign
    # print("No pauli gadget with angle",angle,"could be extracted!")
    # return False        

def test_all_extraction_strings(g: GraphMBQC, v: VT, angle: Fraction):
    """Iterates through all possible Pauli gadgets over all wires and checks whether extracting one of them preserves the linear map 
    when we set the angle of v to 0 and the angle of the pauli gadget to the parameter, i.e. the original angle of v"""
    g_orig = g.copy()
    g_temp = g.copy()
    found_valid_extraction_string = False
    
    for extraction_string in list(itertools.product(['I','X','Y','Z'],repeat=len(g_temp.outputs())))[1:]:
        circuit = Circuit(len(g_temp.outputs())) 
        # sign = calculate_extraction_string_sign_zx(g,v)
        # print(v, extraction_string)
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
        
        phase_vertex = v if g_temp.mtype(v) == MeasurementType.XY else g.effect(v)
        circuit.add_gate("ZPhase",last_non_identity, angle)
        # print("add zphase on ",last_non_identity, g.phase(phase_vertex) if sign == -1 else -g.phase(phase_vertex))
        g_temp.set_phase(phase_vertex,0)

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

        test_graph = circuit.copy()
        test_graph.gates = list(reversed(test_graph.gates))
        test_graph = test_graph.to_graph()
        if compare_tensors(g_orig, g_temp+test_graph):
            print(extraction_string)
            found_valid_extraction_string = True
            # import pdb
            # pdb.set_trace()
            # return True
        g_temp = g_orig.copy()
    if found_valid_extraction_string:
        return True
    print("No pauli gadget with angle",angle,"could be extracted!")
    return False

def extract_single_pauli_gadget(g: GraphMBQC, v: VT, pauli_flow: Flow):
    circuit = Circuit(len(g.outputs())) 

    frontier: Dict[int,VT] = init_frontier(g, circuit)
    extract_pauli_gadget(g, v, pauli_flow, circuit, frontier)
    circuit.gates = list(reversed(circuit.gates))
    return circuit