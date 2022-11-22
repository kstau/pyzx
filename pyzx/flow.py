from .graph.base import BaseGraph, VT, ET
from .graph.graph_mbqc import GraphMBQC
from typing import Dict, Set, List
from .utils import MeasurementType, VertexType
from .extract import bi_adj
from .linalg import Mat2, CNOTMaker

"""Generic Flow type = Tuple of correction sets and vertex depth"""
Flow = (Dict[VT, Set[VT]], Dict[VT,int])

def identify_causal_flow(g: BaseGraph[VT,ET]) -> Flow:
    """Function for calculating the maximally delayed causal flow in a graph-like ZX-Diagram
    Nearly identical to tket version, but with reversed ordering
    Algorithm is taken from https://arxiv.org/pdf/0709.2670.pdf"""
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


def identify_xy_gflow(g: BaseGraph[VT, ET]) -> Flow:
    """Compute the maximally delayed gflow of a diagram in graph-like form where every spider is measured in XY plane.

    Based on algorithm by Perdrix and Mhalla.
    See dx.doi.org/10.1007/978-3-540-70575-8_70
    """

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
                if len(processed) + len(inputs) != g.num_vertices():
                    return None
                inv_depth = inverse_depth(res[1])
                return (res[0], inv_depth)
            return None
        else:
            processed.update(correct)
            k += 1

def get_gauss_solution(gauss: Mat2, vec: Mat2, cnots: CNOTMaker):
    """helper function for solving M * x = b if additions (cnots) for getting M in echelon form are already calculated"""
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

def inverse_depth(depth: Dict) -> Dict:
    """helper function for inversing the depth dictionary of the flow algorithms"""
    # reverse ordering of vertices
    max_depth = depth[max(depth, key=depth.get)]
    inv_depth = dict()
    for k,v in depth.items():
        inv_depth[k] = max_depth-v
    return inv_depth

def identify_gflow(g: GraphMBQC) -> Flow:
    """Compute maximally delayed gflow of a graph-like diagram as in https://arxiv.org/pdf/2003.01664.pdf"""
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

def identify_pauli_flow(g: GraphMBQC) -> Flow:
    """Compute maximally delayed pauli flow as in https://arxiv.org/pdf/2109.05654.pdf"""
    res: Flow = (dict(), dict())
    solved = []
    correctors = []
    inputs = []
    for input in g.inputs():
        n = list(g.neighbors(input))[0]
        inputs.append(n)

    for output in g.outputs():
        solved.append(output)
        n = list(g.neighbors(output))[0]
        if not n in g.inputs():
            solved.append(n)
            res[0][n] = set()
            res[1][n] = 0
    
    for v in g.non_outputs():
        if not v in g.inputs() and g.mtype(v) in [MeasurementType.X, MeasurementType.Y]:
            correctors.append(v)

    depth = 1

    while True:
        new_corrections = solve_pauli_correctors(g, solved, correctors)
        if not new_corrections:
            break
        for v,c_s in new_corrections.items():
            res[0][v] = c_s
            res[1][v] = depth
            solved.append(v)
            if not v in inputs:
                correctors.append(v)
        depth += 1
    if len(solved) + len(inputs) != len(g.vertices()):
        return False
    return res 


def solve_pauli_correctors(g: GraphMBQC, solved: list[int], correctors: list[int]):
    """Helper function for pauli flow identification"""
    to_solve = []
    unsolved_ys = []
    preserve = []

    for v in g.non_inputs():
        if not v in solved:
            to_solve.append(v)
            if g.mtype(v) == MeasurementType.Y:
                unsolved_ys.append(v)
            elif g.mtype(v) != MeasurementType.Z:
                preserve.append(v)
    
    mat = Mat2.zeros(len(preserve) + len(unsolved_ys), len(correctors) + len(to_solve))

    # fill lhs, aka. M A,u
    for corrector in correctors:
        for n in g.neighbors(corrector):
            if n in preserve:
                mat.data[preserve.index(n)][correctors.index(corrector)] = 1
            else:
                if n in unsolved_ys:
                    mat.data[len(preserve) + unsolved_ys.index(n)][correctors.index(corrector)] = 1
    
    for unsolved_y in unsolved_ys:
        if unsolved_y in correctors:
            mat.data[len(preserve) + unsolved_ys.index(unsolved_y)][correctors.index(unsolved_y)] = 1
    
    #Fill rhs, aka. S lambda
    for candidate in to_solve:
        mtype = g.mtype(candidate)
        if mtype in [MeasurementType.XY, MeasurementType.X, MeasurementType.XZ]:
            mat.data[preserve.index(candidate)][len(correctors) + to_solve.index(candidate)] = 1
        if mtype in [MeasurementType.XZ, MeasurementType.YZ, MeasurementType.Z]:
            for n in g.neighbors(candidate):
                if n in preserve:
                    mat.data[preserve.index(n)][len(correctors) + to_solve.index(candidate)] = 1
                else:
                    if n in unsolved_ys:
                        mat.data[len(preserve) + unsolved_ys.index(n)][len(correctors) + to_solve.index(candidate)] = 1
        if mtype == MeasurementType.Y:
            mat.data[len(preserve) + unsolved_ys.index(candidate)][len(correctors) + to_solve.index(candidate)] = 1

    # gaussian elimination
    sub_matrix = Mat2([[mat.data[i][j] for j in range(len(correctors))] for i in range(len(preserve) + len(unsolved_ys))])
    cnot_maker = CNOTMaker()
    sub_matrix.gauss(x=cnot_maker, full_reduce=True)

    for cnot in cnot_maker.cnots:
        mat.row_add(cnot.target,cnot.control)
    
    #Back substitution
    row_correctors = dict()
    for i in range(sub_matrix.rows()):
        for j in range(sub_matrix.cols()):
            if j >= len(mat.data[i]):
                import pdb
                pdb.set_trace()
            if mat.data[i][j]:
                row_correctors[i] = correctors[j]
    
    solved_flow = dict()
    
    for i in range(len(to_solve)):
        fail = False
        c_i = set()
        for j in range(len(preserve) + len(unsolved_ys)):
            if mat.data[j][len(correctors) + i]:
                if j in row_correctors.keys():
                    c_i.add(row_correctors[j])
                else:
                    fail = True
                    break
        if not fail:
            v = to_solve[i]
            if g.mtype(v) in [MeasurementType.XZ, MeasurementType.YZ, MeasurementType.Z]:
                c_i.add(v)
            solved_flow[v] = c_i

    return solved_flow

def focus(g: GraphMBQC, flow: Flow) -> Flow:
    outputs = [list(g.neighbors(output))[0] for output in g.outputs()]

    order: Dict(int, List) = dict()
    for v, d in flow[1].items():
        if d in order.keys():
            order[d].append(v)
        else:
            order[d] = [v]
    
    for d, vertices in order.items():
        for v in vertices:
            corrections = flow[0][v]
            odd_n = get_odd_nh(g, corrections) #TODO: Update this function
            parities = dict()
            for correction in corrections:
                parities[correction] = 1
            for correction in corrections:
                if correction == v or g.mtype(correction) in [MeasurementType.XY, MeasurementType.X]:
                    continue
                if g.mtype(correction) == MeasurementType.Y and correction in odd_n:
                    continue
                for w in flow[0][correction]:
                    if w in parities:
                        parities[w] += 1
                    else:
                        parities[w] = 1
            
            for w in odd_n:
                if v == w or w in outputs or g.mtype(w) in [MeasurementType.XZ, MeasurementType.YZ, MeasurementType.Z]:
                    continue
                if g.mtype(w) == MeasurementType.Y and w in corrections:
                    continue
                for correction in flow[0][w]:
                    if correction in parities.keys():
                        parities[correction] += 1
                    else:
                        parities[correction] = 1
            new_c = set()
            for w, parity in parities.items():
                if parity % 2 == 1:
                    new_c.add(w)
            flow[0][v] = new_c

def get_odd_nh(g: GraphMBQC, vertex_set):
    """Calculates Odd Neighborhood as in http://arxiv.org/abs/1610.02824v2"""
    odd_n = set()
    for v in vertex_set:
        odd_n.symmetric_difference_update(set(g.neighbors(v)).difference(set(g.outputs())).difference(set(g.effects())))
    return odd_n

def check_focussed_property(g: GraphMBQC, flow: Flow):
    for v, corrections in flow[0].items():
        # FX
        to_check1 = corrections.difference(set(g.outputs())).difference(set(v))
        if not all([g.mtype(w) in [MeasurementType.XY, MeasurementType.X, MeasurementType.Y] for w in to_check1]):
            print("A vertex in the correction set of ",v," is not measured in XY, X or Y plane")
            return False
        #FZ
        to_check2 = get_odd_nh(g, corrections).difference(set(g.outputs())).difference(set(v))
        if not all([g.mtype(w) in [MeasurementType.XZ, MeasurementType.YZ, MeasurementType.Y, MeasurementType.Z] for w in to_check2]):
            print("A vertex in the odd neighborhood of the correction set of ",v," is not measured in XZ, YZ, Y or Z plane")
            return False
        #FY
        # ys_1 = filter()
        # if not to_check1.intersection([])

    return True
            

## For Testing, may be outdated

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