
from .graph.graph_mbqc import GraphMBQC
from .graph.base import VT, ET
from .simplify import MatchLcompType, MatchPivotType
from .utils import MeasurementType, VertexType, EdgeType


def lcomp(g: GraphMBQC, v: VT):
    vn = list(g.neighbors(v))
    vn.sort()

    for n in vn:
        for n2 in vn[vn.index(n)+1:]:
            if g.connected(n,n2):
                g.remove_edge(g.edge(n,n2))
            else:
                g.add_edge(g.edge(n,n2), EdgeType.HADAMARD)
        g.set_mtype(n, {
            MeasurementType.YZ: MeasurementType.XZ,
            MeasurementType.XZ: MeasurementType.YZ,
        }.get(g.mtype(n), MeasurementType.XY))
    
    g.set_mtype(v, {
        MeasurementType.XY: MeasurementType.XZ,
        MeasurementType.XZ: MeasurementType.XY,
    }.get(g.mtype(v), MeasurementType.YZ))
    
def pivot(g: GraphMBQC, u: VT, v: VT):
    lcomp(g, u)
    lcomp(g, v)
    lcomp(g, u)
