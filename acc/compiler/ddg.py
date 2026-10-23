"""Dynamic dataflow graph.

Every loop in the language is counted and data independent, so the SSA IR can
be fully unrolled into a DAG of scalar operations. Each node remembers which
static site produced it, so a per-site approximation choice applies to all of
that site's dynamic instances, and each error term attaches to exactly one node.
"""
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .ir import Instr, Loop, Store, ast_str
from .lexer import CompileError
from .parser import Bin, Neg, Num, Var


@dataclass
class Node:
    id: int
    op: str                       # in const add sub mul div neg call
    args: Tuple[int, ...] = ()
    site: Optional[int] = None
    value: float = 0.0            # const
    input: Optional[Tuple[str, int]] = None   # (array, flat index)
    fn: str = ""
    red: Optional[Tuple[int, int, int, int]] = None  # (loop site, k, n, acc arg)


class DDG:
    def __init__(self, prog):
        self.prog = prog
        self.nodes: List[Node] = []
        self.input_nodes = {}       # (array, flat) -> node id
        self.outputs = {}           # (array, flat) -> node id
        self.const_cache = {}
        self.site_nodes = {}        # site -> [node ids]
        self.loop_instances = {}    # loop site -> number of dynamic executions
        self.loop_iters = {}        # loop site -> max trip count seen

    def add(self, **kw):
        n = Node(len(self.nodes), **kw)
        self.nodes.append(n)
        if n.site is not None:
            self.site_nodes.setdefault(n.site, []).append(n.id)
        return n.id

    def const(self, v):
        if v not in self.const_cache:
            self.const_cache[v] = self.add(op="const", value=v)
        return self.const_cache[v]

    def input(self, name, flat):
        key = (name, flat)
        if key not in self.input_nodes:
            self.input_nodes[key] = self.add(op="in", input=key)
        return self.input_nodes[key]

    def output_list(self):
        """Stable ordering of outputs: declaration order, then flat index."""
        order = []
        for d in self.prog.kernel.outputs:
            size = 1
            for n in d.dims:
                size *= n
            for f in range(size):
                if (d.name, f) in self.outputs:
                    order.append((d.name, f))
        return order


def eval_int(e, lv, line):
    if isinstance(e, Num):
        return int(e.value)
    if isinstance(e, Var):
        return lv[e.name]
    if isinstance(e, Neg):
        return -eval_int(e.e, lv, line)
    if isinstance(e, Bin):
        a, b = eval_int(e.l, lv, line), eval_int(e.r, lv, line)
        return a + b if e.op == "+" else a - b if e.op == "-" else a * b
    raise CompileError("non-integer index expression", line, None, "semantic")


def flat_index(name, dims, idx, lv, line):
    flat = 0
    for d, e in zip(dims, idx):
        v = eval_int(e, lv, line)
        if not 0 <= v < d:
            shown = ", ".join(ast_str(x) for x in idx)
            env = ", ".join(f"{k}={v2}" for k, v2 in lv.items())
            raise CompileError(f"index {name}[{shown}] out of bounds ({env})", line, None, "semantic")
        flat = flat * d + v
    return flat


MAX_NODES = 60000


def build_ddg(prog):
    g = DDG(prog)
    inputs = {d.name: d for d in prog.kernel.inputs}
    outputs = {d.name: d for d in prog.kernel.outputs}

    def run(body, env, lv, loop_ctx):
        for s in body:
            if isinstance(s, Instr):
                if s.op == "const":
                    env[s.dest] = g.const(s.value)
                elif s.op == "itof":
                    env[s.dest] = g.const(float(eval_int(s.idx[0], lv, s.line)))
                elif s.op == "load":
                    d = inputs[s.array]
                    env[s.dest] = g.input(s.array, flat_index(s.array, d.dims, s.idx, lv, s.line))
                else:
                    red = None
                    if s.red_loop is not None:
                        k, n = loop_ctx[s.red_loop]
                        red = (s.red_loop, k, n, s.red_phi_arg)
                    env[s.dest] = g.add(op=s.op, args=tuple(env[a] for a in s.args),
                                        site=s.site, fn=s.fn, red=red)
                if len(g.nodes) > MAX_NODES:
                    raise CompileError(f"unrolled program exceeds {MAX_NODES} operations; "
                                       "shrink the problem size", s.line, None, "ir")
            elif isinstance(s, Store):
                d = outputs[s.array]
                flat = flat_index(s.array, d.dims, s.idx or [], lv, s.line)
                g.outputs[(s.array, flat)] = env[s.value]
            elif isinstance(s, Loop):
                lo = eval_int(s.lo, lv, s.line)
                hi = eval_int(s.hi, lv, s.line)
                n = max(0, hi - lo)
                g.loop_instances[s.id] = g.loop_instances.get(s.id, 0) + 1
                g.loop_iters[s.id] = max(g.loop_iters.get(s.id, 0), n)
                cur = {p.dest: env[p.init] for p in s.phis}
                for k in range(n):
                    inner = dict(env)
                    inner.update(cur)
                    lv2 = dict(lv)
                    lv2[s.var] = lo + k
                    ctx = dict(loop_ctx)
                    ctx[s.id] = (k, n)
                    run(s.body, inner, lv2, ctx)
                    cur = {p.dest: inner[p.back] for p in s.phis}
                env.update(cur)

    run(prog.body, {}, {}, {})
    if not g.outputs:
        raise CompileError("the kernel never writes an output", 1, None, "semantic")
    return g


def topo_order(g):
    """Nodes are created in dependency order, so id order is topological."""
    return range(len(g.nodes))
