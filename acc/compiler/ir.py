"""Phase 4 - intermediate code generation.

Lowers the AST into a structured SSA IR: three-address instructions with one
definition per value, counted loops as regions, and loop-carried scalars as phi
nodes. Every arithmetic instruction and call is a *site*: the unit to which the
optimizer attaches an approximation choice and an error term.

Reduction loops (bodies made only of `acc = acc (+|-) term` updates whose terms
do not depend on any accumulator) are recognised here; they are the candidates
for loop perforation.
"""
from dataclasses import dataclass, field
from typing import List, Optional

from .parser import Assign, Bin, Call, For, Index, Let, Neg, Num, Var
from .semantic import is_int_expr

ARITH = {"+": "add", "-": "sub", "*": "mul", "/": "div"}


@dataclass
class Instr:
    dest: str
    op: str                 # const load itof add sub mul div neg call
    args: List[str] = field(default_factory=list)
    value: float = 0.0      # const
    array: str = ""         # load
    idx: list = None        # load: integer index ASTs ; itof: [int AST]
    fn: str = ""            # call
    site: Optional[int] = None
    line: int = 0
    red_loop: Optional[int] = None   # set on reduction updates
    red_phi_arg: int = 0             # which arg is the accumulator


@dataclass
class Phi:
    dest: str
    init: str
    back: str = ""


@dataclass
class Loop:
    id: int
    var: str
    lo: object
    hi: object
    phis: List[Phi] = field(default_factory=list)
    body: list = field(default_factory=list)
    line: int = 0
    reduction: bool = False


@dataclass
class Store:
    array: str
    idx: Optional[list]
    value: str
    line: int = 0


@dataclass
class Site:
    id: int
    kind: str        # 'arith' | 'call' | 'loop'
    op: str          # add/sub/mul/div/neg/call name/'loop'
    line: int
    text: str = ""


@dataclass
class Program:
    kernel: object
    sem: object
    body: list
    sites: dict
    fns: dict


def ast_str(e):
    if isinstance(e, Num):
        v = e.value
        return str(int(v)) if float(v).is_integer() and abs(v) < 1e15 else repr(v)
    if isinstance(e, Var):
        return e.name
    if isinstance(e, Index):
        return f"{e.name}[{', '.join(ast_str(i) for i in e.idx)}]"
    if isinstance(e, Call):
        return f"{e.name}({', '.join(ast_str(a) for a in e.args)})"
    if isinstance(e, Neg):
        return f"-{ast_str(e.e)}"
    if isinstance(e, Bin):
        return f"({ast_str(e.l)} {e.op} {ast_str(e.r)})"
    return "?"


class Lowerer:
    def __init__(self, sem):
        self.sem = sem
        self.n_tmp = 0
        self.ver = {}
        self.sites = {}
        self.next_site = 1

    def tmp(self):
        self.n_tmp += 1
        return f"t{self.n_tmp}"

    def version(self, name):
        v = self.ver.get(name, -1) + 1
        self.ver[name] = v
        return f"{name}.{v}"

    def new_site(self, kind, op, line, text=""):
        s = Site(self.next_site, kind, op, line, text)
        self.sites[s.id] = s
        self.next_site += 1
        return s.id

    # -- expressions: emit into `out`, return the SSA name holding the value
    def expr(self, e, env, loopvars, out, dest=None):
        if isinstance(e, Num):
            d = dest or self.tmp()
            out.append(Instr(d, "const", value=float(e.value), line=e.line))
            return d
        if isinstance(e, Var):
            if e.name in loopvars:
                d = dest or self.tmp()
                out.append(Instr(d, "itof", idx=[e], line=e.line))
                return d
            if e.name in env:
                return env[e.name]
            # scalar input
            d = dest or self.tmp()
            out.append(Instr(d, "load", array=e.name, idx=[], line=e.line))
            return d
        if isinstance(e, Index):
            d = dest or self.tmp()
            out.append(Instr(d, "load", array=e.name, idx=list(e.idx), line=e.line))
            return d
        if is_int_expr(e, loopvars):
            d = dest or self.tmp()
            out.append(Instr(d, "itof", idx=[e], line=getattr(e, "line", 0)))
            return d
        if isinstance(e, Neg):
            a = self.expr(e.e, env, loopvars, out)
            d = dest or self.tmp()
            out.append(Instr(d, "neg", [a], line=e.line))
            return d
        if isinstance(e, Bin):
            a = self.expr(e.l, env, loopvars, out)
            b = self.expr(e.r, env, loopvars, out)
            d = dest or self.tmp()
            op = ARITH[e.op]
            site = self.new_site("arith", op, e.line, ast_str(e))
            out.append(Instr(d, op, [a, b], site=site, line=e.line))
            return d
        if isinstance(e, Call):
            a = self.expr(e.args[0], env, loopvars, out)
            d = dest or self.tmp()
            site = self.new_site("call", e.name, e.line, ast_str(e))
            out.append(Instr(d, "call", [a], fn=e.name, site=site, line=e.line))
            return d
        raise AssertionError(e)

    # -- statements
    def block(self, stmts, env, loopvars, out):
        env = dict(env)
        for s in stmts:
            if isinstance(s, Let):
                env[s.name] = self.expr(s.e, env, loopvars, out, self.version(s.name))
            elif isinstance(s, Assign):
                e = s.e
                if s.op != "=":
                    e = Bin(s.op[0], Var(s.name, s.line), s.e, s.line)
                if s.idx is None and s.name in env:
                    env[s.name] = self.expr(e, env, loopvars, out, self.version(s.name))
                else:
                    v = self.expr(e, env, loopvars, out)
                    out.append(Store(s.name, s.idx, v, s.line))
            elif isinstance(s, For):
                out.append(self.loop(s, env, loopvars))
                # after the loop, carried variables refer to the phi (exit value)
                lp = out[-1]
                for phi in lp.phis:
                    env[phi.dest.rsplit(".", 1)[0]] = phi.dest
        return env

    def loop(self, s, env, loopvars):
        lid = self.new_site("loop", "loop", s.line, f"for {s.var} in {ast_str(s.lo)}..{ast_str(s.hi)}")
        lp = Loop(lid, s.var, s.lo, s.hi, line=s.line)
        # Every outer scalar assigned somewhere in the body is loop-carried.
        carried = sorted(n for n in assigned_scalars(s.body) if n in env)
        inner_env = dict(env)
        for name in carried:
            phi = Phi(self.version(name), env[name])
            lp.phis.append(phi)
            inner_env[name] = phi.dest
        inner_loopvars = dict(loopvars)
        inner_loopvars[s.var] = True
        end_env = self.block(s.body, inner_env, inner_loopvars, lp.body)
        for phi, name in zip(lp.phis, carried):
            phi.back = end_env[name]
        detect_reduction(lp)
        return lp


def assigned_scalars(stmts):
    names = set()
    for s in stmts:
        if isinstance(s, Assign) and s.idx is None:
            names.add(s.name)
        elif isinstance(s, For):
            names |= assigned_scalars(s.body)
    # names introduced with `let` inside the body are local to it
    lets = {s.name for s in stmts if isinstance(s, Let)}
    return names - lets


def detect_reduction(lp):
    if not lp.phis or any(not isinstance(i, Instr) for i in lp.body):
        return
    defs = {i.dest: i for i in lp.body}
    phi_names = {p.dest for p in lp.phis}
    uses = {}
    for i in lp.body:
        for a in i.args:
            uses.setdefault(a, []).append(i)
    updates = []
    for p in lp.phis:
        u = defs.get(p.back)
        if u is None or u.op not in ("add", "sub"):
            return
        if u.args.count(p.dest) != 1:
            return
        acc_pos = u.args.index(p.dest)
        if u.op == "sub" and acc_pos != 0:
            return
        if len(uses.get(p.dest, [])) != 1 or uses.get(u.dest):
            return
        updates.append((u, acc_pos))

    # terms must not depend on any accumulator
    def depends_on_phi(name, seen):
        if name in phi_names:
            return True
        if name in seen or name not in defs:
            return False
        seen.add(name)
        return any(depends_on_phi(a, seen) for a in defs[name].args)

    for u, acc_pos in updates:
        term = u.args[1 - acc_pos]
        if depends_on_phi(term, set()):
            return
    lp.reduction = True
    for u, acc_pos in updates:
        u.red_loop = lp.id
        u.red_phi_arg = acc_pos


def lower(kernel, sem):
    lw = Lowerer(sem)
    body = []
    lw.block(kernel.body, {}, {}, body)
    # perforation is only offered on reduction loops; drop other loop sites
    for lp in iter_loops(body):
        if not lp.reduction:
            lw.sites[lp.id].kind = "loop-plain"
    return Program(kernel, sem, body, lw.sites, {f.name: f for f in kernel.fns})


def iter_loops(body):
    for s in body:
        if isinstance(s, Loop):
            yield s
            yield from iter_loops(s.body)


# ---------------------------------------------------------------- printing
def format_ir(prog):
    k = prog.kernel
    lines = [f"kernel {k.name}    ; budget |error| <= {prog.sem.budget:g}"]
    for d in k.inputs:
        dims = "".join(f"[{n}]" for n in d.dims)
        lines.append(f"  input  {d.name}{dims} in [{d.lo:g}, {d.hi:g}]")
    for d in k.outputs:
        dims = "".join(f"[{n}]" for n in d.dims)
        lines.append(f"  output {d.name}{dims}")
    for f in k.fns:
        lines.append(f"  fn {f.name}({f.param}) = {ast_str(f.body)}")
    lines.append("")
    _fmt_block(prog.body, lines, 1)
    return "\n".join(lines)


def _fmt_instr(i):
    if i.op == "const":
        rhs = f"const {i.value:g}"
    elif i.op == "load":
        rhs = f"load {i.array}" + (f"[{', '.join(ast_str(x) for x in i.idx)}]" if i.idx else "")
    elif i.op == "itof":
        rhs = f"itof {ast_str(i.idx[0])}"
    elif i.op == "call":
        rhs = f"call {i.fn}({i.args[0]})"
    else:
        rhs = f"{i.op} {', '.join(i.args)}"
    note = []
    if i.site is not None:
        note.append(f"site s{i.site}")
    if i.red_loop is not None:
        note.append(f"reduction of L{i.red_loop}")
    tail = f"    ; {', '.join(note)}" if note else ""
    return f"{i.dest} = {rhs}{tail}"


def _fmt_block(body, lines, depth):
    pad = "  " * depth
    for s in body:
        if isinstance(s, Instr):
            lines.append(pad + _fmt_instr(s))
        elif isinstance(s, Store):
            idx = f"[{', '.join(ast_str(x) for x in s.idx)}]" if s.idx else ""
            lines.append(f"{pad}store {s.array}{idx} = {s.value}")
        elif isinstance(s, Loop):
            tag = "  ; reduction (perforable)" if s.reduction else ""
            lines.append(f"{pad}L{s.id}: for {s.var} in {ast_str(s.lo)}..{ast_str(s.hi)} {{{tag}")
            for p in s.phis:
                lines.append(f"{pad}  {p.dest} = phi({p.init}, {p.back})")
            _fmt_block(s.body, lines, depth + 1)
            lines.append(pad + "}")
