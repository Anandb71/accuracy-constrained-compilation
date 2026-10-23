"""Phase 3 - semantic analysis.

Checks scoping, typing (integer index arithmetic vs real arithmetic), call
arity, read/write discipline on inputs and outputs, loop-bound well formedness
and budget well formedness. Index bounds are checked later, during unrolling,
when every loop variable has a concrete value.
"""
import math

from .lexer import CompileError
from .parser import Assign, Bin, Call, For, Index, Let, Neg, Num, Var

# name -> arity. All builtins are scalar functions of one argument.
BUILTINS = {"exp": 1, "log": 1, "sqrt": 1, "sin": 1, "cos": 1, "tanh": 1,
            "sigmoid": 1, "relu": 1, "abs": 1}

FP32_MAX = 3.4028234663852886e38


def err(msg, line):
    raise CompileError(msg, line, None, "semantic")


def is_int_expr(e, loopvars):
    """Integer-typed expressions: loop variables, integer literals, and + - * of those."""
    if isinstance(e, Num):
        return float(e.value).is_integer() and abs(e.value) < 2 ** 31
    if isinstance(e, Var):
        return e.name in loopvars
    if isinstance(e, Bin) and e.op in "+-*":
        return is_int_expr(e.l, loopvars) and is_int_expr(e.r, loopvars)
    if isinstance(e, Neg):
        return is_int_expr(e.e, loopvars)
    return False


class Analyzer:
    def __init__(self, k, budget_override=None):
        self.k = k
        self.inputs = {}
        self.outputs = {}
        self.fns = {}
        self.warnings = []
        self.budget = budget_override if budget_override is not None else k.budget
        self.stats = {"loops": 0, "max_depth": 0, "statements": 0}

    def run(self):
        k = self.k
        if self.budget is None:
            err("no accuracy budget: add 'budget <value>;' to the kernel", 1)
        if not (self.budget > 0) or not math.isfinite(self.budget):
            err(f"budget must be a positive finite number, got {self.budget}", k.budget_line or 1)

        for d in k.inputs:
            self.declare(d.name, d.line)
            if any(n <= 0 for n in d.dims):
                err(f"input '{d.name}' has a non-positive dimension", d.line)
            if not (d.lo <= d.hi):
                err(f"input '{d.name}' has an empty range [{d.lo}, {d.hi}]", d.line)
            if max(abs(d.lo), abs(d.hi)) > FP32_MAX:
                err(f"input '{d.name}' range is not representable in fp32", d.line)
            self.inputs[d.name] = d
        for d in k.outputs:
            self.declare(d.name, d.line)
            if any(n <= 0 for n in d.dims):
                err(f"output '{d.name}' has a non-positive dimension", d.line)
            self.outputs[d.name] = d
        if not k.outputs:
            err("kernel declares no outputs; the budget has nothing to constrain", 1)
        for f in k.fns:
            self.declare(f.name, f.line)
            if f.name in BUILTINS:
                err(f"'{f.name}' shadows a builtin function", f.line)
            self.check_fn_body(f, f.body)
            self.fns[f.name] = f

        self.block(k.body, scope={}, loopvars={}, depth=0)
        return self

    def declare(self, name, line):
        if name in self.inputs or name in self.outputs or name in self.fns:
            err(f"'{name}' is declared twice", line)

    # -- user functions: pure single-argument expressions
    def check_fn_body(self, f, e):
        if isinstance(e, Num):
            return
        if isinstance(e, Var):
            if e.name != f.param:
                err(f"function '{f.name}' may only reference its parameter '{f.param}', not '{e.name}'", e.line)
            return
        if isinstance(e, Bin):
            self.check_fn_body(f, e.l); self.check_fn_body(f, e.r); return
        if isinstance(e, Neg):
            self.check_fn_body(f, e.e); return
        if isinstance(e, Call):
            if e.name not in BUILTINS and e.name not in self.fns:
                err(f"unknown function '{e.name}' in body of '{f.name}'", e.line)
            if len(e.args) != 1:
                err(f"'{e.name}' takes exactly one argument", e.line)
            self.check_fn_body(f, e.args[0]); return
        if isinstance(e, Index):
            err(f"function '{f.name}' cannot index arrays", e.line)
        err("unsupported expression in function body", getattr(e, "line", 0))

    # -- statements
    def block(self, stmts, scope, loopvars, depth):
        scope = dict(scope)
        for s in stmts:
            self.stats["statements"] += 1
            if isinstance(s, Let):
                self.expr(s.e, scope, loopvars, s.line)
                if s.name in self.inputs or s.name in self.outputs or s.name in loopvars:
                    err(f"'{s.name}' shadows an input, output or loop variable", s.line)
                scope[s.name] = True
            elif isinstance(s, Assign):
                self.expr(s.e, scope, loopvars, s.line)
                if s.idx is None:
                    if s.name in loopvars:
                        err(f"cannot assign to loop variable '{s.name}'", s.line)
                    if s.name in self.inputs:
                        err(f"cannot assign to input '{s.name}'", s.line)
                    if s.name in self.outputs:
                        if self.outputs[s.name].dims:
                            err(f"output '{s.name}' is an array and needs an index", s.line)
                        if s.op != "=":
                            err(f"compound assignment cannot read output '{s.name}'", s.line)
                        continue
                    if s.name not in scope:
                        err(f"assignment to undeclared variable '{s.name}' (use 'let')", s.line)
                else:
                    if s.name in self.inputs:
                        err(f"cannot assign to input '{s.name}'", s.line)
                    if s.name not in self.outputs:
                        err(f"'{s.name}' is not an output array", s.line)
                    if s.op != "=":
                        err(f"compound assignment cannot read output '{s.name}'", s.line)
                    self.index(s.name, self.outputs[s.name].dims, s.idx, loopvars, s.line)
            elif isinstance(s, For):
                if s.var in scope or s.var in loopvars or s.var in self.inputs or s.var in self.outputs:
                    err(f"loop variable '{s.var}' shadows another name", s.line)
                for b in (s.lo, s.hi):
                    if not is_int_expr(b, loopvars):
                        err("loop bounds must be integer expressions of constants and outer loop variables", s.line)
                self.stats["loops"] += 1
                self.stats["max_depth"] = max(self.stats["max_depth"], depth + 1)
                inner = dict(loopvars)
                inner[s.var] = True
                self.block(s.body, scope, inner, depth + 1)
            else:
                err("unsupported statement", getattr(s, "line", 0))

    def index(self, name, dims, idx, loopvars, line):
        if len(idx) != len(dims):
            err(f"'{name}' has {len(dims)} dimension(s) but was indexed with {len(idx)}", line)
        for e in idx:
            if not is_int_expr(e, loopvars):
                err(f"index into '{name}' must be an integer expression of loop variables", line)

    def expr(self, e, scope, loopvars, line):
        if isinstance(e, Num):
            return
        if isinstance(e, Var):
            if e.name in scope or e.name in loopvars:
                return
            if e.name in self.inputs:
                if self.inputs[e.name].dims:
                    err(f"input '{e.name}' is an array and needs an index", e.line)
                return
            if e.name in self.outputs:
                err(f"outputs are write-only; cannot read '{e.name}'", e.line)
            err(f"undeclared variable '{e.name}'", e.line)
        if isinstance(e, Index):
            if e.name in self.outputs:
                err(f"outputs are write-only; cannot read '{e.name}'", e.line)
            if e.name not in self.inputs:
                err(f"'{e.name}' is not an input array", e.line)
            self.index(e.name, self.inputs[e.name].dims, e.idx, loopvars, e.line)
            return
        if isinstance(e, Call):
            if e.name not in BUILTINS and e.name not in self.fns:
                err(f"unknown function '{e.name}'", e.line)
            if len(e.args) != 1:
                err(f"'{e.name}' takes exactly one argument", e.line)
            self.expr(e.args[0], scope, loopvars, line)
            return
        if isinstance(e, Bin):
            self.expr(e.l, scope, loopvars, line)
            self.expr(e.r, scope, loopvars, line)
            return
        if isinstance(e, Neg):
            self.expr(e.e, scope, loopvars, line)
            return
        err("unsupported expression", line)


def analyze(kernel, budget_override=None):
    return Analyzer(kernel, budget_override).run()
