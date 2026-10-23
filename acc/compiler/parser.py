"""Phase 2 - syntax analysis.

Recursive descent parser producing an abstract syntax tree.

    kernel   := 'kernel' ID '{' item* '}'
    item     := 'budget' num ';'
              | 'input'  ID dims? 'in' '[' num ',' num ']' ';'
              | 'output' ID dims? ';'
              | 'fn' ID '(' ID ')' '=' expr ';'
              | stmt
    stmt     := 'let' ID '=' expr ';'
              | lvalue ('=' | '+=' | '-=' | '*=') expr ';'
              | 'for' ID 'in' expr '..' expr '{' stmt* '}'
    expr     := term (('+' | '-') term)*
    term     := unary (('*' | '/') unary)*
    unary    := '-' unary | primary
    primary  := num | ID | ID '[' expr,* ']' | ID '(' expr,* ')' | '(' expr ')'
"""
from dataclasses import dataclass, field
from typing import List, Optional

from .lexer import CompileError, tokenize


# ---------------------------------------------------------------- AST nodes
@dataclass
class Num:
    value: float
    line: int = 0


@dataclass
class Var:
    name: str
    line: int = 0


@dataclass
class Index:
    name: str
    idx: list
    line: int = 0


@dataclass
class Call:
    name: str
    args: list
    line: int = 0


@dataclass
class Bin:
    op: str
    l: object
    r: object
    line: int = 0


@dataclass
class Neg:
    e: object
    line: int = 0


@dataclass
class Let:
    name: str
    e: object
    line: int = 0


@dataclass
class Assign:
    name: str
    idx: Optional[list]
    op: str
    e: object
    line: int = 0


@dataclass
class For:
    var: str
    lo: object
    hi: object
    body: list
    line: int = 0


@dataclass
class InputDecl:
    name: str
    dims: List[int]
    lo: float
    hi: float
    line: int = 0


@dataclass
class OutputDecl:
    name: str
    dims: List[int]
    line: int = 0


@dataclass
class FnDecl:
    name: str
    param: str
    body: object
    line: int = 0


@dataclass
class Kernel:
    name: str
    budget: Optional[float]
    inputs: List[InputDecl] = field(default_factory=list)
    outputs: List[OutputDecl] = field(default_factory=list)
    fns: List[FnDecl] = field(default_factory=list)
    body: list = field(default_factory=list)
    budget_line: int = 0


# ---------------------------------------------------------------- parser
class Parser:
    def __init__(self, toks):
        self.toks = toks
        self.i = 0

    @property
    def tok(self):
        return self.toks[self.i]

    def err(self, msg, tok=None):
        t = tok or self.tok
        raise CompileError(msg, t.line, t.col, "parser")

    def accept(self, text):
        if self.tok.text == text and self.tok.kind in ("op", "kw"):
            self.i += 1
            return True
        return False

    def expect(self, text):
        if not self.accept(text):
            got = self.tok.text or "end of input"
            self.err(f"expected '{text}' but found '{got}'")
        return self.toks[self.i - 1]

    def ident(self):
        if self.tok.kind != "id":
            self.err(f"expected an identifier but found '{self.tok.text or 'end of input'}'")
        t = self.tok
        self.i += 1
        return t.text

    def signed_number(self):
        neg = self.accept("-")
        if self.tok.kind != "num":
            self.err("expected a number")
        v = self.tok.value
        self.i += 1
        return -v if neg else v

    def int_literal(self):
        if self.tok.kind != "num" or not self.tok.text.isdigit():
            self.err("expected an integer dimension")
        v = int(self.tok.text)
        self.i += 1
        return v

    # -- top level
    def kernel(self):
        self.expect("kernel")
        name = self.ident()
        self.expect("{")
        k = Kernel(name, None)
        while not self.accept("}"):
            if self.tok.kind == "eof":
                self.err("unexpected end of input, missing '}'")
            self.item(k)
        if self.tok.kind != "eof":
            self.err("unexpected text after kernel")
        return k

    def dims(self):
        dims = []
        if self.accept("["):
            dims.append(self.int_literal())
            while self.accept(","):
                dims.append(self.int_literal())
            self.expect("]")
        return dims

    def item(self, k):
        t = self.tok
        if self.accept("budget"):
            if k.budget is not None:
                self.err("budget declared twice", t)
            k.budget = self.signed_number()
            k.budget_line = t.line
            self.expect(";")
        elif self.accept("input"):
            name = self.ident()
            dims = self.dims()
            self.expect("in")
            self.expect("[")
            lo = self.signed_number()
            self.expect(",")
            hi = self.signed_number()
            self.expect("]")
            self.expect(";")
            k.inputs.append(InputDecl(name, dims, lo, hi, t.line))
        elif self.accept("output"):
            name = self.ident()
            dims = self.dims()
            self.expect(";")
            k.outputs.append(OutputDecl(name, dims, t.line))
        elif self.accept("fn"):
            name = self.ident()
            self.expect("(")
            param = self.ident()
            self.expect(")")
            self.expect("=")
            body = self.expr()
            self.expect(";")
            k.fns.append(FnDecl(name, param, body, t.line))
        else:
            k.body.append(self.stmt())

    def stmt(self):
        t = self.tok
        if self.accept("let"):
            name = self.ident()
            self.expect("=")
            e = self.expr()
            self.expect(";")
            return Let(name, e, t.line)
        if self.accept("for"):
            var = self.ident()
            self.expect("in")
            lo = self.expr()
            self.expect("..")
            hi = self.expr()
            self.expect("{")
            body = []
            while not self.accept("}"):
                if self.tok.kind == "eof":
                    self.err("unexpected end of input inside loop body")
                body.append(self.stmt())
            return For(var, lo, hi, body, t.line)
        if self.tok.kind == "id":
            name = self.ident()
            idx = None
            if self.accept("["):
                idx = self.expr_list("]")
            for op in ("=", "+=", "-=", "*="):
                if self.accept(op):
                    e = self.expr()
                    self.expect(";")
                    return Assign(name, idx, op, e, t.line)
            self.err("expected an assignment operator")
        self.err(f"expected a statement but found '{self.tok.text or 'end of input'}'")

    def expr_list(self, close):
        items = [self.expr()]
        while self.accept(","):
            items.append(self.expr())
        self.expect(close)
        return items

    # -- expressions
    def expr(self):
        e = self.term()
        while self.tok.kind == "op" and self.tok.text in ("+", "-"):
            t = self.tok
            self.i += 1
            e = Bin(t.text, e, self.term(), t.line)
        return e

    def term(self):
        e = self.unary()
        while self.tok.kind == "op" and self.tok.text in ("*", "/"):
            t = self.tok
            self.i += 1
            e = Bin(t.text, e, self.unary(), t.line)
        return e

    def unary(self):
        t = self.tok
        if self.accept("-"):
            inner = self.unary()
            if isinstance(inner, Num):
                return Num(-inner.value, t.line)
            return Neg(inner, t.line)
        return self.primary()

    def primary(self):
        t = self.tok
        if t.kind == "num":
            self.i += 1
            return Num(t.value, t.line)
        if self.accept("("):
            e = self.expr()
            self.expect(")")
            return e
        if t.kind == "id":
            name = self.ident()
            if self.accept("["):
                return Index(name, self.expr_list("]"), t.line)
            if self.accept("("):
                return Call(name, self.expr_list(")"), t.line)
            return Var(name, t.line)
        self.err(f"expected an expression but found '{t.text or 'end of input'}'")


def parse(src):
    return Parser(tokenize(src)).kernel()
