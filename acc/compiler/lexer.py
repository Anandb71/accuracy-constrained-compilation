"""Phase 1 - lexical analysis.

Turns kernel source text into a flat token list. The language is small: numeric
literals, identifiers, a handful of keywords (including the `budget` annotation)
and single/double character punctuation.
"""
from dataclasses import dataclass

KEYWORDS = {"kernel", "budget", "input", "output", "in", "for", "let", "fn"}

# Longest match first.
PUNCT = ["..", "+=", "-=", "*=", "{", "}", "(", ")", "[", "]", ",", ";", ":",
         "=", "+", "-", "*", "/"]


class CompileError(Exception):
    """Any user-facing compile error. Carries a source position when known."""

    def __init__(self, msg, line=None, col=None, phase=None):
        super().__init__(msg)
        self.msg = msg
        self.line = line
        self.col = col
        self.phase = phase

    def to_dict(self):
        return {"message": self.msg, "line": self.line, "col": self.col, "phase": self.phase}

    def __str__(self):
        where = f"{self.line}:{self.col}: " if self.line else ""
        return f"{where}{self.msg}"


@dataclass
class Token:
    kind: str   # 'num', 'id', 'kw', 'op', 'eof'
    text: str
    line: int
    col: int

    @property
    def value(self):
        return float(self.text)


def tokenize(src):
    toks = []
    i, line, col = 0, 1, 1
    n = len(src)
    while i < n:
        c = src[i]
        if c == "\n":
            i += 1; line += 1; col = 1
            continue
        if c in " \t\r":
            i += 1; col += 1
            continue
        if src.startswith("//", i) or c == "#":
            while i < n and src[i] != "\n":
                i += 1
            continue
        start_col = col
        # Numbers: 12, 1.5, .5, 1e-3, 2.5E+4. A '..' range operator must not be eaten.
        if c.isdigit() or (c == "." and i + 1 < n and src[i + 1].isdigit()):
            j = i
            while j < n and src[j].isdigit():
                j += 1
            if j < n and src[j] == "." and not src.startswith("..", j):
                j += 1
                while j < n and src[j].isdigit():
                    j += 1
            if j < n and src[j] in "eE":
                k = j + 1
                if k < n and src[k] in "+-":
                    k += 1
                if k < n and src[k].isdigit():
                    j = k
                    while j < n and src[j].isdigit():
                        j += 1
            toks.append(Token("num", src[i:j], line, start_col))
            col += j - i; i = j
            continue
        if c.isalpha() or c == "_":
            j = i
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            word = src[i:j]
            toks.append(Token("kw" if word in KEYWORDS else "id", word, line, start_col))
            col += j - i; i = j
            continue
        for p in PUNCT:
            if src.startswith(p, i):
                toks.append(Token("op", p, line, start_col))
                i += len(p); col += len(p)
                break
        else:
            raise CompileError(f"unexpected character {c!r}", line, start_col, "lexer")
    toks.append(Token("eof", "", line, col))
    return toks
