"""
Implementa o transformador da árvore sintática que converte entre as representações

    lark.Tree -> lox.ast.Node.

A resolução de vários exercícios requer a modificação ou implementação de vários
métodos desta classe.
"""

from typing import Callable
from lark import Transformer, v_args

from . import runtime as op
from .ast import *
from .ast import UnaryOp


def op_handler(op: Callable):
    """
    Fábrica de métodos que lidam com operações binárias na árvore sintática.

    Recebe a função que implementa a operação em tempo de execução.
    """

    def method(self, left, right):
        return BinOp(left, right, op)

    return method


@v_args(inline=True)
class LoxTransformer(Transformer):
    # Programa
    def program(self, *stmts):
        return Program(list(stmts))

    # Operações matemáticas básicas
    mul = op_handler(op.mul)
    div = op_handler(op.truediv)
    sub = op_handler(op.sub)
    add = op_handler(op.add)

    # Comparações
    gt = op_handler(op.gt)
    lt = op_handler(op.lt)
    ge = op_handler(op.ge)
    le = op_handler(op.le)
    eq = op_handler(op.eq)
    ne = op_handler(op.ne)

    # Outras expressões
    def call(self, func, params):
        return Call(func, params)

    def params(self, *args):
        params = list(args)
        return params

    # Comandos
    def print_cmd(self, expr):
        return Print(expr)

    def VAR(self, token):
        name = str(token)
        return Var(name)

    def NUMBER(self, token):
        num = float(token)
        return Literal(num)
    
    def STRING(self, token):
        text = str(token)[1:-1]
        return Literal(text)
    
    def NIL(self, _):
        return Literal(None)

    def BOOL(self, token):
        return Literal(token == "true")

    def getattr(self, obj, attr):
        return Getattr(obj, attr.name)

    def setattr_stmt(self, obj, attr, value):
        return Setattr(obj, attr.name, value)

    def not_(self, value):
        return UnaryOp('!', value)

    def neg_(self, value):
        return UnaryOp('-', value)

    # Adicionando suporte a operações unárias na árvore sintática
    def test(self, op, value):
        if op == '!':
            return UnaryOp('!', value)
        elif op == '-':
            return UnaryOp('-', value)
        return value

    def and_(self, *args):
        expr = args[0]
        for next_expr in args[1:]:
            expr = And(expr, next_expr)
        return expr

    def or_(self, *args):
        expr = args[0]
        for next_expr in args[1:]:
            expr = Or(expr, next_expr)
        return expr

    def assign(self, var, value):
        return Assign(var, value)

    def var_decl(self, name, value=None):
        if value is None:
            value = Literal(None)
        return VarDef(name.name, value)

    def declaration(self, node):
        return node