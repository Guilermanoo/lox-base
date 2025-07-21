"""
Implementa o transformador da árvore sintática que converte entre as representações

    lark.Tree -> lox.ast.Node.

A resolução de vários exercícios requer a modificação ou implementação de vários
métodos desta classe.
"""

from typing import Callable
from lark import Transformer, v_args

from . import runtime as op
from .ast import BinOp, Block, UnaryOp, Program, Expr, Stmt, Function, Class, Var, Literal, Return, VarDef, If, While, Assign, Block as AstBlock, Print, Call, Getattr, Setattr, And, Or


def op_handler_str(op_str: str):
    """
    Fábrica de métodos que lidam com operações binárias na árvore sintática.

    Recebe a função que implementa a operação em tempo de execução.
    """

    def method(self, left, right):
        return BinOp(left, right, op_str)

    return method


@v_args(inline=True)
class LoxTransformer(Transformer):
    # Programa
    def program(self, *stmts):
        return Program(list(stmts))

    # Operações matemáticas básicas
    mul = op_handler_str('*')
    div = op_handler_str('/')
    sub = op_handler_str('-')
    add = op_handler_str('+')

    # Comparações
    gt = op_handler_str('>')
    lt = op_handler_str('<')
    ge = op_handler_str('>=')
    le = op_handler_str('<=')
    eq = op_handler_str('==')
    ne = op_handler_str('!=')

    # Outras expressões
    def call(self, func, params):
        return Call(func, params)

    def params(self, *args):
        params = list(args)
        return params

    def params_list(self, *args):
        # Sempre retorna lista de nomes (strings ou Var)
        return [a.name if hasattr(a, 'name') else a for a in args]

    # Comandos
    def print_cmd(self, expr):
        return Print(expr)

    RESERVED_WORDS = {
        "true", "false", "nil", "return", "class", "fun", "var", "if", "else", "while", "for", "print", "super", "this"
    }

    def VAR(self, token):
        name = str(token)
        if name in self.RESERVED_WORDS:
            from lark.exceptions import UnexpectedToken
            raise UnexpectedToken(token, expected=None)
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
        # Se for atribuição a campo (obj.x = ...), gera Setattr
        from .ast import Setattr, Getattr, Var
        if isinstance(var, Getattr):
            return Setattr(var.obj, var.attr, value)
        return Assign(var, value)

    def var_decl(self, name, value=None):
        if value is None:
            value = Literal(None)
        return VarDef(name.name, value)

    def declaration(self, node):
        return node

    def block(self, *stmts):
        return Block(list(stmts))

    def if_stmt(self, cond, then_branch, else_branch=None):
        return If(cond, then_branch, else_branch)

    def while_stmt(self, cond, body):
        return While(cond, body)

    def for_stmt(self, init, cond, incr, body):
        # Açúcar sintático: for (init; cond; incr) body => { init; while (cond) { body; incr } }
        from .ast import Block, While, Literal
        # cond pode ser None (empty_cond), nesse caso vira Literal(True)
        if cond is None:
            cond = Literal(True)
        # incr pode ser None (empty_incr), nesse caso vira Literal(None)
        if incr is None:
            incr = Literal(None)
        # init pode ser None (empty_init), nesse caso vira Literal(None)
        if init is None:
            init = Literal(None)
        # Corpo do while: body + incr
        if isinstance(body, Block):
            while_body = Block(body.stmts + [incr])
        else:
            while_body = Block([body, incr])
        return Block([init, While(cond, while_body)])

    def for_init(self, node):
        return node
    def for_cond(self, node=None):
        return node
    def cond_expr(self, expr):
        return expr
    def for_incr(self, node=None):
        return node
    def incr_expr(self, expr):
        return expr
    def empty_init(self):
        return None
    def empty_cond(self):
        return None
    def empty_incr(self):
        return None

    def setattr_expr(self, obj, attr, value):
        return Setattr(obj, attr.name, value)

    def method_decl(self, name, params=None, body=None):
        from .ast import Block
        method_name = name.name if hasattr(name, 'name') else str(name)
        if params is None:
            params = []
        elif isinstance(params, Block):
            params = []
        elif isinstance(params, list):
            params = [p.name if hasattr(p, 'name') else p for p in params]
        else:
            params = [params.name if hasattr(params, 'name') else params]
        if body is None:
            body = Block([])
        elif not isinstance(body, Block):
            body = Block([body])
        return Function(method_name, params, body)

    def function_decl(self, name, params=None, body=None):
        from .ast import Block
        if params is None:
            params = []
        elif isinstance(params, Block):
            params = []
        if body is None:
            body = Block([])
        elif not isinstance(body, Block):
            body = Block([body])
        return Function(name.name if hasattr(name, 'name') else name, params, body)

    def class_decl(self, name, superclass=None, body=None):
        # Remove Trees e listas aninhadas de superclass
        from lark.tree import Tree
        # Se não houver superclasse, garantir None
        if superclass is None or (isinstance(superclass, list) and not superclass):
            superclass = None
        elif isinstance(superclass, list):
            superclass = superclass[0]
        if isinstance(superclass, Tree):
            superclass = None
        # Se superclass for um método (Function), mas não há '<', deve ser None
        from .ast import Function
        if isinstance(superclass, Function):
            superclass = None
        # Garante que body é lista de métodos Function
        if body is None:
            methods = []
        elif isinstance(body, list):
            methods = [m for m in body if hasattr(m, 'name') and hasattr(m, 'params') and hasattr(m, 'body')]
        elif hasattr(body, 'name') and hasattr(body, 'params') and hasattr(body, 'body'):
            methods = [body]
        else:
            methods = []
        return Class(
            name=name.name if hasattr(name, 'name') else name,
            superclass=superclass,
            methods=methods
        )

    def class_body(self, *methods):
        # Garante que sempre retorna lista plana de métodos Function
        from .ast import Function
        method_list = []
        for m in methods:
            if isinstance(m, list):
                for x in m:
                    if isinstance(x, Function):
                        method_list.append(x)
            elif isinstance(m, Function):
                method_list.append(m)
            elif hasattr(m, 'name') and hasattr(m, 'params') and hasattr(m, 'body'):
                method_list.append(Function(m.name, m.params, m.body))
        return method_list