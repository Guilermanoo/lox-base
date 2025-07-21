"""
Implementa o transformador da árvore sintática que converte entre as representações

    lark.Tree -> lox.ast.Node.

A resolução de vários exercícios requer a modificação ou implementação de vários
métodos desta classe.
"""

from typing import Callable
from lark import Transformer, v_args

from . import runtime as op
from .ast import BinOp, Block, UnaryOp, Program, Expr, Stmt, Function, Class, Var, Literal, Return, VarDef, If, While, Assign, Block as AstBlock, Print, Call, Getattr, Setattr, And, Or, Super, This


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
        # Flatten nested lists that might come from declarations
        flat_stmts = []
        for stmt in stmts:
            if isinstance(stmt, list):
                flat_stmts.extend(stmt)
            else:
                flat_stmts.append(stmt)
        return Program(flat_stmts)

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

    def primary(self, *args):
        from lark import Token
        # Handle empty primary (which usually means a consumed token)
        if len(args) == 0:
            # This is likely a THIS token that was consumed somewhere
            return This()
        elif len(args) == 1 and isinstance(args[0], Token):
            if args[0].type == 'THIS':
                return This()
            elif args[0].type == 'SUPER':
                # Super alone doesn't make sense, but we'll handle it
                return None
        # For other cases, return the argument as-is
        return args[0] if len(args) == 1 else args

    def getattr(self, obj, attr):
        # Handle super.attr and this.attr cases
        from lark import Tree, Token
        
        # Special case: empty primary tree usually means a token was consumed but not preserved
        if isinstance(obj, Tree) and obj.data == 'primary' and len(obj.children) == 0:
            # This likely means a THIS or SUPER token was processed somewhere else
            # For now, we'll assume it's THIS since SUPER alone doesn't make sense
            return Getattr(This(), attr.name)
        
        if isinstance(obj, Tree) and obj.data == 'primary' and len(obj.children) == 1:
            child = obj.children[0]
            if isinstance(child, Token):
                if child.type == 'SUPER':
                    return Super(attr.name)
                elif child.type == 'THIS':
                    return Getattr(This(), attr.name)
        
        return Getattr(obj, attr.name)

    def setattr_expr(self, obj, attr, value):
        # Handle this.attr = value case
        from lark import Tree
        if isinstance(obj, Tree) and obj.data == 'primary' and len(obj.children) == 0:
            # This likely means a THIS token was processed somewhere else
            return Setattr(This(), attr.name, value)
        
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
        # If only one argument, just return it unwrapped
        if len(args) == 1:
            return args[0]
        expr = args[0]
        for next_expr in args[1:]:
            expr = And(expr, next_expr)
        return expr

    def or_(self, *args):
        # If only one argument, just return it unwrapped
        if len(args) == 1:
            return args[0]
        expr = args[0]
        for next_expr in args[1:]:
            expr = Or(expr, next_expr)
        return expr

    def assign(self, var, value):
        # Se for atribuição a campo (obj.x = ...), gera Setattr
        from .ast import Setattr, Getattr, Var
        if isinstance(var, Getattr):
            return Setattr(var.obj, var.attr, value)
        # Se var é um objeto Var, extrai o nome
        if isinstance(var, Var):
            return Assign(var.name, value)
        return Assign(var, value)

    def var_decl(self, name, value=None):
        if value is None:
            value = Literal(None)
        return VarDef(name.name, value)

    def declaration(self, node):
        # If it's a single-item list, unwrap it
        if isinstance(node, list) and len(node) == 1:
            return node[0]
        return node

    def block(self, *stmts):
        # Flatten nested lists that might come from declarations
        flat_stmts = []
        for stmt in stmts:
            if isinstance(stmt, list):
                flat_stmts.extend(stmt)
            else:
                flat_stmts.append(stmt)
        return Block(flat_stmts)

    def if_stmt(self, cond, then_branch, else_branch=None):
        return If(cond, then_branch, else_branch)

    def while_stmt(self, cond, body):
        return While(cond, body)

    def return_stmt(self, expr=None):
        if expr is None:
            expr = Literal(None)
        elif isinstance(expr, list):
            # Deeply flatten nested lists
            def flatten(lst):
                if not isinstance(lst, list):
                    return lst
                if len(lst) == 1:
                    return flatten(lst[0])
                return lst
            expr = flatten(expr)
        return Return(expr)

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

    def method_decl(self, *args):
        from .ast import Block
        
        # Handle both cases: individual args or a list
        if len(args) == 1 and isinstance(args[0], list):
            args = args[0]
        
        name = args[0] if args else None
        # For methods without parameters, args[1] should be the body directly
        if len(args) == 2:
            params = []
            body = args[1]
        else:
            params = args[1] if len(args) > 1 else None
            body = args[2] if len(args) > 2 else None
        
        # Handle params
        if params is None:
            params = []
        elif isinstance(params, Block):
            params = []
        elif isinstance(params, list):
            params = [p.name if hasattr(p, 'name') else p for p in params]
        else:
            params = [params.name if hasattr(params, 'name') else params]
            
        # Handle body
        if body is None:
            body = Block([])
        elif not isinstance(body, Block):
            body = Block([body])
            
        method_name = name.name if hasattr(name, 'name') else str(name)
        return Function(method_name, params, body)

    def function_decl(self, *args):
        from .ast import Block
        
        # Handle both cases: individual args or a list
        if len(args) == 1 and isinstance(args[0], list):
            args = args[0]
        
        name = args[0] if args else None
        # For functions without parameters, args[1] should be the body directly
        if len(args) == 2:
            params = []
            body = args[1]
        else:
            params = args[1] if len(args) > 1 else None
            body = args[2] if len(args) > 2 else None
        
        # Handle params
        if params is None:
            params = []
        elif isinstance(params, Block):
            params = []
            
        # Handle body
        if body is None:
            body = Block([])
        elif not isinstance(body, Block):
            body = Block([body])
            
        func_name = name.name if hasattr(name, 'name') else name
        return Function(func_name, params, body)

    def class_decl(self, *args):
        from lark.tree import Tree
        from .ast import Function
        
        # Handle both cases: individual args or a list
        if len(args) == 1 and isinstance(args[0], list):
            args = args[0]
        
        name = args[0] if args else None
        
        # Determine if there's a superclass by checking arguments
        if len(args) == 2:
            # class Name { ... } - no superclass
            superclass = None
            body = args[1]
        elif len(args) == 3:
            # class Name < Super { ... } or class Name { ... } with complex parsing
            if isinstance(args[1], list) and all(isinstance(x, Function) for x in args[1]):
                # It's actually class Name { methods... } - no superclass
                superclass = None
                body = args[1]
            else:
                # class Name < Super { ... }
                superclass = args[1]
                body = args[2]
        else:
            superclass = None
            body = None
        
        # Clean up superclass (remove Trees e listas aninhadas)
        if superclass is None or (isinstance(superclass, list) and not superclass):
            superclass = None
        elif isinstance(superclass, list):
            superclass = superclass[0]
        if isinstance(superclass, Tree):
            superclass = None
        # Se superclass for um método (Function), mas não há '<', deve ser None
        if isinstance(superclass, Function):
            superclass = None
            
        # Handle body/methods
        if body is None:
            methods = []
        elif isinstance(body, list):
            methods = [m for m in body if isinstance(m, Function)]
        elif isinstance(body, Function):
            methods = [body]
        else:
            methods = []
            
        class_name = name.name if hasattr(name, 'name') else name
        return Class(
            name=class_name,
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