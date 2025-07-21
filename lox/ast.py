from abc import ABC
from dataclasses import dataclass
from typing import Callable

from .ctx import Ctx
from .runtime import (
    LoxFunction, LoxReturn, print as lox_print,
    lox_add, lox_sub, lox_mul, lox_div, lox_eq, lox_ne,
    lox_gt, lox_ge, lox_lt, lox_le, lox_not, lox_neg, truthy,
    LoxClass, LoxInstance  # Adiciona LoxInstance para exportação
)

# Declaramos nossa classe base num módulo separado para esconder um pouco de
# Python relativamente avançado de quem não se interessar pelo assunto.
#
# A classe Node implementa um método `pretty` que imprime as árvores de forma
# legível. Também possui funcionalidades para navegar na árvore usando cursores
# e métodos de visitação.
from .node import Node, Cursor
from .errors import SemanticError


#
# TIPOS BÁSICOS
#

# Tipos de valores que podem aparecer durante a execução do programa
Value = bool | str | float | None


class Expr(Node, ABC):
    """
    Classe base para expressões.

    Expressões são nós que podem ser avaliados para produzir um valor.
    Também podem ser atribuídos a variáveis, passados como argumentos para
    funções, etc.
    """


class Stmt(Node, ABC):
    """
    Classe base para comandos.

    Comandos são associdos a construtos sintáticos que alteram o fluxo de
    execução do código ou declaram elementos como classes, funções, etc.
    """


@dataclass
class Program(Node):
    """
    Representa um programa.

    Um programa é uma lista de comandos.
    """

    stmts: list[Stmt]

    def eval(self, ctx: Ctx):
        for stmt in self.stmts:
            stmt.eval(ctx)


#
# EXPRESSÕES
#
@dataclass
class BinOp(Expr):
    """
    Uma operação infixa com dois operandos.

    Ex.: x + y, 2 * x, 3.14 > 3 and 3.14 < 4
    """

    left: Expr
    right: Expr
    op: str

    def eval(self, ctx: Ctx):
        left_value = self.left.eval(ctx)
        right_value = self.right.eval(ctx)
        if self.op == '+':
            return lox_add(left_value, right_value)
        elif self.op == '-':
            return lox_sub(left_value, right_value)
        elif self.op == '*':
            return lox_mul(left_value, right_value)
        elif self.op == '/':
            return lox_div(left_value, right_value)
        elif self.op == '==':
            return lox_eq(left_value, right_value)
        elif self.op == '!=':
            return lox_ne(left_value, right_value)
        elif self.op == '>':
            return lox_gt(left_value, right_value)
        elif self.op == '>=':
            return lox_ge(left_value, right_value)
        elif self.op == '<':
            return lox_lt(left_value, right_value)
        elif self.op == '<=':
            return lox_le(left_value, right_value)
        else:
            raise RuntimeError(f"Operador desconhecido: {self.op}")


@dataclass(frozen=True)
class Var(Expr):
    """
    Uma variável no código

    Ex.: x, y, z
    """

    name: str

    def eval(self, ctx: Ctx):
        return ctx[self.name]


@dataclass
class Literal(Expr):
    """
    Representa valores literais no código, ex.: strings, booleanos,
    números, etc.

    Ex.: "Hello, world!", 42, 3.14, true, nil
    """

    value: Value

    def eval(self, ctx: Ctx):
        return self.value


def is_lox_false(value):
    return not truthy(value)


@dataclass
class And(Expr):
    """
    Uma operação infixa com dois operandos.

    Ex.: x and y
    """

    left: Expr
    right: Expr

    def eval(self, ctx: Ctx):
        left_value = self.left.eval(ctx)
        if not truthy(left_value):
            return left_value
        return self.right.eval(ctx)


@dataclass
class Or(Expr):
    """
    Uma operação infixa com dois operandos.
    Ex.: x or y
    """

    left: Expr
    right: Expr

    def eval(self, ctx: Ctx):
        left_value = self.left.eval(ctx)
        if truthy(left_value):
            return left_value
        return self.right.eval(ctx)


@dataclass
class UnaryOp(Expr):
    """
    Uma operação prefixa com um operando.

    Ex.: -x, !x
    """

    op: str   # Ex: '-' ou '!'
    value: Expr

    def eval(self, ctx: Ctx):
        v = self.value.eval(ctx)
        if self.op == '-':
            return lox_neg(v)
        elif self.op == '!':
            return lox_not(v)
        else:
            raise RuntimeError(f"Operador unário desconhecido: {self.op}")


@dataclass(frozen=True)
class Call(Expr):
    """
    Uma chamada de função.

    Ex.: fat(42)
    """
    func: Expr
    args: list

    def eval(self, ctx: Ctx):
        func_value = self.func.eval(ctx)
        arg_values = [arg.eval(ctx) for arg in self.args]
        return func_value(*arg_values)


@dataclass
class Getattr(Expr):
    """
    Representa acesso a atributo/método de um objeto.

    Ex.: obj.attr
    """
    obj: Expr
    attr: str

    def eval(self, ctx: Ctx):
        obj_value = self.obj.eval(ctx)
        # Para instâncias de LoxInstance, usa o método get
        if hasattr(obj_value, '__class__') and 'LoxInstance' in str(obj_value.__class__):
            return obj_value.get(self.attr)
        # Para outros objetos Python, usa getattr
        else:
            return getattr(obj_value, self.attr)


@dataclass
class Setattr(Expr):
    """
    Representa atribuição a atributo de um objeto.

    Ex.: obj.attr = value;
    """
    obj: Expr
    attr: str
    value: Expr

    def eval(self, ctx: Ctx):
        obj_value = self.obj.eval(ctx)
        value_result = self.value.eval(ctx)
        # Para instâncias de LoxInstance, usa o método set
        if hasattr(obj_value, '__class__') and 'LoxInstance' in str(obj_value.__class__):
            obj_value.set(self.attr, value_result)
        # Para outros objetos Python, usa setattr
        else:
            setattr(obj_value, self.attr, value_result)
        return value_result


@dataclass
class This(Expr):
    """
    Acesso ao `this`.

    Ex.: this
    """
    
    def children(self):
        """This node has no children"""
        return []
    
    def visit(self, visitors):
        """Custom visit method since This has no fields"""
        # Visit children (none for This)
        for child in self.children():
            child.visit(visitors)
        # Apply visitor to self
        for typ, visitor in visitors.items():
            if isinstance(self, typ):
                visitor(self)
    def eval(self, ctx: Ctx):
        return ctx["this"]

    def validate_self(self, cursor: Cursor):
        # Deve ser descendente de algum nó do tipo Class
        for parent_cursor in cursor.parents():
            if isinstance(parent_cursor.node, Class):
                return
        raise SemanticError("'this' só pode ser usado dentro de métodos de uma classe.")

@dataclass
class Super(Expr):
    """
    Acesso a method ou atributo da superclasse.

    Ex.: super.x
    """
    attr: str

    def eval(self, ctx: Ctx):
        method_name = self.attr
        superclass = ctx["super"]
        this = ctx["this"]
        method = superclass.get_method(method_name)
        if method is None:
            raise RuntimeError(f"Superclasse não tem método '{method_name}'")
        return method.bind(this)

    def validate_self(self, cursor: Cursor):
        # Deve ser descendente de algum nó do tipo Class que tenha superclasse
        for parent_cursor in cursor.parents():
            if isinstance(parent_cursor.node, Class):
                klass = parent_cursor.node
                superclass_field = klass.superclass
                # Desaninha listas/árvores
                while isinstance(superclass_field, list):
                    if not superclass_field:
                        superclass_field = None
                        break
                    superclass_field = superclass_field[0]
                try:
                    from lark.tree import Tree
                    if isinstance(superclass_field, Tree):
                        superclass_field = None
                except ImportError:
                    pass
                if superclass_field is not None:
                    return
                else:
                    raise SemanticError("'super' só pode ser usado em classes que herdam de outra classe.")
        raise SemanticError("'super' só pode ser usado dentro de métodos de uma classe.")

@dataclass
class Return(Stmt):
    """
    Representa uma instrução de retorno.

    Ex.: return x;
    """
    expr: Expr

    def eval(self, ctx: Ctx):
        value = self.expr.eval(ctx)
        raise LoxReturn(value)

    def validate_self(self, cursor: Cursor):
        # Deve ser descendente de algum nó do tipo Function
        for parent_cursor in cursor.parents():
            if isinstance(parent_cursor.node, Function):
                return
        raise SemanticError("'return' só pode ser usado dentro de funções.")

@dataclass
class VarDef(Stmt):
    """
    Representa uma declaração de variável.

    Ex.: var x = 42;
    """

    name: str
    value: Expr

    def eval(self, ctx: Ctx):
        v = self.value.eval(ctx)
        ctx.var_def(self.name, v)  # Cria a variável no escopo atual
        return v

    def validate_self(self, cursor: Cursor):
        # Duplicidade no mesmo bloco
        parent = cursor.parent().node if not cursor.is_root() else None
        if isinstance(parent, Block):
            names = [stmt.name for stmt in parent.stmts if isinstance(stmt, VarDef)]
            if names.count(self.name) > 1:
                raise SemanticError("variável duplicada no bloco", token=self.name)

@dataclass
class If(Stmt):
    """
    Representa uma instrução condicional.

    Ex.: if (x > 0) { ... } else { ... }
    """

    cond: Expr
    then_branch: Stmt
    else_branch: Stmt | None = None

    def eval(self, ctx: Ctx):
        if truthy(self.cond.eval(ctx)):
            return self.then_branch.eval(ctx)
        elif self.else_branch is not None:
            return self.else_branch.eval(ctx)
        return None


@dataclass
class While(Stmt):
    cond: Expr
    body: Stmt

    def eval(self, ctx: Ctx):
        while self.cond.eval(ctx):
            self.body.eval(ctx)


@dataclass
class Block(Stmt):
    """
    Representa bloco de comandos.

    Ex.: { var x = 42; print x;  }
    """
    stmts: list[Stmt]

    def eval(self, ctx: Ctx):
        ctx = ctx.push({})
        for stmt in self.stmts:
            stmt.eval(ctx)
        return None

    def validate_self(self, cursor: Cursor):
        # Duplicidade de variáveis no bloco
        names = [stmt.name for stmt in self.stmts if isinstance(stmt, VarDef)]
        if len(names) != len(set(names)):
            for name in names:
                if names.count(name) > 1:
                    raise SemanticError("variável duplicada no bloco", token=name)

@dataclass
class Function(Stmt):
    """
    Representa uma função.

    Ex.: fun f(x, y) { ... }
    """

    name: str
    params: list
    body: Stmt

    def eval(self, ctx: Ctx, register=True):
        # Corrige casos em que params é um Block vazio
        if isinstance(self.params, Block):
            param_names = []
        else:
            param_names = [p.name if hasattr(p, 'name') else p for p in self.params]
        # Garante que body é uma lista de comandos
        if self.body is None:
            body_stmts = []
        elif isinstance(self.body, Block):
            body_stmts = self.body.stmts
        elif isinstance(self.body, list):
            body_stmts = self.body
        else:
            body_stmts = [self.body]
        fn = LoxFunction(
            name=self.name,
            args=param_names,
            body=body_stmts,
            ctx=ctx
        )
        if register:
            ctx.var_def(self.name, fn)
        return fn

    def validate_self(self, cursor: Cursor):
        # Parâmetros duplicados
        param_names = [p.name if hasattr(p, 'name') else p for p in self.params] if not isinstance(self.params, Block) else []
        if len(param_names) != len(set(param_names)):
            for name in param_names:
                if param_names.count(name) > 1:
                    raise SemanticError("parâmetro duplicado", token=name)
        # Variável no corpo com mesmo nome de parâmetro
        body_stmts = self.body.stmts if isinstance(self.body, Block) else ([self.body] if self.body else [])
        var_names = [stmt.name for stmt in body_stmts if isinstance(stmt, VarDef)]
        for name in var_names:
            if name in param_names:
                raise SemanticError("variável colide com parâmetro", token=name)

@dataclass
class Class(Stmt):
    """
    Representa uma classe.

    Ex.: class B < A { ... }
    """
    name: str
    superclass: Expr | None
    methods: list[Function]

    def eval(self, ctx: Ctx):
        # Corrige superclass: pode vir como lista ou Tree
        superclass_field = self.superclass
        while isinstance(superclass_field, list):
            if not superclass_field:
                superclass_field = None
                break
            superclass_field = superclass_field[0]
        try:
            from lark.tree import Tree
            if isinstance(superclass_field, Tree):
                superclass_field = None
        except ImportError:
            pass
        superclass = None
        if superclass_field is not None:
            superclass = superclass_field.eval(ctx)
            if not isinstance(superclass, LoxClass):
                raise SemanticError(f"Superclasse '{superclass}' não é uma classe.")
        # Cria um novo contexto para o corpo da classe
        class_ctx = ctx.push({})
        class_ctx.var_def(self.name, None)
        # Não pop class_ctx; mantém vivo enquanto a classe existir
        # Garante que methods é uma lista de Function (desaninha listas)
        def flatten_methods(methods):
            result = []
            for m in methods:
                if isinstance(m, list):
                    result.extend(flatten_methods(m))
                elif hasattr(m, 'name') and hasattr(m, 'params') and hasattr(m, 'body'):
                    result.append(m)
            return result
        methods = flatten_methods(self.methods if self.methods is not None else [])
        # Herdar métodos da superclasse
        if superclass is not None:
            methods_dict = dict(superclass.methods)
        else:
            methods_dict = {}
        # Se houver superclasse, métodos recebem contexto com 'super'
        if superclass is not None:
            method_ctx = class_ctx.push({"super": superclass})
        else:
            method_ctx = class_ctx
        for method in methods:
            if hasattr(method, 'name') and method.name:
                lox_fn = method.eval(method_ctx, register=False)
                methods_dict[str(method.name)] = lox_fn
        klass = LoxClass(self.name, methods_dict, superclass)
        klass.ctx = class_ctx  # Mantém o contexto vivo
        class_ctx.assign(self.name, klass)
        ctx.var_def(self.name, klass)
        return klass

@dataclass
class Assign(Expr):
    """
    Representa uma atribuição de variável existente.

    Ex.: x = 42;
    """
    name: str
    value: Expr

    def eval(self, ctx: Ctx):
        v = self.value.eval(ctx)
        ctx.assign(self.name, v)
        return v

    def validate_self(self, cursor: Cursor):
        # Em Lox, atribuições podem ser feitas a variáveis existentes no contexto
        # A validação de existência será feita em runtime
        pass

@dataclass
class Print(Stmt):
    """
    Representa um comando de impressão.

    Ex.: print x;
    """
    expr: Expr

    def eval(self, ctx: Ctx):
        value = self.expr.eval(ctx)
        lox_print(value)
        return value

    def validate_self(self, cursor: Cursor):
        pass
