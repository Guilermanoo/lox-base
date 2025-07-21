import builtins
from dataclasses import dataclass
from operator import add, eq, ge, gt, le, lt, mul, ne, neg, not_, sub, truediv
from types import FunctionType, BuiltinFunctionType
from typing import TYPE_CHECKING, Optional, TYPE_CHECKING

from .ctx import Ctx

if TYPE_CHECKING:
    from .runtime import LoxFunction, LoxError
    from .ast import Stmt, Value

__all__ = [
    "add",
    "eq",
    "ge",
    "gt",
    "le",
    "lt",
    "mul",
    "ne",
    "neg",
    "not_",
    "print",
    "show",
    "sub",
    "truthy",
    "truediv",
]


class LoxError(Exception):
    """
    Exceção para erros de execução Lox.
    """


class LoxReturn(Exception):
    """
    Exceção para retornar de uma função Lox.
    """
    def __init__(self, value):
        self.value = value
        super().__init__()


@dataclass
class LoxFunction:
    """
    Classe base para todas as funções Lox.
    """

    name: str
    args: list[str]
    body: list["Stmt"]
    ctx: Ctx
    is_bound_method: bool = False

    def __call__(self, *args):
        env = dict(zip(self.args, args, strict=True))
        call_ctx = self.ctx.push(env)
        # Only treat as init constructor if it's a bound method named 'init'
        is_init = self.name == 'init' and self.is_bound_method and 'this' in call_ctx
        try:
            for stmt in self.body:
                stmt.eval(call_ctx)
        except LoxReturn as e:
            # Se for init, sempre retorna this
            if is_init:
                return call_ctx['this']
            return e.value
        # Se for init, sempre retorna this
        if is_init:
            return call_ctx['this']
        return None

    def bind(self, obj):
        # Retorna uma nova função LoxFunction com o contexto extendido com {'this': obj}
        return LoxFunction(
            name=self.name,
            args=self.args,
            body=self.body,
            ctx=self.ctx.push({'this': obj}),
            is_bound_method=True
        )


@dataclass
class LoxClass:
    """
    Representa uma classe Lox em tempo de execução.
    """

    name: str
    methods: dict[str, 'LoxFunction']
    base: Optional['LoxClass'] = None

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def __call__(self, *args):
        instance = LoxInstance(self)
        # Procura init na hierarquia de métodos
        init_method = self.get_method('init')
        if init_method:
            # Chama init vinculado à instância
            bound_init = init_method.bind(instance)
            bound_init(*args)
        else:
            # Se não há init mas foram passados argumentos, é erro
            if args:
                raise RuntimeError(f"Expected 0 arguments but got {len(args)}.")
        return instance

    def get_method(self, name: str) -> 'LoxFunction | None':
        name = str(name).strip()
        if name in self.methods:
            return self.methods[name]
        if self.base is not None:
            return self.base.get_method(name)
        return None


class LoxInstance:
    """
    Representa uma instância de uma classe Lox.
    """
    def __init__(self, klass: 'LoxClass'):
        super().__setattr__('klass', klass)
        super().__setattr__('fields', {})

    def get(self, name):
        """
        Obtém um atributo ou método da instância.
        """
        fields = super().__getattribute__('fields')
        if name in fields:
            return fields[name]
        klass = super().__getattribute__('klass')
        method = klass.get_method(name)
        if method is not None:
            return method.bind(self)
        raise AttributeError(f"'{klass.name}' instance has no attribute '{name}'")

    def set(self, name, value):
        """
        Define um atributo da instância.
        """
        fields = super().__getattribute__('fields')
        fields[name] = value

    def __getattr__(self, name):
        fields = super().__getattribute__('fields')
        if name in fields:
            return fields[name]
        klass = super().__getattribute__('klass')
        method = klass.get_method(name)
        if method is not None:
            return method.bind(self)
        raise AttributeError(f"'{klass.name}' instance has no attribute '{name}'")

    def __setattr__(self, name, value):
        if name in ("klass", "fields"):
            super().__setattr__(name, value)
        else:
            fields = super().__getattribute__('fields')
            fields[name] = value

    def init(self, *args):
        # Permite chamada manual de init: u.init(...)
        init_method = self.klass.get_method('init')
        if not init_method:
            return self
        bound_init = init_method.bind(self)
        bound_init(*args)
        return self

    def __str__(self):
        return f"{self.klass.name} instance"

    def __repr__(self):
        return str(self)


nan = float("nan")
inf = float("inf")


def print(value: "Value"):
    """
    Imprime um valor lox.
    """
    builtins.print(show(value))


def show(value: "Value") -> str:
    """
    Converte valor lox para string no formato Lox.
    """
    # None (nil)
    if value is None:
        return "nil"
    # Booleanos
    if value is True:
        return "true"
    if value is False:
        return "false"
    # Números inteiros
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return str(value)
    # Função Lox
    if hasattr(value, "__class__") and value.__class__.__name__ == "LoxFunction":
        return f"<fn {getattr(value, 'name', '?')}>"
    # Função nativa Python
    if isinstance(value, (FunctionType, BuiltinFunctionType)):
        return "<native fn>"
    # Instância Lox
    if hasattr(value, "__class__") and value.__class__.__name__ == "LoxInstance":
        # Use o nome da classe Lox, não o nome Python
        return f"{value.klass.name} instance"
    # Classe Lox
    if hasattr(value, '__class__') and value.__class__.__name__ == "LoxClass":
        return value.name
    # String
    if isinstance(value, str):
        return value
    # Fallback
    return str(value)


def show_repr(value: "Value") -> str:
    """
    Mostra um valor lox, mas coloca aspas em strings.
    """
    if isinstance(value, str):
        return f'"{value}"'
    return show(value)


def truthy(value: "Value") -> bool:
    """
    Converte valor lox para booleano segundo a semântica do lox.
    """
    if value is None or value is False:
        return False
    return True


def lox_eq(a, b):
    # Igualdade estrita: tipos diferentes nunca são iguais
    if type(a) != type(b):
        return False
    return a == b


def lox_ne(a, b):
    return not lox_eq(a, b)


def lox_add(a, b):
    # Soma de números
    if isinstance(a, float) and isinstance(b, float):
        return a + b
    # Concatenação de strings
    if isinstance(a, str) and isinstance(b, str):
        return a + b
    raise LoxError("Operands must be two numbers or two strings.")


def lox_sub(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return a - b
    raise LoxError("Operands must be numbers.")


def lox_mul(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return a * b
    raise LoxError("Operands must be numbers.")


def lox_div(a, b):
    if isinstance(a, float) and isinstance(b, float):
        if b == 0:
            raise LoxError("Division by zero.")
        return a / b
    raise LoxError("Operands must be numbers.")


def lox_neg(a):
    if isinstance(a, float):
        return -a
    raise LoxError("Operand must be a number.")


def lox_gt(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return a > b
    raise LoxError("Operands must be numbers.")


def lox_ge(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return a >= b
    raise LoxError("Operands must be numbers.")


def lox_lt(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return a < b
    raise LoxError("Operands must be numbers.")


def lox_le(a, b):
    if isinstance(a, float) and isinstance(b, float):
        return a <= b
    raise LoxError("Operands must be numbers.")


def lox_not(a):
    return not truthy(a)
