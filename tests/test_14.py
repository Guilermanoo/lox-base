import builtins
import random
from contextlib import contextmanager

import pytest

from lox import *
from lox import testing
from lox.ast import *
from lox.ctx import Ctx

msg_init1 = "Ctx({}, parent) deve criar um contexto com o escopo vazio e um dado pai."
msg_init2 = "Deve ser possível criar um contexto sem pai com Ctx({}, None)"
msg_init3 = """Se o pai for omitido, como em Ctx({}), construa o contexto com os builtins.

Dica! Veja a documentação da função dataclasses.field. 
A opção field(default_factory=...) pode ser útil para inicializar o escopo com 
os builtins como pai.

https://docs.python.org/3/library/dataclasses.html#dataclasses.field
"""
msg_init4 = """Finalmente, Ctx() deve criar um contexto usando um 
dicionário vazio.

field(default_factory=...) pode ser útil novamente.
"""

msg_vardef = """
O método var_def(name, value) deve adicionar uma variável ao 
escopo mais alto do contexto.
"""

msg_setitem1 = """O método __setitem__ deve sobrescrever uma variável
no escopo em que ela foi definida.
"""

msg_setitem2 = """Caso a variável não exista, deve levantar um KeyError."""

msg_iter_scopes = """Vamos implementar um método Ctx.iter_scopes() para iterar
sobre a lista de escopos definidas em Ctx. Para esta tarefa, vale 
a pena conhecer sobre geradore e a palavra reservada yield.

https://realpython.com/introduction-to-python-generators/

Isso facilitará muito a implementação do método iter_scopes.
"""


class TestScope(testing.ExerciseTester):
    is_expr = False
    src1 = "x = 42;"
    src2 = "var x = 1;"
    src3 = "var x = 2; x = 3;"
    test_ast = False
    test_cst = False
    fuzzy_output = True

    def eval_env(self, n):
        ctx = {"x": 0}
        return (ctx, {"x": [42, 1, 3][n - 1]})


def test_métodos_de_inicialização_de_ctx():
    parent = object.__new__(Ctx)

    with show(msg_init1):
        ctx = Ctx({"x": 1}, parent)
        assert ctx["x"] == 1

    with show(msg_init2):
        ctx = Ctx({"x": 1}, None)
        assert ctx["x"] == 1

    with show(msg_init3):
        ctx = Ctx({"x": 1})
        assert ctx["x"] == 1
        assert isinstance(ctx["clock"](), float)

    with show(msg_init4):
        ctx = Ctx()
        assert isinstance(ctx["clock"](), float)


def test_implementa_o_método_var_def():
    with show(msg_vardef):
        ctx = Ctx()
        ctx.var_def("n", n := random.randint(1, 10))
        assert ctx["n"] == n

        ctx2 = Ctx({}, ctx)
        ctx2.var_def("n", 42)
        assert ctx2["n"] == 42

        msg = "ctx não deve ser modificado ao definir uma variável em ctx2"
        assert ctx["n"] == n, msg


def test_implementa_o_método_setitem_corretamente():
    with show(msg_setitem1):
        ctx = Ctx({"a": -1})
        ctx["a"] = 42
        assert ctx["a"] == 42

    with show(msg_setitem2):
        with pytest.raises(KeyError):
            ctx["b"] = 42

    with show(msg_setitem1):
        parent = Ctx({"a": -1, "b": -1})
        ctx = Ctx({"a": 1}, parent)

        ctx["a"] = 42
        assert ctx["a"] == 42, "ctx deve sobrescrever a variável 'a' no escopo atual"
        assert parent["a"] == -1, "ctx não deve modificar o pai ao sobrescrever 'a'"

        ctx["b"] = 42
        assert ctx["b"] == 42, "ctx deve encontrar a variável 'b' ao escopo atual"
        assert parent["b"] == 42, "como ctx não define b, deve sobrescrever no pai"

        parent["b"] = 43
        assert ctx["b"] == 43, "ctx deve usar b do pai, já que não foi declarada em ctx"

        ctx.var_def("b", 10)
        assert ctx["b"] == 10, "agora ctx declarou b, não dependendo mais do pai"
        assert parent["b"] == 43, "pai não deve ser modificado ao declarar b em ctx"


def test_implementa_o_método_iter_scopes():
    with show(msg_iter_scopes):
        parent = Ctx({"a": 1}, None)
        ctx = Ctx({"b": 2}, parent)
        assert hasattr(ctx, "iter_scopes")

    msg = "ctx.iter_scopes() começa iterando pelo escopo atual passando por todos os pais."
    with show(msg):
        assert list(ctx.iter_scopes()) == [{"b": 2}, {"a": 1}]

    msg = "ctx.iter_scopes() deve aceitar qualquer nível de aninhamento."
    with show(msg):
        top = Ctx({"c": 3}, ctx)
        assert list(top.iter_scopes()) == [{"c": 3}, {"b": 2}, {"a": 1}]

    msg = "A função deve aceitar o argumento opcional reverse, com o valor padrão False"
    with show(msg):
        assert list(top.iter_scopes(reverse=False)) == [{"c": 3}, {"b": 2}, {"a": 1}]

    msg = "Se chamarmos ctx.iter_scopes(reverse=True) a ordem de iteração deve ser invertida."
    with show(msg):
        assert list(top.iter_scopes(reverse=True)) == [{"a": 1}, {"b": 2}, {"c": 3}]


def test_implementa_o_método_from_dict():
    msg = """O método Ctx.from_dict deve criar um contexto a partir de um 
dicionário. Ele funciona basicamente como Ctx.from_dict(d) <==> Ctx(d)"""

    with show(msg):
        ctx = Ctx.from_dict({"a": 1})
        assert ctx["a"] == 1
        assert isinstance(ctx["clock"](), float)


def test_implementa_o_método_to_dict():
    msg = "O método Ctx.to_dict() converte o contexto para um dicionário."
    with show(msg):
        parent = Ctx({"a": 1}, None)
        ctx = Ctx({"b": 2}, parent)
        assert ctx.to_dict() == {"a": 1, "b": 2}

    msg = "Damos prioridade para as variáveis nos escopos mais internos."
    with show(msg):
        top = Ctx({"b": 3}, ctx)
        assert top.to_dict() == {"a": 1, "b": 3}


def test_implementa_o_método_contains():
    msg = """O método Ctx.__contains__(name) deve verificar se uma variável
existe no contexto ou em um pai, retornando True ou False. 

ctx.__contains__(name) <==> name in ctx."""

    with show(msg):
        parent = Ctx({"a": 1}, None)
        assert "a" in parent
        assert "b" not in parent

        ctx = Ctx({"b": 2}, parent)
        assert "a" in ctx
        assert "b" in ctx


def test_implementa_o_método_is_global():
    msg = """O método Ctx.is_global() deve retornar True se o contexto 
representar o escopo global, o primeiro escopo acima de builtins.

ctx.is_global() é falso para builtins (que não possui pai)
ctx.is_global() é verdadeiro para o contexto acima de builtins.
ctx.is_global() é falso para qualquer outro contexto.
"""

    with show(msg):
        builtins = Ctx({}, None)
        assert not builtins.is_global(), "builtins não é o escopo global"

        globals = Ctx({"a": 2}, builtins)
        assert globals.is_global(), "o escopo global é o primeiro acima de builtins"

        inner = Ctx({"b": 2}, globals)
        assert not inner.is_global(), "um escopo interno ao global não é global"


def test_implementa_o_método_pretty():
    msg = """O método Ctx.pretty() deve retornar uma representação do contexto
como string, com cada escopo em uma linha, começando pelo mais interno.
"""

    with show(msg):
        builtins = Ctx({"a": 1}, None)
        globals = Ctx({"b": 2}, builtins)
        inner = Ctx({"a": 4, "c": 3}, globals)
        ctx = Ctx({"x": 1}, Ctx({}, globals))

        assert hasattr(inner, "pretty"), "ctx deve ter o método pretty()"

    print("Exemplo")
    print("ctx = Ctx({'x': 1}, Ctx({}, Ctx({'b': 2}, Ctx({'a': 1}, None))))")
    print("ctx.pretty():")
    print(" 3: x = 1\n 2: <empty>\n 1: b = 2\n 0: a = 1")

    assert builtins.pretty() == " 0: a = 1"
    assert globals.pretty() == " 1: b = 2\n 0: a = 1"
    assert inner.pretty() == " 2: a = 4; c = 3\n 1: b = 2\n 0: a = 1"
    assert ctx.pretty() == " 3: x = 1\n 2: <empty>\n 1: b = 2\n 0: a = 1"


@contextmanager
def show(msg):
    try:
        from rich import print

        prefix = "[red bold]ERRO:[/] "
    except ImportError:
        print = builtins.print
        prefix = "ERRO: "

    try:
        yield
    except Exception:
        print(prefix + msg)
        raise
