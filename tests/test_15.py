import builtins
from contextlib import contextmanager

import pytest

from lox import *
from lox import testing
from lox.ast import *
from lox.ctx import Ctx


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
        return (ctx, {})

    def test_variável_definida_em_escopo_interno(self):
        src = "var x = 1; { var x = 2; print x; } print x;"
        self.verify(src, {}, expect_stdout="2\n1\n")

    def test_podemos_atribuir_a_variável_de_contexto_pai(self):
        src = "var x; { x = 1; } print x;"
        self.verify(src, {}, expect_stdout="1\n")

    def test_não_podemos_atribuir_a_variáveis_não_declaradas_num_escopo_local(self):
        src = "{ x = 1; print x; }"
        with pytest.raises((NameError, KeyError, SemanticError)):
            self.verify(src, {}, expect_none=True)

    def test_podemos_declarar_repetidas_vezes_no_escopo_global(self):
        src = "var x = 1; var x = 1; print x;"
        self.verify(src, {}, expect_stdout="1\n")

    def test_não_podemos_declarar_repetidas_vezes_num_escopo_local(self):
        src = "{ var x = 1; var x = 1; }"
        with pytest.raises((NameError, KeyError, SemanticError)):
            self.verify(src, {}, expect_none=True)


def test_implementa_o_método_pop():
    msg = """O método Ctx.pop() remove o escopo mais interno e retorna uma 
dupla (escopo, contexto_pai)"""

    with show_error(msg):
        parent = Ctx({"a": 1}, None)
        ctx = Ctx(d := {"b": 2}, parent)

        scope, other = ctx.pop()
        assert scope is d, "ctx.pop() deve retornar o escopo mais interno"
        assert other is parent, "ctx.pop() deve retornar o contexto pai"


def test_implementa_o_método_push():
    msg = """O método Ctx.push(env) empilha um novo escopo no contexto atual,
retornando o contexto atualizado. ctx.push(env) <==> Ctx(env, ctx)"""

    with show_error(msg):
        parent = Ctx({"a": 1}, None)
        ctx = parent.push({"a": 2, "b": 3})

        assert ctx["a"] == 2
        assert ctx["b"] == 3

        with pytest.raises(KeyError):
            parent["b"]
            print("ctx.push deve manter o escopo do pai inalterado")


class TestExamplesBlock(testing.ExampleTester):
    module = "block"
    examples = [
        "empty",
        "scope",
    ]
    fuzzy_output = True


class TestExamplesVar(testing.ExampleTester):
    module = "variable"
    examples = [
        # "collide_with_parameter",
        # "duplicate_local",
        # "duplicate_parameter",
        # "early_bound",
        "in_middle_of_block",
        "in_nested_block",
        # "local_from_method",
        "redeclare_global",
        "redefine_global",
        "scope_reuse_in_different_blocks",
        "shadow_and_local",
        "shadow_global",
        "shadow_local",
        "undefined_global",
        "undefined_local",
        "uninitialized",
        "unreached_undefined",
        # "use_false_as_var",
        "use_global_in_initializer",
        # "use_local_in_initializer",
        # "use_nil_as_var",
        # "use_this_as_var",
    ]
    fuzzy_output = True


@contextmanager
def show_error(msg):
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
