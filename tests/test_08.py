from lox import *
from lox import testing
from lox.ast import *


class TestVarDef(testing.ExerciseTester):
    is_expr = False
    src1 = "var x = 42;"
    src2 = "var y;"
    src3 = "var x = 40 + 2;"
    tks1 = "x 42"
    tks2 = "y"
    tks3 = "x 40 2"
    ast_class = VarDef

    def eval_env(self, n):
        result = [{"x": 42}, {"y": None}, {"x": 42}][n - 1]
        return ({}, result)

    def test_declaração_de_variáveis_modifica_o_contexto(self):
        ctx = Ctx.from_dict({})
        ast = self.ast(1)
        ast.eval(ctx)
        assert ctx["x"] == 42, "Atribuição não modificou o contexto corretamente"

    def test_declaração_de_variáveis_não_inicializadas_inicializa_o_valor_com_nil(self):
        ctx = Ctx.from_dict({})
        ast = self.ast(2)
        ast.eval(ctx)
        assert ctx["y"] is None


class TestExamples(testing.ExampleTester):
    module = "variable"
    examples = {
        # "redeclare_global",
        "redefine_global",
        "undefined_global",
        "uninitialized",
        "use_global_in_initializer",
    }
    fuzzy_output = True
