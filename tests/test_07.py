from lox import *
from lox import testing
from lox.ast import *


class TestAttribution(testing.ExerciseTester):
    is_expr = True
    src1 = "x = 42"
    src2 = "y = x and y"
    src3 = "x = y = z = 42"
    tks1 = "x 42"
    tks2 = "y x"
    tks3 = "x y z 42"
    ast_class = Assign

    def eval_env(self, n):
        ctx = {"x": True, "y": False, "z": True}
        result = (42, False, 42)[n - 1]
        return (ctx, result)

    def test_atribuição_modifica_o_contexto(self):
        ctx = Ctx.from_dict({"x": 0})
        self.ast(1).eval(ctx)
        assert ctx["x"] == 42, "Atribuição não modificou o contexto corretamente"

    def test_atribuição_avalia_como_o_lado_direito(self):
        ctx = Ctx.from_dict({"x": 0})
        assert self.ast(1).eval(ctx) == 42
