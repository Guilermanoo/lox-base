from lox import *
from lox import testing
from lox.ast import *


class TestLogicalOperators(testing.ExerciseTester):
    is_expr = True
    src1 = "x and f()"
    src2 = "y or g()"
    src3 = "z > a and z < b"
    tks1 = "x f"
    tks2 = "y g"
    tks3 = "z a b"
    ast_class1 = And
    ast_class2 = Or
    ast_class3 = And

    def eval_env(self, _):
        ctx = {
            "x": True,
            "f": lambda: True,
            "y": False,
            "g": lambda: True,
            "z": 1.0,
            "a": 0.0,
            "b": 2.0,
        }
        result = True
        return (ctx, result)

    def test_avaliação_de_curto_circuito(self):
        def raises():
            raise RecursionError("Curto-circuito: não deveria avaliar essa função")

        ctx = Ctx.from_dict({"x": False, "f": raises})
        assert self.ast(1).eval(ctx) is False

        ctx = Ctx.from_dict({"y": True, "g": raises})
        assert self.ast(2).eval(ctx) is True


class TestExamples(testing.ExampleTester):
    module = "logical_operator"
    examples = {"and_truth", "or_truth"}
    fuzzy_output = True
