import math

from lox import *
from lox import testing
from lox.ast import *


class TestUnaryOperators(testing.ExerciseTester):
    is_expr = True
    src1 = "-42"
    src2 = "!true"
    src3 = "-sqrt(9)"
    tks1 = "42"
    tks2 = "true"
    tks3 = "sqrt 9"
    ast_class = UnaryOp

    def eval_env(self, n):
        ctx = {"sqrt": math.sqrt}
        result = (-42, False, -math.sqrt(9))[n - 1]
        return (ctx, result)


class TestExamples(testing.ExampleTester):
    module = "operator"
    examples = {"negate", "negate_nonnum", "not_bool"}
    fuzzy_output = True
