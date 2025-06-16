from lox import *
from lox import testing
from lox.ast import *


class TestIf(testing.ExerciseTester):
    is_expr = False
    src1 = 'if (x) print "a"; else print "b";'
    src2 = 'if (x) print "a";'
    src3 = 'if (!x) if (y) print "a"; else print "b";'
    tks1 = "x a b"
    tks2 = "x a"
    tks3 = "x a y b c"
    ast_class = If

    def eval_env(self, n):
        ctx = {"x": True, "y": True}
        prints = ["a\n", "a\n", ""]
        return (ctx, prints[n - 1])

    def eval_env_alt(self, n):
        ctx = {"x": False, "y": False}
        prints = ["b\n", "", "b\n"]
        return (ctx, prints[n - 1])


class TestExamples(testing.ExampleTester):
    module = "if"
    examples = {
        "dangling_else",
        "else",
        "if",
        "var_in_else",
        "var_in_then",
    }
    fuzzy_output = True
