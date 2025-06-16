import builtins
import math
from types import SimpleNamespace

from lox import *
from lox import testing
from lox.ast import *


class TestCall(testing.ExerciseTester):
    is_expr = True
    src1 = "sqrt(2 + 2)"
    src2 = "obj.method(1, 2, 3)"
    src3 = 'foo("arg")("sub-arg-1", sub_arg_2())'
    tks1 = "sqrt 2"
    tks2 = "obj method 1 2 3"
    tks3 = "foo arg sub_arg_2 sub-arg-1"
    ast_class = Call

    def eval_env(self, n):
        ctx = {
            "sqrt": math.sqrt,
            "obj": SimpleNamespace(method=lambda *args: args),
            "foo": lambda *args: lambda *args2: args + args2,
            "sub_arg_2": lambda *args: args,
        }
        result = builtins.eval(self.src(n), ctx.copy())
        return (ctx, result)

    def test_renomeou_o_atributo_name_da_ast(self):
        ast = self.ast(1)
        if hasattr(ast, "name"):
            msg = "Renomeie o atributo Call.name para algo mais informativo já que agora devemos guardar um nó e não uma string."
            raise ValueError(msg)


class TestExamples(testing.ExampleTester):
    module = "call"
    examples = {"bool", "nil", "num", "string"}
    fuzzy_output = False
