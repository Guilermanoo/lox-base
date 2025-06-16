from lox import *
from lox import testing
from lox.ast import *


def test_criou_objeto_LoxClass():
    try:
        from lox.runtime import LoxClass as _
    except ImportError:
        pass
    try:
        from lox.ast import LoxClass as _
    except ImportError:
        msg = "Não consegui achar a classe LoxClass. Implemente-a no arquivo lox/runtime.py ou lox/ast.py"
        assert False, msg
    del _


class TestClassSyntax(testing.ExerciseTester):
    is_expr = False
    src1 = "class A {}"
    src2 = "class A { f() { return 42; } }"
    src3 = "class A < B {\n    f() { return 42; }\n    g() { return 42; }\n}"
    tks1 = "A"
    tks2 = "A f 42"
    tks3 = "A f g 42"
    ast_class = Class
    fuzzy_output = True

    def eval_env(self, n):  # -> tuple[dict[str, Any], Any]:
        ctx = {}
        self.parse("class B {}").eval(Ctx.from_dict(ctx))
        return (ctx, n)

    def verify_eval_result(self, result, stdout, ctx):
        a = ctx["A"]
        assert isinstance(a, LoxClass)


class TestExamples(testing.ExampleTester):
    module = "class"
    examples = {
        "empty",
        # "inherit_self",
        # "inherited_method",
        # "local_inherit_other",
        # "local_inherit_self",
        # "local_reference_self",
        # "reference_self",
    }
    fuzzy_output = True
