from lox import *
from lox import testing
from lox.ast import *


def test_criou_objeto_LoxInstance():
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


class TestLoxClassCreation(testing.ExerciseTester):
    is_expr = False
    src1 = "class A {}"
    src2 = "class A { f() { return 42; } }"
    src3 = "class A < B { f() { return 42; } g() { return 42; } }"
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

    def test_lox_class_implementa_get_method_e_base(self):
        src = "class A { f() { } }"
        self.verify(src, ctx := {}, expect_none=True)

        base = ctx["A"]
        assert isinstance(base, LoxClass), "A não é uma instância de LoxClass"
        assert base.get_method("f").name == "f", "Método f não foi criado corretamente"

        src = "class B < A { g() { } }"
        self.verify(src, ctx, expect_none=True)
        cls = ctx["B"]

        assert isinstance(cls, LoxClass), "B não é uma instância de LoxClass"
        assert cls.get_method("f").name == "f", "Método f não foi criado corretamente"
        assert cls.get_method("f") is base.get_method(
            "f"
        ), "Subclasse B deve herdar o método f da superclasse A"
        assert cls.get_method("g").name == "g", "Método g não foi criado corretamente"

    def test_podemos_chamar_lox_class_para_criar_instâncias(self):
        src = "class A {}"
        self.verify(src, ctx := {}, expect_none=True)

        cls = ctx["A"]
        assert callable(cls), "A classe A não é chamável, implemente __call__"

        instance = cls()
        assert isinstance(
            instance, LoxInstance
        ), "A classe A deve retornar objetos LoxInstance"


class TestExamples(testing.ExampleTester):
    module = "class"
    examples = {
        "empty",
        # "inherit_self",
        "inherited_method",
        "local_inherit_other",
        # "local_inherit_self",
        "local_reference_self",
        "reference_self",
    }
    fuzzy_output = True
