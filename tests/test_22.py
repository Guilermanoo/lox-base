from lox import *
from lox import testing
from lox.ast import *


def test_criou_objeto_LoxInstance():
    try:
        from lox.runtime import LoxInstance as _
    except ImportError:
        pass
    try:
        from lox.ast import LoxInstance as _
    except ImportError:
        msg = "Não consegui achar a classe LoxInstance. Implemente-a no arquivo lox/runtime.py ou lox/ast.py"
        assert False, msg
    del _


class TestLoxInstance(testing.ExerciseTester):
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

    def test_é_possível_salvar_campos_da_instância(self):
        src = 'class A {}\nvar a = A();a.foo = "foo"; a.bar = "bar";\nprint a.foo + a.bar;'
        self.verify(src, {}, "foobar\n")

    def test_instância_consegue_chamar_métodos(self):
        src = 'class A { f() { return "a"; } }\nvar a = A();\nprint a.f();'
        self.verify(src, {}, "a\n")

    def test_instância_consegue_chamar_métodos_de_superclasse(self):
        src = 'class A { f() { return "a"; } }\nclass B < A { g() { return super.f(); } }\nvar b = B();\nprint b.g();'
        self.verify(src, {}, "a\n")

    def test_instância_consegue_chamar_métodos_de_superclasse_de_superclasse(self):
        src = (
            'class A { f() { return "a"; } }\n'
            "class B < A { g() { return super.f(); } }\n"
            "class C < B { h() { return super.g(); } }\n"
            "var c = C();\nprint c.h();"
        )
        self.verify(src, {}, "a\n")

    def test_representação_de_instância(self):
        src = "class A {}\nvar a = A();\nprint a;"
        self.verify(src, {}, "A instance\n")


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
