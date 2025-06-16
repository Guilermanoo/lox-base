from lox import *
from lox import testing
from lox.ast import *


class TestThis(testing.ExerciseTester):
    is_expr = False
    test_ast = test_cst = test_eval = False
    fuzzy_output = True

    def test_método_com_this(self):
        src = 'class A { f() { return this.foo; }}\nvar a = A();a.foo = "foo";\nprint a.f();'
        self.verify(src, {}, "foo\n")

    def test_método_usa_o_mecanismo_de_bind(self):
        src = 'class A { f() { return this.bar; } }\nvar a = A();\na.bar = "bar";\nvar f = a.f;\nprint f();'
        self.verify(src, ctx := {}, "bar\n")

        instance = ctx["a"]
        method = ctx["f"]

        for obj in method.__dict__.values():
            if isinstance(obj, Ctx):
                bound_ctx = obj
                break
        else:
            msg = "Não foi possível encontrar o contexto vinculado ao método"
            raise AssertionError(msg)

        assert bound_ctx["this"] == instance

    def test_bind_retorna_funções_diferentes_para_diferentes_binds(self):
        src = "class A { f() { return this.x; } }\nvar a = A();\na.x = 1;\nvar f = a.f; var b = A(); b.x = 2;\nvar g = b.f;\nprint f();\nprint g();"
        self.verify(src, ctx := {}, "1\n2\n")

        f = ctx["f"]
        g = ctx["g"]
        A_f = ctx["A"].get_method("f")
        assert f is not g, "LoxFunction.bind() deve retornar uma nova função"
        assert f is not A_f
        assert g is not A_f


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


class TestExamplesThis(testing.ExampleTester):
    module = "this"
    examples = {
        "closure",
        "nested_class",
        "nested_closure",
        # "this_at_top_level",
        "this_in_method",
        # "this_in_top_level_function",
    }
    fuzzy_output = True


class TestExamplesField(testing.ExampleTester):
    module = "field"
    fuzzy_output = True
