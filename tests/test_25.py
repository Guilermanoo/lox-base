from lox import *
from lox import testing
from lox.ast import *


class TestSuper(testing.ExerciseTester):
    is_expr = False
    test_ast = test_cst = test_eval = False
    fuzzy_output = True

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


class TestExamplesSuper(testing.ExampleTester):
    module = "super"
    examples = {
        "bound_method",
        "call_other_method",
        "call_same_method",
        "closure",
        "constructor",
        "extra_arguments",
        "indirectly_inherited",
        "missing_arguments",
        # "no_superclass_bind",
        # "no_superclass_call",
        # "no_superclass_method",
        "parenthesized",
        "reassign_superclass",
        # "super_at_top_level",
        "super_in_closure_in_inherited_method",
        "super_in_inherited_method",
        # "super_in_top_level_function",
        "super_without_dot",
        "super_without_name",
        "this_in_superclass_method",
    }
    fuzzy_output = True
