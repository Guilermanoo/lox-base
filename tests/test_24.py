from lox import *
from lox import testing
from lox.ast import *

src_vec = """
class Vec {
    init(x, y) {
        this.x = x;
        this.y = y;
    } 
}

var u = Vec(1, 2);
"""


src_single_arg = """
class A {
    init(a) {
        this.x = 42;
        if (a > 10) return;
        this.x = a;
    } 
}

var u = A(5);
var v = A(20);
print u.x;
print v.x;
"""

src_no_arg = """
class A {
    init() {
        this.x = 1;
        this.y = 2;
    } 
}

var u = A();
print u.x + u.y;
"""


class TestInit(testing.ExerciseTester):
    is_expr = False
    test_ast = test_cst = test_eval = False
    fuzzy_output = True

    def test_executa_o_método_init_e_salva_atributos(self):
        self.verify(src_vec, ctx := {}, expect_none=True)
        u = ctx["u"]

        assert u.x == 1
        assert u.y == 2

    def test_executa_init_com_1_argumento(self):
        self.verify(src_single_arg, {}, "5\n42\n")

    def test_executa_init_sem_argumentos(self):
        self.verify(src_no_arg, {}, "3\n")

    def test_init_pode_ser_executado_numa_instância_existente(self):
        self.verify(src_vec, ctx := {}, expect_none=True)
        self.verify("u.init(3, 4); print u.x + u.y;", ctx, "7\n")

    def test_executado_em_instância_retorna_instância(self):
        self.verify(src_vec, ctx := {}, expect_none=True)
        self.verify("var v = u.init(3, 4);", ctx, expect_none=True)
        u = ctx["u"]
        v = ctx["v"]
        assert v is u


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


class TestExamplesConstructor(testing.ExampleTester):
    module = "constructor"
    examples = {
        "arguments",
        # "call_init_early_return",
        # "call_init_explicitly",
        "default",
        "default_arguments",
        "early_return",
        "extra_arguments",
        "init_not_method",
        "missing_arguments",
        "return_in_nested_function",
        # "return_value",
    }
