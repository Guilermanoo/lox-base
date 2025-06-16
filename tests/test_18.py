import pytest

from lox import *
from lox import testing
from lox.ast import *
from lox.runtime import LoxError


class TestNonPythonBehavior(testing.ExerciseTester):
    is_expr = False
    test_ast = False
    test_cst = False
    test_eval = False

    @pytest.mark.parametrize("item", "1.5 -1.5 42 10 100".split())
    def test_show_imprime_números_corretamente(self, item):
        self.test_show_imprime_primitivo_corretamente(item)

    @pytest.mark.parametrize("item", ["nil", "true", "false"])
    def test_show_imprime_primitivo_corretamente(self, item):
        src = f"print {item};"
        out = f"{item}\n"
        self.verify(src, {}, out)

    def test_imprime_classes_corretamente(self):
        src = "class Foo {}\nprint Foo;"
        out = "Foo\n"
        self.verify(src, {}, out)

    def test_imprime_instâncias_corretamente(self):
        src = "class Foo {}\nvar foo = Foo();\nprint foo;"
        out = "Foo instance\n"
        self.verify(src, {}, out)

    def test_imprime_funções_corretamente(self):
        src = "fun foo() {}\nprint foo;"
        out = "<fn foo>\n"
        self.verify(src, {}, out)

    def test_imprime_funções_nativas_corretamente(self):
        self.verify("print clock;", {}, "<native fn>\n")

    @pytest.mark.parametrize(
        "src",
        [
            "true + true",
            "1 + true",
            "1 - false",
            "1 * true",
            "1 / true",
            "false + 1",
            "true - 1",
            "1 * false",
            "1 / false",
            "true > 0",
            "false < 1",
            "false >= 0",
            "true <= 1",
            '"foo" > "bar"',
            '"foo" < "bar"',
            '"foo" >= "bar"',
            '"foo" <= "bar"',
            '"foo" * "bar"',
        ],
    )
    def test_proíbe_operações_matemáticas_com_booleanos(self, src):
        src = f"print {src};"
        self.verify(src, {}, expect_raises=(LoxError, TypeError))

    @pytest.mark.parametrize(
        "src, expect",
        [
            ("true != 1", "true"),
            ("false != 0", "true"),
            ("true == 1", "false"),
            ("false == 0", "false"),
        ],
    )
    def test_igualde_estrita_com_booleanos(self, src, expect):
        src = f"print {src};"
        out = f"{expect}\n"
        self.verify(src, {}, out)

    @pytest.mark.parametrize("src", ["0", '""', "!!0", "!nil", '!!""'])
    def test_somente_false_e_nil_são_falsy(self, src):
        src = f'if ({src}) print "ok";'
        self.verify(src, {}, "ok\n")

    @pytest.mark.parametrize("src", ["0", '""', "!!0"])
    def test_condições_no_loop_while(self, src):
        src = f"while ({src}) nil.does_not_exist;"
        self.verify(src, {}, expect_raises=(LoxError, AttributeError))

    def test_aceita_soma_de_strings(self):
        src = 'print "foo" + "bar";'
        out = "foobar\n"
        self.verify(src, {}, out)


class TestExamples(testing.ExampleTester):
    module = "logical_operator"
    fuzzy_output = False
