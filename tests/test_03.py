from types import SimpleNamespace

from lox import *
from lox import testing
from lox.ast import *


class TestAttributes(testing.ExerciseTester):
    is_expr = True
    src1 = "obj.attr"
    src2 = "(x + y).attr"
    tks1 = "obj attr"
    tks2 = "x y attr"
    ast_class = Getattr

    def test_ast_salva_nome_do_atributo_como_string(self):
        types = set(map(type, self.ast(1).__dict__.values()))
        if str not in types:
            msg = f"O nome do atributo deve ser uma string, mas achei atributos dos tipos: {types}"
            raise ValueError(msg)

    def eval_env1(self):
        obj = SimpleNamespace(attr=42)
        return ({"obj": obj}, 42)

    def eval_env2(self):
        class Num(float):
            def __add__(self, other):
                return Num(super().__add__(other))

            @property
            def attr(self):
                return int(self)

        x = Num(3)
        y = Num(0.1415)
        return ({"x": x, "y": y}, (x + y).attr)


class TestExamples(testing.ExampleTester):
    module = "field"
    examples = {
        "get_on_bool",
        "get_on_nil",
        "get_on_num",
        "get_on_string",
        "set_on_bool",
        "set_on_nil",
        "set_on_num",
        "set_on_string",
    }
