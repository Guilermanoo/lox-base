import builtins

from lox import *
from lox import testing
from lox.ast import *


class TestSetattr(testing.ExerciseTester):
    is_expr = True
    src1 = "obj.attr = 42"
    src2 = "func(obj).attr = 42"
    src3 = "obj.attr.subattr = 42"
    tks1 = "obj attr 42"
    tks2 = "func obj attr 42"
    tks3 = "obj attr subattr 42"
    ast_class = Setattr

    def eval_env(self, n):
        ctx = {"func": lambda x: x, "obj": Obj()}
        src = self.src(n)
        builtins.exec(src, ctx.copy())
        return (ctx, 42.0)

    def test_atribuição_de_atributo_modifica_objeto(self):
        obj = Obj()
        ctx = {"obj": obj}
        print(ctx)
        expect = Obj(attr=Obj(subattr=42.0))
        self.verify("obj.attr.subattr = 42", ctx, expect_none=True)
        ast = self.parse("obj.attr.subattr = 42")
        ast.eval(Ctx.from_dict({"obj": obj}))
        assert obj == expect


class Obj:
    def __init__(self, **kwargs):
        self._data = kwargs

    def __getattr__(self, attr):
        # prevent some pytest magic with mocks
        if attr == "awehoi234_wdfjwljet234_234wdfoijsdfmmnxpi492":
            raise AttributeError
        if attr.startswith("_"):
            raise AttributeError

        try:
            return self._data[attr]
        except KeyError:
            value = Obj()
            self._data[attr] = value
            return value

    def __setattr__(self, attr, value):
        if attr.startswith("_"):
            super().__setattr__(attr, value)
        else:
            self._data[attr] = value

    def __eq__(self, other):
        if isinstance(other, Obj):
            return self._data == other._data
        return NotImplemented

    def __repr__(self):
        data = ", ".join(f"{k}={v!r}" for k, v in self._data.items())
        return f"Obj({data})"

    def __str__(self):
        return repr(self)


class TestExamples(testing.ExampleTester):
    module = "field"
    examples = [
        "get_on_bool",
        "get_on_nil",
        "get_on_num",
        "get_on_string",
        "set_on_bool",
        "set_on_nil",
        "set_on_num",
        "set_on_string",
    ]
    fuzzy_output = True
