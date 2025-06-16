from lox import *
from lox import testing
from lox.ast import *


class TestExamplesVarAndBlock(testing.ExampleTester):
    module = "variable"
    examples = [
        "use_nil_as_var",
        "use_return_as_var",
        "use_return_as_assign",
        "use_return_as_expr",
        "use_false_as_var",
        "use_nil_as_param",
        "use_true_as_param",
        "duplicate_local",
        "duplicate_parameter",
        "collide_with_parameter",
    ]
    fuzzy_output = False
