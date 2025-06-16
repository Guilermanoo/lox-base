from lox import *
from lox import testing
from lox.ast import *


class TestFunction(testing.ExerciseTester):
    is_expr = False
    src1 = "fun f() { return 42; }"
    src2 = "fun f(a, b) { return a + b; }"
    src3 = "fun f(a) { fun g(b) { return a + b; } return g(2); }"
    tks1 = "f 42"
    tks2 = "f a b"
    tks3 = "f g a b 2"
    ast_class = Function
    fuzzy_output = True

    def eval_env(self, n):
        ctx = {"n": n, "m": 2}
        return (ctx, n)

    def verify_eval_result(self, result, stdout, ctx):
        f = ctx["f"]

        match result:
            case 1:
                assert f() == 42
            case 2:
                assert f(40, 2) == 42
                assert f(10, 3) == 13
            case 3:
                assert f(40) == 42
                assert f(2) == 4

    def test_declara_função_sem_return(self):
        src = "fun f() { }"

        self._verify_ast(src)


class TestExamples(testing.ExampleTester):
    module = "function"
    examples = [
        "max",
        "print_function",
        "empty_body",
        "extra_arguments",
        "local_recursion",
        "missing_arguments",
        "mutual_recursion",
        "nested_call_with_arguments",
        "parameters",
        "recursion",
        "missing_comma_in_parameters",
    ]
    fuzzy_output = True
