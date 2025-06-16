from dataclasses import dataclass

from lox import *
from lox import testing
from lox.ast import *


class TestWhile(testing.ExerciseTester):
    is_expr = False
    src1 = 'while (cond()) print "a";'
    src2 = 'while (cond()) { print "a"; print "b"; }'
    src3 = 'while (cond1()) while (cond2()) { print "a"; }'
    tks1 = "cond a"
    tks2 = "cond a b"
    tks3 = "cond1 cond2 a"
    ast_class = While
    fuzzy_output = True

    def eval_env(self, n):
        ctx = {
            "cond": Cond(max_iter=1),
            "cond1": Cond(max_iter=2),
            "cond2": Cond(max_iter=6),
        }
        prints = ["a\n", "a\nb\n", "a\n" * 6]
        return (ctx, prints[n - 1])

    def test_termina_laço_quando_a_condição_é_falsa(self):
        src = "while (n > 0) { print n; n = n - 1; }"
        _, stdout = self._eval_in_context(src, {"n": 4.0})
        self.assert_stdout_eq(stdout, "4\n3\n2\n1\n")


@dataclass
class Cond:
    max_iter: int
    n_iter: int = 0

    def __call__(self):
        self.n_iter += 1
        return self.n_iter <= self.max_iter


class TestExamples(testing.ExampleTester):
    module = "while"
    examples = {
        # "class_in_body",
        # "fun_in_body",
        "var_in_body",
    }
