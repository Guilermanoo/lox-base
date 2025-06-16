from lox import *
from lox import testing
from lox.ast import *


class TestBlock(testing.ExerciseTester):
    is_expr = False
    src1 = "{}"
    src2 = '{ print "a"; print "b"; }'
    src3 = '{ print "a"; { print "b"; print "c"; } print "d"; }'
    tks1 = ""
    tks2 = "print a b"
    tks3 = "print a b c d"
    ast_class = Block

    def eval_env(self, n):
        ctx = {}
        prints = ["", "a\nb\n", "a\nb\nc\nd\n"]
        return (ctx, prints[n - 1])
