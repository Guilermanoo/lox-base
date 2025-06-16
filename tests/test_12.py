import pytest

from lox import *
from lox import testing
from lox.ast import *


class TestFor(testing.ExerciseTester):
    is_expr = False
    src1 = 'for (var i = 0; i < n; i = i + 1) print "a";'
    src2 = 'for (var i = 0; i < n; i = i + 1) { print "a"; print "b"; }'
    src3 = 'for (var i = 0; i < n; i = i + 1) for (var j = 0; j < m; j = j + 1) { print "a"; print "b"; }'
    tks1 = "i n a"
    tks2 = "i n a b"
    tks3 = "i n m j a b"
    ast_class = Block
    fuzzy_output = True

    def eval_env(self, n):
        ctx = {"n": float(n), "m": 2.0}
        prints = ["a\n" * n, "a\nb\n" * n, "a\nb\n" * (n * 2)]
        return (ctx, prints[n - 1])

    def test_aceita_for_sem_inicializador(self):
        src = """
        var i = 0;
        for (; i < 3; i = i + 1) print "a";
        """
        self.verify(src, {}, "a\na\na\n")

    def test_aceita_for_sem_incremento(self):
        src = "for (var i = 0; i < 5;) { i = i + 2; print i; }"
        self.verify(src, {}, "2\n4\n6\n")

    def test_aceita_for_sem_condição(self):
        src = """
        for (var i = 0;; i = i + 1) stop_at_3(i);
        """
        results = []

        def stop_at_3(i):
            results.append(i)
            if i >= 3:
                raise StopIteration

        with pytest.raises(StopIteration):
            self.verify(src, {"stop_at_3": stop_at_3}, {"i": 3})
        assert results == [0, 1, 2, 3]

    def test_aceita_for_vazio(self):
        n_iter = 0

        def body():
            nonlocal n_iter
            n_iter += 1

            if n_iter >= 10:
                raise StopIteration

        with pytest.raises(StopIteration):
            self.verify("for (;;) body();", {"body": body}, {})
        assert n_iter == 10

    def test_apagou_a_class_For(self):
        import lox.ast as lox_ast

        with pytest.raises(AttributeError):
            For = getattr(lox_ast, "For")
            print(f"{For=}")
            print("A classe For não deve existir mais no módulo AST, apague-a!")


class TestExamples(testing.ExampleTester):
    module = "for"
    examples = {
        "no_increment",
        "simple",
        "statement_condition",
        "statement_increment",
        "statement_initializer",
        "var_in_body",
        # "fun_in_body",
        # "class_in_body",
    }
    fuzzy_output = True
