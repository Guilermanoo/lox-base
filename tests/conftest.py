import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Iterable, NamedTuple

import pytest
from lark import Tree

import lox
from lox.ast import Expr, Program

pytest.register_assert_rewrite("lox.testing")

from lox import testing  # noqa: E402

BASE_DIR = Path(__file__).parent.parent
EXERCISES = BASE_DIR / "exercicios"
EXAMPLES = BASE_DIR / "exemplos"
EXERCISES_ALT = BASE_DIR / "exercícios"
LEX_REGEX = re.compile(
    r"""
    (?://\ *expect:\ (?P<EXPECT>[^\n]*))
    | (?://\ *expect\ runtime\ error:\ (?P<RUNTIME_ERROR>[^\n]*))
    | (?://[^\n]*Error\ at\ '(?P<ERROR_AT>[^'\n]*)'[^\n]*)
    | (?://[^\n]*Error\ at\ end:(?P<ERROR_EOF>[^\n]*))
    | (?://[^\n]*Error:(?P<ERROR>[^\n]*))
    | (?P<COMMENT>//[^\n])
    | (?P<IGNORE>[^"/]+|"[^"]*"|//[^\n]*)
    """,
    re.VERBOSE,
)


class Example(NamedTuple):
    src: str
    ast: Expr | Program
    cst: Tree


def pytest_addoption(parser):
    parser.addoption(
        "--full-suite",
        action="store_true",
        help="Run the full suite of tests. This includes all examples.",
    )


def pytest_runtest_setup(item):
    if "full_suite" in item.keywords and not item.config.getoption("--full-suite"):
        pytest.skip("need --full-suite option to run this test")


@pytest.fixture
def exercises_folder():
    """
    Retorna o caminho para a pasta de exercícios.
    """
    if EXERCISES.exists():
        return EXERCISES
    elif EXERCISES_ALT.exists():
        return EXERCISES_ALT
    else:
        raise FileNotFoundError(
            f"Não foi possível encontrar a pasta de exercícios. "
            f"Verifique se a pasta {EXERCISES} ou {EXERCISES_ALT} existe."
        )


@pytest.fixture()
def examples() -> Callable[[str], Iterable[testing.Example]]:
    """
    Carrega exemplos de código
    """

    return testing.load_examples


@pytest.fixture
def mod_loader(exercises_folder: Path):
    """
    Retorna o caminho para a pasta de exercícios.
    """

    def loader(name: str):
        file = exercises_folder / f"{name}.py"
        if not file.exists():
            raise FileNotFoundError(f"Arquivo {file} não encontrado.")

        ns: dict[str, Any] = {}
        with file.open("r") as fd:
            exec(fd.read(), ns)

        return SimpleNamespace(**ns)

    return loader


@pytest.fixture
def parser(expr: bool):
    def parser(src: str) -> Expr | Program:
        ast: Expr | Program
        if expr:
            ast = lox.parse_expr(src)
        ast = lox.parse(src)
        for subtree in ast.lark_descendents():
            msg = f"Árvore sintática ainda contém nós Lark: {subtree}"
            raise ValueError(msg)
        return ast

    return parser


@pytest.fixture
def cst(src, expr) -> Tree:
    return lox.parse_cst(src, expr=expr)


@pytest.fixture
def ast(src, parser):
    return parser(src)


@pytest.fixture
def cst_(src_, expr) -> Tree:
    return lox.parse_cst(src_, expr=expr)


@pytest.fixture
def ast_(src_, parser):
    return parser(src_)


@pytest.fixture
def cst__(src__, expr) -> Tree:
    return lox.parse_cst(src__, expr=expr)


@pytest.fixture
def ast__(src__, parser):
    return parser(src__)


@pytest.fixture
def ex(src, ast, cst):
    return Example(src, ast, cst)


@pytest.fixture
def ex_(src_, ast_, cst_):
    return Example(src_, ast_, cst_)


@pytest.fixture
def ex__(src__, ast__, cst__):
    return Example(src__, ast__, cst__)


@pytest.fixture
def astf(src, parser):
    return lambda: parser(src)


@pytest.fixture
def ast_f(src_, parser):
    return lambda: parser(src_)


@pytest.fixture
def ast__f(src__, parser):
    return lambda: parser(src__)


@pytest.fixture
def exs(ex, ex_, ex__) -> list[Example]:
    return [ex, ex_, ex__]
