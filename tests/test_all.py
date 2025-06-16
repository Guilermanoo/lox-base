from pathlib import Path

import pytest

import lox.testing

EXAMPLES_PATH = Path(__file__).parent.parent / "exemplos"


def examples():
    yield from lox.testing.load_examples("")

    excluded_mods = {"benchmark", "scanning", "limit", "expressions"}
    excluded_tests = {
        # Sem limites arbitrários
        "method": {
            "too_many_arguments",
            "too_many_parameters",
        },
        "function": {
            "too_many_arguments",
            "too_many_parameters",
            "local_mutual_recursion",
        },
        # Py-lox não realiza o early binding
        "closure": {
            "close_over_method_parameter",
            "assign_to_shadowed_later",
        },
        "variable": {
            "early_bound",
        },
    }

    for subdir in sorted(EXAMPLES_PATH.iterdir()):
        if not subdir.is_dir() or (mod := subdir.name) in excluded_mods:
            continue

        exclude = excluded_tests.get(mod, set())
        yield from lox.testing.load_examples(mod, exclude=exclude)


def get_id(example: Path):
    """
    Gera um ID para o exemplo.
    """
    path = example.relative_to(EXAMPLES_PATH)
    return str(path).removesuffix(".lox").replace("\\", "/")


@pytest.mark.full_suite
@pytest.mark.parametrize("path", exs := [*examples()], ids=map(get_id, exs))
def test_all(path: Path):
    src = path.read_text(encoding="utf-8")
    example = lox.testing.Example(src, path)
    example.test_example()
