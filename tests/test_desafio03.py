import pytest

from lox import *
from lox import testing
from lox.ast import *


@pytest.mark.full_suite
class TestExamplesOperator(testing.ExampleTester):
    module = "operator"
