from lox import *
from lox import testing
from lox.ast import *


class TestExamplesReturn(testing.ExampleTester):
    module = "return"


class TestExamplesThis(testing.ExampleTester):
    module = "this"


class TestExamplesSuper(testing.ExampleTester):
    module = "super"
