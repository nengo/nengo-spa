import pytest

from nengo_spa.ast.expr_tree import (
    BinaryOperator,
    Leaf,
    UnaryOperator,
)


def test_leaf_str():
    assert str(Leaf("foo")) == "foo"


def test_unary_operator_str():
    assert str(UnaryOperator("~", UnaryOperator("-", Leaf("leaf")))) == "~(-leaf)"
    assert (
        str(UnaryOperator("~", BinaryOperator("+", Leaf("a"), Leaf("b")))) == "~(a + b)"
    )
    assert (
        str(UnaryOperator("~", BinaryOperator("**", Leaf("a"), Leaf("b")))) == "~a ** b"
    )


def test_binary_operator_str():
    a, b, c = Leaf("a"), Leaf("b"), Leaf("c")
    assert str(BinaryOperator("+", a, BinaryOperator("*", b, c))) == "a + b * c"
    assert str(BinaryOperator("*", BinaryOperator("+", a, b), c)) == "(a + b) * c"
    assert str(BinaryOperator("-", BinaryOperator("+", a, b), c)) == "a + b - c"


@pytest.mark.parametrize("op", ["+", "-", "~"])
def test_unary_operator(op):
    leaf = Leaf("leaf")
    assert leaf
    assert str(eval(op + "leaf")) == op + "leaf"


@pytest.mark.parametrize("op", ["<<", ">>", "+", "-", "*", "@", "/", "//", "%", "**"])
def test_binary_operator(op):
    lhs, rhs = Leaf("lhs"), Leaf("rhs")
    assert lhs
    assert rhs
    assert str(eval("lhs" + op + "rhs")) == "lhs {} rhs".format(op)
