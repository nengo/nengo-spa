"""Representation of simple expression trees."""

from collections import namedtuple


# Reference precedence table:
# https://docs.python.org/3/reference/expressions.html
precedence = {
    op: p
    for p, operators in enumerate(
        (
            {":="},
            {"lambda"},
            {"if", "else"},
            {"or"},
            {"and"},
            {"not x"},
            {"in", "not in", "is", "is not", "<", "<=", ">", ">=", "!=", "=="},
            {"|"},
            {"^"},
            {"&"},
            {"<<", ">>"},
            {"+", "-"},
            {"*", "@", "/", "//", "%"},
            {"+x", "-x", "~x"},
            {"**"},
            {"await x"},
            {"x[index]", "x[index:index]", "x(arguments...)", "x.attribute"},
            {
                "(expressions...)",
                "[expressions...]",
                "{key: value...}",
                "{expressions...}",
            },
        )
    )
    for op in operators
}


class Node(namedtuple("Node", ["value", "precedence", "children"])):
    def __invert__(self):
        return UnaryOperator("~", self)

    def __neg__(self):
        return UnaryOperator("-", self)

    def __pos__(self):
        return UnaryOperator("+", self)

    def __add__(self, other):
        return BinaryOperator("+", self, other)

    def __radd__(self, other):
        return BinaryOperator("+", other, self)

    def __sub__(self, other):
        return BinaryOperator("-", self, other)

    def __rsub__(self, other):
        return BinaryOperator("-", other, self)

    def __mul__(self, other):
        return BinaryOperator("*", self, other)

    def __rmul__(self, other):
        return BinaryOperator("*", other, self)

    def __truediv__(self, other):
        return BinaryOperator("/", self, other)

    def __floordiv__(self, other):
        return BinaryOperator("//", self, other)

    def __mod__(self, other):
        return BinaryOperator("%", self, other)

    def __matmul__(self, other):
        return BinaryOperator("@", self, other)

    def __rmatmul__(self, other):
        return BinaryOperator("@", other, self)

    def __pow__(self, other):
        return BinaryOperator("**", self, other)

    def __lshift__(self, other):
        return BinaryOperator("<<", self, other)

    def __rshift__(self, other):
        return BinaryOperator(">>", self, other)


class Leaf(Node):
    __slots__ = ()

    def __new__(cls, value):
        return super().__new__(cls, value, precedence["(expressions...)"], tuple())

    def __str__(self):
        return str(self.value)


class UnaryOperator(Node):
    __slots__ = ()

    def __new__(cls, value, child):
        return super().__new__(cls, value, precedence[value + "x"], (child,))

    def __str__(self):
        child = self.children[0]
        operand = str(child)
        if self.precedence >= child.precedence:
            operand = "({})".format(operand)
        return self.value + operand


class BinaryOperator(Node):
    __slots__ = ()

    def __new__(cls, value, lhs, rhs):
        return super().__new__(cls, value, precedence[value], (lhs, rhs))

    def __str__(self):
        lhs, rhs = self.children
        if self.precedence > lhs.precedence:
            lhs = "({})".format(lhs)
        if self.precedence >= rhs.precedence:
            rhs = "({})".format(rhs)
        return "{} {} {}".format(lhs, self.value, rhs)
