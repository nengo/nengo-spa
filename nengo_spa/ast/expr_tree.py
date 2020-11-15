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


class EllipsisLeaf(Leaf):
    __slots__ = ()

    def __new__(cls, value="..."):
        return super().__new__(cls, value)


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

    @property
    def lhs(self):
        return self.children[0]

    @property
    def rhs(self):
        return self.children[1]


class AttributeAccess(Node):
    __slots__ = ()

    def __new__(cls, value, child):
        return super().__new__(cls, value, precedence["x.attribute"], (child,))

    def __str__(self):
        child = self.children[0]
        operand = str(child)
        if self.precedence > child.precedence:
            operand = "({})".format(operand)
        return operand + "." + self.value


class FunctionCall(Node):
    __slots__ = ()

    def __new__(cls, arguments, child):
        return super().__new__(
            cls,
            [arg if isinstance(arg, Node) else Leaf(arg) for arg in arguments],
            precedence["x(arguments...)"],
            (child,),
        )

    def __str__(self):
        child = self.children[0]
        operand = str(child)
        if self.precedence > child.precedence:
            operand = "({})".format(operand)
        return "{}({})".format(operand, ", ".join(str(arg) for arg in self.value))


class KeywordArgument(Node):
    __slots__ = ()

    def __new__(cls, value, child):
        return super().__new__(cls, value, 0, (child,))

    def __str__(self):
        return "{}={}".format(self.value, str(self.children[0]))


class _LimitStrLengthVisitor:
    # General idea of the algorithm: Do an in-order traversal and keep track
    # how much characters we may still use (max_len). When the limit is
    # exceeded, start replacing nodes with ellipsis and re-adjust max_len to
    # account for the characters saved through the replacement.

    def __init__(self, max_len):
        self.max_len = max_len

    def visit_Leaf(self, node):
        if len(node.value) <= 3 or len(node.value) <= self.max_len:
            self.max_len -= len(node.value)
            return node
        else:
            self.max_len -= 3
            return EllipsisLeaf()

    def visit_EllipsisLeaf(self, node):
        self.max_len -= len(str(node))
        return node

    def visit_UnaryOperator(self, node):
        self.max_len -= len(node.value)
        if node.precedence >= node.children[0].precedence:
            self.max_len -= 2
        return UnaryOperator(node.value, self.visit_node(node.children[0]))

    def visit_BinaryOperator(self, node):
        initial_max_len = self.max_len

        # Operator + 2 spaces + reserve 3 characters for the second operand, so
        # that it can be at least turned into an ellipsis
        self.max_len -= len(node.value) + 2 + 3
        lhs = self.visit_node(node.lhs)
        self.max_len += 3  # return reserved characters for the second operand
        # Adjust for parenthesis due to operator precedence
        if node.precedence > lhs.precedence:
            self.max_len -= 2
        if node.precedence >= node.rhs.precedence:
            self.max_len -= 2

        rhs = self.visit_node(node.rhs)
        if self.max_len < 0:
            if node.precedence >= node.rhs.precedence:
                self.max_len += 2
            self.max_len += len(str(rhs)) - 3
            rhs = EllipsisLeaf()
        if self.max_len >= 0:
            return BinaryOperator(node.value, lhs, rhs)
        else:
            self.max_len = initial_max_len - 3
            return EllipsisLeaf()

    def visit_AttributeAccess(self, node):
        initial_max_len = self.max_len
        self.max_len -= len(node.value) + 1
        child = self.visit_node(node.children[0])
        if node.precedence > child.precedence:
            self.max_len -= 2
        if self.max_len < 0:
            self.max_len += len(str(child)) - 5
            child = EllipsisLeaf("(...)")
        if self.max_len >= 0:
            return AttributeAccess(node.value, child)
        else:
            self.max_len = initial_max_len - 3
            return EllipsisLeaf()

    def visit_FunctionCall(self, node):
        initial_max_len = self.max_len
        child = self.visit_node(node.children[0])
        self.max_len -= 2  # parenthesis invoking call
        if node.precedence > child.precedence:
            self.max_len -= 2  # parenthesis around function

        args = []
        for i, arg in enumerate(node.value):
            if i > 0:
                self.max_len -= 2  # comma + space argument separator
            arg = self.visit_node(arg)
            if i > 0 and self.max_len < 0:
                self.max_len += (
                    len(str(args[-1])) - 3
                )  # replacement of previous arg with ellipsis
                self.max_len += (
                    len(str(arg)) + 2
                )  # adjust for dropped, but processed arg and separator
                args[-1] = EllipsisLeaf()
            else:
                args.append(arg)

        if self.max_len >= 0:
            return FunctionCall(args, child)
        else:
            self.max_len = initial_max_len - 3
            return EllipsisLeaf()

    def visit_KeywordArgument(self, node):
        initial_max_len = self.max_len
        self.max_len -= len(str(node.value)) + 1  # 1 for assignment (=)
        child = self.visit_node(node.children[0])
        if self.max_len >= 0:
            return KeywordArgument(node.value, child)
        else:
            self.max_len = initial_max_len - 3
            return EllipsisLeaf()

    def visit_node(self, node):
        return getattr(self, "visit_" + type(node).__name__)(node)


def limit_str_length(expr_tree, max_len):
    """Returns a modified expression tree with a string length limited to
    approximately *max_len*."""
    return _LimitStrLengthVisitor(max_len).visit_node(expr_tree)
