import pytest

from nengo_spa.ast.expr_tree import (
    AttributeAccess,
    BinaryOperator,
    EllipsisLeaf,
    FunctionCall,
    KeywordArgument,
    Leaf,
    UnaryOperator,
    limit_str_length,
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


def test_attribute_access_str():
    assert str(AttributeAccess("attr", Leaf("foo"))) == "foo.attr"
    assert (
        str(AttributeAccess("attr", BinaryOperator("+", Leaf("a"), Leaf("b"))))
        == "(a + b).attr"
    )


def test_function_call_str():
    assert str(FunctionCall((), Leaf("fn"))) == "fn()"
    assert str(FunctionCall(("str1", "str2"), Leaf("fn"))) == "fn(str1, str2)"
    assert (
        str(FunctionCall((UnaryOperator("~", Leaf("foo")),), Leaf("fn"))) == "fn(~foo)"
    )
    assert (
        str(FunctionCall((KeywordArgument("kwarg", "val"),), Leaf("fn")))
        == "fn(kwarg=val)"
    )
    assert (
        str(FunctionCall((), BinaryOperator("+", Leaf("a"), Leaf("b")))) == "(a + b)()"
    )


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


@pytest.mark.parametrize(
    "expr_tree,max_len,expected",
    [
        (expr_tree, len(str(expr_tree)), expr_tree)
        for expr_tree in (
            Leaf("foo"),
            Leaf("not_too_long"),
            -Leaf("foo"),
            Leaf("a") + Leaf("b"),
            AttributeAccess("attr", Leaf("foo")),
            FunctionCall(("arg",), Leaf("foo")),
        )
    ]
    + [
        (Leaf("foo"), 0, Leaf("foo")),
        (Leaf("too_long"), len("too_long") - 1, EllipsisLeaf()),
        (Leaf("too_long"), 3, EllipsisLeaf()),
        (Leaf("too_long"), 0, EllipsisLeaf()),
        (
            Leaf("varA") + Leaf("varB"),
            len("varA + varB") - 1,
            Leaf("varA") + EllipsisLeaf(),
        ),
        (Leaf("varA") + Leaf("varB"), len("varA + ..."), Leaf("varA") + EllipsisLeaf()),
        (
            Leaf("varA") + Leaf("varB"),
            len("varA + ...") - 1,
            EllipsisLeaf() + EllipsisLeaf(),
        ),
        (Leaf("varA") + Leaf("varB"), 3, EllipsisLeaf()),
        (Leaf("a") + Leaf("b"), len("a + b") - 1, EllipsisLeaf()),
        (
            (Leaf("a") + Leaf("b")) * Leaf("c"),
            len("(a + b) * c") - 1,
            EllipsisLeaf() * Leaf("c"),
        ),
        ((Leaf("a") + Leaf("b")) * Leaf("c"), len("(a + b) * c") - 5, EllipsisLeaf()),
        (
            (Leaf("a") + Leaf("b")) * Leaf("varC"),
            len("(a + b) * varC") - 1,
            (Leaf("a") + Leaf("b")) * EllipsisLeaf(),
        ),
        (
            Leaf("a") * (Leaf("b") + Leaf("c")),
            len("a * (b + c)") - 1,
            Leaf("a") * EllipsisLeaf(),
        ),
        (
            Leaf("a") * (Leaf("b") + Leaf("varC")),
            len("a * (b + varC)") - 1,
            Leaf("a") * (Leaf("b") + EllipsisLeaf()),
        ),
        (
            Leaf("a") + Leaf("a") * Leaf("b") + Leaf("a") * Leaf("b"),
            len("a + a * b + a * b") - 2,
            Leaf("a") + Leaf("a") * Leaf("b") + EllipsisLeaf(),
        ),
        (
            Leaf("a") / (Leaf("a") * Leaf("b")) / (Leaf("a") * Leaf("b")),
            len("a / (a * b) / (a * b)") - 3,
            Leaf("a") / (Leaf("a") * Leaf("b")) / EllipsisLeaf(),
        ),
        (
            AttributeAccess("attribute", Leaf("varA") + Leaf("varB")),
            len("(varA + varB).attribute") - 3,
            AttributeAccess("attribute", EllipsisLeaf("(...)")),
        ),
        (
            AttributeAccess("attribute", Leaf("a")),
            len("a.attribute") - 1,
            EllipsisLeaf(),
        ),
        (FunctionCall([], Leaf("fn_name")), len("...()") - 1, EllipsisLeaf()),
        (
            FunctionCall([], Leaf("fn_name")),
            len("fn_name()") - 1,
            EllipsisLeaf(),
        ),
        (
            FunctionCall([], Leaf("a") + Leaf("b")),
            len("(a + b)()") - 1,
            EllipsisLeaf(),
        ),
        (
            FunctionCall(["arg1", "arg2"], Leaf("fn_name")),
            len("fn_name(arg1, arg2)") - 1,
            FunctionCall(["arg1", EllipsisLeaf()], Leaf("fn_name")),
        ),
        (
            FunctionCall(["arg1", "arg2"], Leaf("fn_name")),
            len("fn_name(arg1, arg2)") - 2,
            FunctionCall([EllipsisLeaf()], Leaf("fn_name")),
        ),
        (
            FunctionCall(["arg1", "arg2"], Leaf("fn_name")),
            len("fn_name(arg1, arg2)") - 8,
            EllipsisLeaf(),
        ),
        (
            FunctionCall(["arg1", "arg2"], Leaf("a") + Leaf("b")),
            len("(a + b)(arg1, arg2)") - 1,
            FunctionCall(["arg1", "..."], Leaf("a") + Leaf("b")),
        ),
        (
            FunctionCall([Leaf("arg1") + Leaf("arg2")], Leaf("fn_name")),
            len("fn_name(arg1 + arg2)") - 1,
            FunctionCall([Leaf("arg1") + EllipsisLeaf()], Leaf("fn_name")),
        ),
        (
            Leaf("var") + FunctionCall([], Leaf("fn_name")),
            len("var + fn_name()") - 1,
            Leaf("var") + Leaf("..."),
        ),
        (
            FunctionCall([KeywordArgument("key", Leaf("value"))], Leaf("fn_name")),
            len("fn_name(key=value)") - 1,
            FunctionCall([KeywordArgument("key", EllipsisLeaf())], Leaf("fn_name")),
        ),
        (
            FunctionCall([KeywordArgument("key", Leaf("value"))], Leaf("fn_name")),
            len("fn_name(key=...)") - 1,
            FunctionCall([EllipsisLeaf()], Leaf("fn_name")),
        ),
    ],
)
def test_limit_str_length(expr_tree, max_len, expected):
    actual = limit_str_length(expr_tree, max_len)
    assert actual == expected, (
        'When limiting the expression "{expr}" to length {max_len}, '
        '"{expected}" is expected, but got "{actual}"'
    ).format(
        expr=str(expr_tree), max_len=max_len, expected=str(expected), actual=str(actual)
    )
