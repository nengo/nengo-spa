from io import BytesIO

import pytest

from nengo_spa.compiler import tokens
from nengo_spa.compiler.grammar import parse
from nengo_spa.compiler.parsing import MatchToken


def bare_tokens(expr):
    return [
        MatchToken.from_token(tk)
        for tk in tokens.tokenize(BytesIO(expr.encode()).readline)
        if tk[0] != tokens.ENCODING and tk[0] != tokens.ENDMARKER]


def effect_tree(source, target):
    return (
        'effect',
        [('expr', bare_tokens(source))] +
        bare_tokens('->') +
        [('sink', bare_tokens(target))]
    )


@pytest.mark.parametrize('code, parse_tree', [
    ('a -> b', [effect_tree('a', 'b')]),
    ('a.x -> b.y', [effect_tree('a.x', 'b.y')]),
    ('a -> b; c -> d', [
        effect_tree('a', 'b'),
        MatchToken(string=';'),
        effect_tree('c', 'd'),
    ]),
    ('a -> b\nc -> d', [
        effect_tree('a', 'b'),
        MatchToken(type=tokens.NEWLINE),
        effect_tree('c', 'd'),
    ]),
    ('fn(";") -> a; c -> d  # ; ignore', [
        effect_tree('fn(";")', 'a'),
        MatchToken(string=';'),
        effect_tree('c', 'd'),
        MatchToken(type=tokens.COMMENT, string='# ; ignore')
    ]),
    ('''
    fn("""
       multiline madness
      """  # and a weirdly placed comment
    ) -> a''', [
        MatchToken(type=tokens.NL),
        MatchToken(type=tokens.INDENT),
        effect_tree('''fn("""
       multiline madness
      """  # and a weirdly placed comment
    )''', 'a'),
        MatchToken(type=tokens.DEDENT),
    ]),
    ('''
        a -> b
        always:
            c -> d; e -> f
            g -> h
        i -> j''', [
        MatchToken(type=tokens.NL),
        MatchToken(type=tokens.INDENT),
        effect_tree('a', 'b'),
        MatchToken(type=tokens.NEWLINE),
        ('always', [
            MatchToken(type=tokens.NAME, string='always'),
            MatchToken(string=':'),
            MatchToken(type=tokens.NEWLINE),
            MatchToken(type=tokens.INDENT),
            effect_tree('c', 'd'),
            MatchToken(string=';'),
            effect_tree('e', 'f'),
            MatchToken(type=tokens.NEWLINE),
            effect_tree('g', 'h'),
            MatchToken(type=tokens.NEWLINE),
            MatchToken(type=tokens.DEDENT),
        ]),
        effect_tree('i', 'j'),
        MatchToken(type=tokens.DEDENT),
    ]),
    ('''
        always as 'name':
            pass''', [
        MatchToken(type=tokens.NL),
        MatchToken(type=tokens.INDENT),
        ('always', [
            MatchToken(type=tokens.NAME, string='always'),
            ('as', [
                MatchToken(type=tokens.NAME, string='as'),
                MatchToken(type=tokens.STRING, string="'name'"),
            ]),
            MatchToken(string=':'),
            MatchToken(type=tokens.NEWLINE),
            MatchToken(type=tokens.INDENT),
            MatchToken(type=tokens.NAME, string='pass'),
            MatchToken(type=tokens.DEDENT),
        ]),
        MatchToken(type=tokens.DEDENT),
    ]),
    ("always as 'name': pass", [
        ('always', [
            MatchToken(type=tokens.NAME, string='always'),
            ('as', [
                MatchToken(type=tokens.NAME, string='as'),
                MatchToken(type=tokens.STRING, string="'name'"),
            ]),
            MatchToken(string=':'),
            MatchToken(type=tokens.NAME, string='pass'),
        ]),
    ]),
    ('''
        ifmax dot(a, b) as 'rule1':
            a -> b
        elifmax dot(c, FOO) + 0.2:
            c -> d
        elifmax 0.3 as 'rule3':
            e -> f''', [
        MatchToken(type=tokens.NL),
        MatchToken(type=tokens.INDENT),
        ('max_action', [
            ('ifmax', [
                MatchToken(type=tokens.NAME, string='ifmax'),
                ('utility', [('expr', bare_tokens('dot(a, b)'))]),
                ('as', [
                    MatchToken(type=tokens.NAME, string='as'),
                    MatchToken(type=tokens.STRING, string="'rule1'"),
                ]),
                MatchToken(string=':'),
                MatchToken(type=tokens.NEWLINE),
                MatchToken(type=tokens.INDENT),
                effect_tree('a', 'b'),
                MatchToken(type=tokens.NEWLINE),
                MatchToken(type=tokens.DEDENT),
            ]),
            ('elifmax', [
                MatchToken(type=tokens.NAME, string='elifmax'),
                ('utility', [('expr', bare_tokens('dot(c, FOO) + 0.2'))]),
                MatchToken(string=':'),
                MatchToken(type=tokens.NEWLINE),
                MatchToken(type=tokens.INDENT),
                effect_tree('c', 'd'),
                MatchToken(type=tokens.NEWLINE),
                MatchToken(type=tokens.DEDENT),
            ]),
            ('elifmax', [
                MatchToken(type=tokens.NAME, string='elifmax'),
                ('utility', [('expr', bare_tokens('0.3'))]),
                ('as', [
                    MatchToken(type=tokens.NAME, string='as'),
                    MatchToken(type=tokens.STRING, string="'rule3'"),
                ]),
                MatchToken(string=':'),
                MatchToken(type=tokens.NEWLINE),
                MatchToken(type=tokens.INDENT),
                effect_tree('e', 'f'),
                MatchToken(type=tokens.DEDENT),
            ]),
        ]),
        MatchToken(type=tokens.DEDENT),
    ]),
    ('''
        ifmax dot(a, b) as 'rule1': a -> b
        elifmax dot(c, FOO) + 0.2: c -> d
        elifmax 0.3 as 'rule3': e -> f''', [
        MatchToken(type=tokens.NL),
        MatchToken(type=tokens.INDENT),
        ('max_action', [
            ('ifmax', [
                MatchToken(type=tokens.NAME, string='ifmax'),
                ('utility', [('expr', bare_tokens('dot(a, b)'))]),
                ('as', [
                    MatchToken(type=tokens.NAME, string='as'),
                    MatchToken(type=tokens.STRING, string="'rule1'"),
                ]),
                MatchToken(string=':'),
                effect_tree('a', 'b'),
                MatchToken(type=tokens.NEWLINE),
            ]),
            ('elifmax', [
                MatchToken(type=tokens.NAME, string='elifmax'),
                ('utility', [('expr', bare_tokens('dot(c, FOO) + 0.2'))]),
                MatchToken(string=':'),
                effect_tree('c', 'd'),
                MatchToken(type=tokens.NEWLINE),
            ]),
            ('elifmax', [
                MatchToken(type=tokens.NAME, string='elifmax'),
                ('utility', [('expr', bare_tokens('0.3'))]),
                ('as', [
                    MatchToken(type=tokens.NAME, string='as'),
                    MatchToken(type=tokens.STRING, string="'rule3'"),
                ]),
                MatchToken(string=':'),
                effect_tree('e', 'f'),
            ]),
        ]),
        MatchToken(type=tokens.DEDENT),
    ]),
    ('''
        ifmax fn(**{'arg': 0}):
            pass''', [
        MatchToken(type=tokens.NL),
        MatchToken(type=tokens.INDENT),
        ('max_action', [
            ('ifmax', [
                MatchToken(type=tokens.NAME, string='ifmax'),
                ('utility', [('expr', bare_tokens("fn(**{'arg': 0})"))]),
                MatchToken(string=':'),
                MatchToken(type=tokens.NEWLINE),
                MatchToken(type=tokens.INDENT),
                MatchToken(type=tokens.NAME, string='pass'),
                MatchToken(type=tokens.DEDENT),
            ]),
        ]),
        MatchToken(type=tokens.DEDENT),
    ]),
    ('''
        # comment
        always:
            pass''', [  # check leading comments
        MatchToken(type=tokens.NL),
        MatchToken(type=tokens.COMMENT),
        MatchToken(type=tokens.NL),
        MatchToken(type=tokens.INDENT),
        ('always', [
            MatchToken(type=tokens.NAME, string='always'),
            MatchToken(string=':'),
            MatchToken(type=tokens.NEWLINE),
            MatchToken(type=tokens.INDENT),
            MatchToken(type=tokens.NAME, string='pass'),
            MatchToken(type=tokens.DEDENT),
        ]),
        MatchToken(type=tokens.DEDENT),
    ]),
])
def test_grammar(code, parse_tree):
    actual = parse(code)
    if actual[0] == MatchToken(type=tokens.ENCODING):
        actual = actual[1:]
    assert actual == parse_tree + [MatchToken(type=tokens.ENDMARKER)]
