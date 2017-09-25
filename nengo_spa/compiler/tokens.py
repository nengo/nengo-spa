"""A thin wrapper around the `tokenize` module.

Used to provide a common interface across Python versions.
"""

# pylint: disable=unused-import

import sys


from token import tok_name
from tokenize import (
    untokenize,
    TokenError,
    ERRORTOKEN,
    ENDMARKER,
    NAME,
    NUMBER,
    STRING,
    NEWLINE,
    INDENT,
    DEDENT,
    OP,
    COMMENT,
    NL
)

if sys.version_info[0] < 3:
    from tokenize import generate_tokens as tokenize

    ENCODING = None
    TokenInfo = tuple

    def is_token_info(tk):
        return isinstance(tk, TokenInfo) and len(tk) == 5
else:
    from tokenize import ENCODING, TokenInfo, tokenize

    def is_token_info(tk):
        return isinstance(tk, TokenInfo)
