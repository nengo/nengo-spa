from io import BytesIO
from nengo_spa.compiler.tokens import is_token_info, tokenize


def test_is_token_info():
    token = next(tokenize(BytesIO('foo'.encode()).readline))
    assert is_token_info(token)
    assert not is_token_info(('nodename', []))
