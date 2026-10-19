from hmas.codeutil import extract_code


def test_longest_block_wins():
    text = "x\n```python\na = 1\n```\nand\n```python\ndef f():\n    return 2\n```\n"
    assert extract_code(text) == "def f():\n    return 2\n"


def test_unterminated_block():
    assert extract_code("here\n```python\ndef f():\n    pass") == "def f():\n    pass"


def test_plain_text_passthrough():
    assert extract_code("def f(): pass") == "def f(): pass"
