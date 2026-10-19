import pytest

from hmas.sandbox import DockerSandbox

sb = DockerSandbox(timeout=20)
pytestmark = pytest.mark.skipif(not (sb.available() and sb.image_exists()),
                                reason="Docker or sandbox image not available")


def test_runs_code():
    r = sb.run_python("print(6 * 7)")
    assert r.ok and r.stdout.strip() == "42"


def test_reports_errors():
    r = sb.run_python("raise ValueError('boom')")
    assert r.returncode != 0 and "ValueError" in r.stderr


def test_no_network():
    r = sb.run_python("import urllib.request\nurllib.request.urlopen('https://pypi.org', timeout=3)")
    assert r.returncode != 0


def test_read_only_filesystem():
    r = sb.run_python("open('/work/x.txt', 'w').write('x')")
    assert r.returncode != 0


def test_timeout():
    r = sb.run_python("while True: pass", timeout=3)
    assert r.timed_out
