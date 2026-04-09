import importlib


def test_dependencies_installed():
    dependencies = ["pylint"]
    for lib in dependencies:
        assert importlib.util.find_spec(lib) is not None, f"{lib} is not installed!"
