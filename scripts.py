import subprocess


def build():
    subprocess.run(["poetry", "run", "isort", "."])
    subprocess.run(["poetry", "run", "black", "."])
    subprocess.run(["poetry", "build"])


def docs_serve():
    subprocess.run(["poetry", "run", "mkdocs", "serve"])


def docs_build():
    subprocess.run(["poetry", "run", "mkdocs", "build"])
