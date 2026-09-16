import os

ROOT = os.getcwd()

EXCLUDE = {"venv", "__pycache__", ".git"}

def list_tree(path, depth=1, prefix=""):
    if depth < 0:
        return
    for item in sorted(os.listdir(path)):
        if item in EXCLUDE:
            continue
        full = os.path.join(path, item)
        print(prefix + item)
        if os.path.isdir(full):
            list_tree(full, depth - 1, prefix + "    ")

list_tree(ROOT, depth=1)
