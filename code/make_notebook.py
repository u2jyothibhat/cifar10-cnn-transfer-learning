"""Assemble CIFAR10_CNN_Capstone.ipynb from the ordered cell-builder modules."""
import nbformat as nbf
import nb_part1, nb_part2, nb_part3, nb_part4, nb_part5, nb_part6   # order matters
from nb_part1 import CELLS

nb = nbf.v4.new_notebook()
nb.cells = [nbf.v4.new_markdown_cell(t) if kind == "md" else nbf.v4.new_code_cell(t)
            for kind, t in CELLS]
nb.metadata = {
    "kernelspec": {"display_name": "Python 3 (capstone)", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.9.6",
                      "mimetype": "text/x-python", "file_extension": ".py",
                      "pygments_lexer": "ipython3", "nbconvert_exporter": "python"},
    "title": "CIFAR-10 Image Classification using Deep Learning — Capstone",
}
nbf.write(nb, "CIFAR10_CNN_Capstone.ipynb")
md = sum(1 for k, _ in CELLS if k == "md"); cd = len(CELLS) - md
print(f"CIFAR10_CNN_Capstone.ipynb written: {len(CELLS)} cells ({md} markdown, {cd} code)")
