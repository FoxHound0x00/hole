# Getting Started

This quick guide helps you install and run **HOLE** in a fresh environment.

## Installation with Poetry

```bash
# Clone the repository
git clone https://github.com/<your-org>/hole.git
cd hole

# (Optional) create a virtual environment
python -m venv .venv
source .venv/bin/activate

# Install runtime dependencies
pip install poetry
poetry install --without dev

# Run the test-suite
poetry run pytest -q
```

## Running an Example

After installation, try the bundled example:

```bash
poetry run python examples/three_blobs_example.py
```

A figure named `three_blobs_example_sankey.png` will be produced in the project
root.

---

*For more in-depth tutorials jump to the [Tutorials](tutorials/README.md) section.* 