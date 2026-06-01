# CVDS NHL Stenden - Admission Submission

This repository contains my submission for parts **2a** (Personal Programming
Code Example) and **2b** (Solutions to the Admission Debugging Assignments)
of the [CVDS NHL Stenden][cvds] admission procedure.

[cvds]: https://www.cvds-nhlstenden.com/admission/

## Repository layout

```
.
├── part2a_personal_code/        # Part 2a - personal algorithm code samples
│   ├── algorithms/
│   │   ├── astar.py             # *** Reviewer focus #1: A* path search
│   │   └── kmeans.py            # *** Reviewer focus #2: K-Means from scratch
│   ├── tests/                   # pytest unit tests for both algorithms
│   └── README.md
│
└── part2b_admission_solutions/  # Part 2b - solutions to the 4 debugging tasks
    ├── solutions.ipynb          # Executed Jupyter notebook (outputs saved)
    ├── solutions.py             # Exercises 1, 2, 3 as importable functions
    ├── gan.py                   # Exercise 4 (GAN training) - split for clarity
    ├── tests/                   # pytest unit tests pinning each bug fix
    └── README.md
```

## Where to look

For Part 2a the two files worth reviewing are:

- `part2a_personal_code/algorithms/astar.py`
- `part2a_personal_code/algorithms/kmeans.py`

Both are self-contained, fully type-annotated, follow PEP 8, and have a
matching test file under `part2a_personal_code/tests/`.

For Part 2b the entry point is `part2b_admission_solutions/solutions.ipynb`,
which mirrors the structure of the original `debugging.ipynb`, names each
bug, and shows the fix in context. The same fixes are also packaged as
importable modules (`solutions.py`, `gan.py`) with unit tests.

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Part 2a tests
cd part2a_personal_code && python -m pytest -q

# Part 2b tests
cd ../part2b_admission_solutions && python -m pytest -q

# Open / re-execute the notebook
jupyter notebook solutions.ipynb
```
