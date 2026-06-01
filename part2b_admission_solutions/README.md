# Part 2b - Admission Debugging Exercises

Solutions to the four debugging exercises distributed via
`debugging.ipynb` from the [admission page][cvds]. Each fix is documented in
the notebook next to the corrected code, and also as importable Python
modules so it can be reused outside the notebook.

[cvds]: https://www.cvds-nhlstenden.com/admission/

## Files

| File | Purpose |
| ---- | ------- |
| `solutions.ipynb` | **Executed** Jupyter notebook mirroring the four exercises with the fixes applied. Outputs (text, plots, generated images) are saved with the notebook. |
| `solutions.py`    | Exercises 1, 2 and 3 as plain Python functions. |
| `gan.py`          | Exercise 4 (`Generator`, `Discriminator`, `train_gan`) split out so it is reusable from scripts and easy to test. |
| `tests/test_solutions.py` | Pytest suite that pins down each bug fix. |
| `_build_notebook.py` | Helper that builds `solutions.ipynb` from a structured Python definition (run once when regenerating the notebook). |

## Summary of the four bugs

| # | Function       | Bug | Fix |
| - | -------------- | --- | --- |
| 1 | `id_to_fruit`  | Indexing into a `set`, which is unordered, so the returned fruit is non-deterministic. | Take an ordered sequence; for backwards compatibility sort the set deterministically. |
| 2 | `swap`         | (a) RHS used `coords[:, 1]` twice; (b) NumPy slices are views, so simultaneous assignment corrupted the source before all targets were written. | Build a copy of the original columns and assign back in order. |
| 3 | `plot_data`    | `csv.reader` returns strings, so the stacked array is a string array and matplotlib can't sort it numerically; the axes were also swapped relative to the docstring. | Cast the array to `float`; put precision on the x-axis, recall on the y-axis. |
| 4 | `train_gan`    | (a) Last batch can be smaller than `batch_size`, so labels sized from the constant don't match the discriminator output (`Using a target size ([128, 1]) ... different to the input size ([96, 1])`); (b) the end-of-epoch log used `n == batch_size - 1`, which is the wrong variable. | (a) Size labels from `real_samples.size(0)` (and use `drop_last=True` defensively). (b) Log when `n == len(train_loader) - 1`. |

## Running the tests

```bash
python -m pytest -q
```

Expected: 7 tests pass.

## Re-executing the notebook

```bash
jupyter nbconvert --to notebook --execute --inplace solutions.ipynb
```

> Note: the notebook executes `train_gan(batch_size=64, num_epochs=1)` only,
> just enough to demonstrate that the structural bug is fixed (the original
> code raised on `batch_size=64`). Training the GAN for 50-100 epochs is
> identical in behaviour but takes hours on a CPU, so it is not part of the
> executed notebook.
