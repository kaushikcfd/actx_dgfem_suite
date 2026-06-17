## actx_dgfem_suite

Run DG-FEM benchmarks for different
[arraycontext](https://github.com/inducer/arraycontext/) implementations. These
benchmarks construct RHS operators for wave, Euler, and Maxwell equations at
runtime (mesh gen, geometry factors, initial conditions) and measure the
sustained FLOP rate of the generated code across arraycontext implementations.


## Installation

```console
$ git clone https://github.com/kaushikcfd/actx_dgfem_suite && cd actx_dgfem_suite
$ conda env create -f .test-conda-env.yml
$ conda activate actx-dgfem-env
$ pip install -e .
```


## HOWTO: Run the timing suite

```console
$ python -O -m actx_dgfem_suite.run --equations "wave,euler,maxwell" \
                                    --degrees "1,2,3,4" \
                                    --dims "3" \
                                    --actxs "pyopencl,jax:jit,pytato:dgfem_opt"
```

Supported equations: `wave`, `euler`, `maxwell`, and their `tiny_` variants
(`tiny_wave`, `tiny_euler`, `tiny_maxwell`) which use a smaller mesh
(~1000 DOFs) suitable for quick correctness checks.


## HOWTO: Add new equations to the suite

1. Add a `get_<equation>_rhs(*, actx, dim, order, ndofs)` function in
   `actx_dgfem_suite/equations/<equation>.py` returning
   `(rhs_callable, args_tuple)`.
2. Register the new equation in `actx_dgfem_suite/rhs_builder.py`.
3. Add FLOP counting support in `actx_dgfem_suite/perf_analysis.py` if desired.


## HOWTO: Add new arraycontext implementations to compare

Update `_NAME_TO_ACTX_CLASS` in `actx_dgfem_suite/run/__main__.py` to include
your arraycontext type. Happy to accept PRs.


## HOWTO: Obtain libparanumal (hand-written baseline) throughput

Use the following values for N=[BOX NX]=[BOX NY]=[BOX NZ] in
`assets/libparanumal/solvers/acoustics/setups/setupTet3D.rc`:

|   | P1 | P2 | P3 | P4 |
|---|----|----|----|----|
| N | 50 | 37 | 30 | 25 |

## LICENSE

MIT (see LICENSE.txt)
