## actx_dgfem_suite

Run DG-FEM benchmarks for different
[arraycontext](https://github.com/inducer/arraycontext/) implementations. These
benchmarks generate `arraycontext`-based Python code for the operators' RHS to
avoid time spent in the setup phases of the driver (mesh gen, computing geometry
factors, etc.) and evaluate the performance of the generated code.


## Installation

```console
$ git clone https://github.com/kaushikcfd/actx_dgfem_suite && cd actx_dgfem_suite
$ conda env create -f .test-conda-env.yml
$ conda activate actx-dgfem-env
$ pip install -e .
```

Supporting input data is managed with [DVC](https://dvc.org) and hosted on
[DagsHub](https://dagshub.com/kaushikcfd/actx_dgfem_bench). Pull the data with:

```console
$ dvc pull
```

To avoid network traffic and for better security practices, see
`actx_dgfem_suite/suite_generators.py` to generate the data locally instead


## HOWTO: Run the timing suite
```console
$ python -O -m actx_dgfem_suite.run --equations "wave,euler,cns_without_chem" \
                                    --degrees "1,2,3,4" \
                                    --dims "3" \
                                    --actxs "pyopencl,jax:jit,pytato:dgfem_opt"
GFLOPS/s for 3D-wave:
╒════╤══════════╤═════════╤══════════════════╤══════════╕
│    │ pyopencl │ jax:jit │ pytato:dgfem_opt │ Roofline │
├────┼──────────┼─────────┼──────────────────┼──────────┤
│ P1 │ 5.3      │ 33.1    │ 112.1            │ 441.1    │
├────┼──────────┼─────────┼──────────────────┼──────────┤
│ P2 │ 5.1      │ 86.7    │ 224.5            │ 707.9    │
├────┼──────────┼─────────┼──────────────────┼──────────┤
│ P3 │ 5.5      │ 193.8   │ 428.5            │ 1163.9   │
├────┼──────────┼─────────┼──────────────────┼──────────┤
│ P4 │ 4.2      │ 377.2   │ 645.9            │ 1874.8   │
╘════╧══════════╧═════════╧══════════════════╧══════════╛

GFLOPS/s for 3D-euler:
╒════╤══════════╤═════════╤══════════════════╤══════════╕
│    │ pyopencl │ jax:jit │ pytato:dgfem_opt │ Roofline │
├────┼──────────┼─────────┼──────────────────┼──────────┤
│ P1 │ 23.0     │ 44.9    │ 225.8            │ 770.0    │
├────┼──────────┼─────────┼──────────────────┼──────────┤
│ P2 │ 25.8     │ 81.1    │ 281.6            │ 971.7    │
├────┼──────────┼─────────┼──────────────────┼──────────┤
│ P3 │ 31.7     │ 136.0   │ 355.9            │ 1326.4   │
├────┼──────────┼─────────┼──────────────────┼──────────┤
│ P4 │ 40.9     │ 217.7   │ 485.8            │ 1897.8   │
╘════╧══════════╧═════════╧══════════════════╧══════════╛

GFLOPS/s for 3D-cns_without_chem:
╒════╤══════════╤═════════╤══════════════════╤══════════╕
│    │ pyopencl │ jax:jit │ pytato:dgfem_opt │ Roofline │
├────┼──────────┼─────────┼──────────────────┼──────────┤
│ P1 │ 8.2      │ 35.1    │ 147.8            │ 510.3    │
├────┼──────────┼─────────┼──────────────────┼──────────┤
│ P2 │ 7.7      │ 80.5    │ 282.2            │ 722.0    │
├────┼──────────┼─────────┼──────────────────┼──────────┤
│ P3 │ 7.8      │ 160.6   │ 373.1            │ 1087.5   │
├────┼──────────┼─────────┼──────────────────┼──────────┤
│ P4 │ 5.6      │ 283.3   │ 466.4            │ 1650.2   │
╘════╧══════════╧═════════╧══════════════════╧══════════╛
```


## HOWTO: Add new arraycontext implementations to compare

Update `run.py` to include your arraycontext type. We are happy to accept PRs
for your `arraycontext` type.

## HOWTO: Add new tests to the suite

Implement the new operators in `actx_dgfem_suite/suite/`, invoke

```console
$ python -m suite_generators -h
usage:  python -m actx_dgfem_suite.suite_generators [-h] --equations E --dims D --degrees G

Generate DG-FEM benchmarks suite

optional arguments:
  -h, --help     show this help message and exit
  --equations E  comma separated strings representing which equations to time (for ex. 'wave,euler,cns_with_chem,cns_without_chem')
  --dims D       comma separated integers representing the topological dimensions to run the problems on (for ex. 2,3 to run 2D and 3D versions of the
                 problem)
  --degrees G    comma separated integers representing the polynomial degree of the discretizing function spaces to run the problems on (for ex. 1,2,3 to run
                 using P1,P2,P3 function spaces)
```

After running `suite_generators` and producing new pkl/npz files:

```console
# Re-add changed files (DVC detects what changed)
$ find actx_dgfem_suite/suite -name "*.pkl" -o -name "*.npz" | sort | xargs dvc add

# Push new blobs to DagsHub
$ dvc push

# Commit updated pointer files
$ git add actx_dgfem_suite/suite/
$ git commit -m "Update benchmark data for <reason>"
$ git push
```

To prune old blobs from DagsHub remote (avoiding the LFS-style accumulation problem):
```console
$ dvc gc --cloud -w   # keeps only what the current workspace references
```

## LICENSE

MIT (see LICENSE.txt)
