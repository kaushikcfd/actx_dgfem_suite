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

Supporting input data is managed with [DVC](https://dvc.org). Pull the data with:

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
```


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

# Push new blobs to S3
$ dvc push

# Commit updated pointer files
$ git add actx_dgfem_suite/suite/
$ git commit -m "Update benchmark data for <reason>"
$ git push
```

To prune old blobs from DVC remote (avoiding the LFS-style accumulation problem):
```console
$ dvc gc --cloud -w   # keeps only what the current workspace references
```

## HOWTO: Add new arraycontext implementations to compare

Update `run.py` to include your arraycontext type. Happy to accept PRs for your
`arraycontext` type.

## HOWTO: Obtain libparanumal (hand-written baseline) throughput

Use the following values for N=[BOX NX]=[BOX NY]=[BOX NZ] fields in
`assets/libparanumal/solvers/acoustics/setups/setupTet3D.rc`:

|   | P1 | P2 | P3 | P4 |
|---|----|----|----|----|
| N | 50 | 37 | 30 | 25 |

## LICENSE

MIT (see LICENSE.txt)
