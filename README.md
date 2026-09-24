# Nonnegative Neural Networks Derived from Chemical Reaction Network Steady States

Code and experiments for **Nonnegative Neural Networks Derived from Chemical Reaction Network Steady States**.

This repository contains the implementation of the proposed nonnegative neural networks, the experiments reported in the paper and supplementary material, and the PySB validation of the selected chemical reaction network (CRN) implementations.

## Environment

A validated Linux x86-64 Conda environment is provided in `environment-linux-64.txt`.

Create and activate the environment with:

```bash
conda create -n crn-paper --file environment-linux-64.txt -y
conda activate crn-paper
```

The environment includes the dependencies required for neural-network training and CRN simulation, including PyTorch, PySB, BioNetGen, and Perl.

## Reproduce the experiments

Run the complete reproduction workflow with:

```bash
bash run_full_linux.sh
```

The script:

1. downloads and prepares the required datasets;
2. runs the formal and supplementary experiment grids reported in the paper;
3. locates the fixed ST003390 and Chebyshev \(m=3\) checkpoints used for CRN validation;
4. runs the 20 PySB validation cases reported in the paper.

The complete training grid contains 425 runs.

Interrupted experiment runs can be resumed by running the same command again:

```bash
bash run_full_linux.sh
```

Generated datasets and experiment outputs are written locally during execution. Experimental results are stored under:

```text
runs/
```

## Repository structure

```text
.
├── environment-linux-64.txt
├── run_full_linux.sh
├── src/
└── experiments/
```

`src/` contains the model implementation and supporting code.

`experiments/` contains the experiment protocols, data-preparation code, and CRN validation implementation.
