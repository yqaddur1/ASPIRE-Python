# ASPIRE – Algorithms for Single Particle Reconstruction with Max Filtering

> 🚧 **Work in Progress**  
> This is an active research and development project. Features and results are evolving, and parts of the codebase may change frequently.

**ASPIRE** is an open-source software package for processing single-particle cryo-EM data to determine the three-dimensional structures of biological macromolecules. The package includes advanced algorithms grounded in rigorous mathematics, statistics, and machine learning.

ASPIRE offers unique and improved solutions to major computational challenges in the cryo-EM processing pipeline, including:

- 3D *ab initio* modeling  
- 2D class averaging  
- Automatic particle picking  
- 3D heterogeneity analysis

🔗 For more information about ASPIRE and its algorithms, visit the [ASPIRE Project website](http://spr.math.princeton.edu/).  
📘 Full documentation and tutorials are available [here](https://computationalcryoem.github.io/ASPIRE-Python).

---

## My Contribution

In this project, I am modifying the **bispectrum embedding** [1] used in ASPIRE by replacing it with **max filter banks** [2,3,4]. I am also developing methods to **train or select these filters** to improve denoising performance on cryo-EM images.

Development is primarily taking place on the `develop` branch.

---

## Installation Instructions

To set up ASPIRE using Anaconda (recommended):

```bash
cd /path/to/git/clone/folder

# Create the conda environment and install base dependencies
conda env create -f environment.yml --name aspire_dev

# Activate the environment
conda activate aspire_dev

# Install the ASPIRE package in editable mode with developer tools
pip install -e ".[dev]"
```

If you prefer not to use Anaconda, you can also install dependencies via `pip` with Python ≥ 3.7. See the [documentation](https://computationalcryoem.github.io/ASPIRE-Python) for details.

---

## Run Tests

To verify that everything is working, run:

```bash
cd /path/to/git/clone/folder
pytest
```

---

## References

[1] Z. Zhao, A. Singer, *Rotationally invariant image representation for viewing direction classification in cryo-EM*, J. Struct. Biol. **186**.1 (2014) 153–166.  

[2] J. Cahill, J. W. Iverson, D. G. Mixon, D. Packer, *Group-invariant max filtering*, Found. Comput. Math. (2024) 1–38.  

[3] D. G. Mixon, **Y. Qaddura**, *Injectivity, stability and positive definiteness of max filtering*, Constr. Approx. (2025).  

[4] **Y. Qaddura**, *A max filtering local stability theorem with application to weighted phase retrieval and cryo-EM*, arXiv:2403.14042.
