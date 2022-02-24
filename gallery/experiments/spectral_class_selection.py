"""
Spectral Clustering Class Selection - Experiment Stub
=====================================================

This notebook shows how to yield Nearest Neighbors
information from RIRClass2D.

First, construct a small data set and yield the complete NN graph.
Second, ???.
Third, profit.
"""

# %%
# Imports
# -------
# First import some of the usual suspects.
# In addition, import some classes from
# the ASPIRE package that will be used throughout this experiment.

import logging

import matplotlib.pyplot as plt
import numpy as np

from aspire.basis import FFBBasis2D, FFBBasis3D
from aspire.classification import BFSReddyChatterjiAverager2D, RIRClass2D
from aspire.source import Simulation
from aspire.volume import Volume

logger = logging.getLogger(__name__)

# %%
# Complete NN Graph Sanity Check
# ------------------------------
# First we'll compute a complete graph and check the eigenvalues.
# This should check for a few obvious problems.

# %%
# Parameters
# ^^^^^^^^^^
# Configuration to compute a small (but complete) NN graph.

img_size = 32  # Downsample the volume to a desired resolution
num_imgs = 1000  # How many images in our source.
n_classes = num_imgs  # How many class averages to compute.
n_nbor = 2 * num_imgs  # How many neighbors to stack


# %%
# Simulation Data
# ^^^^^^^^^^^^^^^
# Start with a fairly hi-res volume available from EMPIAR/EMDB.
# https://www.ebi.ac.uk/emdb/EMD-2660
# https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-2660/map/emd_2660.map.gz
og_v = Volume.load("emd_2660.map", dtype=np.float64)
logger.info("Original volume map data" f" shape: {og_v.shape} dtype:{og_v.dtype}")

logger.info(f"Downsampling to {(img_size,)*3}")
v = og_v.downsample(img_size)
L = v.resolution

# Create the Simulation
src = Simulation(
    L=v.resolution,
    n=num_imgs,
    vols=v,
)

# Cache to memory for some speedup
src.cache()

# %%
# Class Averaging
# ^^^^^^^^^^^^^^^
#
# Now perform classification and averaging for each class.

logger.info("Begin Class Averaging")

rir = RIRClass2D(
    src,  # Source used for classification
    fspca_components=400,
    bispectrum_components=300,  # Compressed Features after last PCA stage.
    n_nbor=n_nbor,
    n_classes=n_classes,
    large_pca_implementation="legacy",
    nn_implementation="sklearn",
    bispectrum_implementation="legacy",
)

# The Adjacency List can be retrieved once Nearest Neighbors
#   has run in `classify`, which is before averaging.
# This allows experimenting with overloading `rir.select_classes`
#   prior to calling the averager.
rir.classify()


def random_class_selector(self, classes, reflections):

    from np.random import default_rng

    # Random without replacement
    rng = default_rng()
    selection = rng.choice(len(classes), size=self.n_classes, replace=False)

    return classes[selection], reflections[selection]


def custom_class_selector(self, classes, reflections):

    adj_list, wts_list = self.get_nn_graph()

    # Insert Smart Algo Here
    # ... something something ...
    # selections = ...

    return classes[selection], reflections[selection]


# Overload the selector.
# For proofs of concept, this should be adequate.
# If this is a fruitful area, simply make a smart selector class
#   and assign/instantiate during RIRClass2D init.
rir.select_classes = custom_class_selector

# Continue on with averaging...
# avgs = rir.averages(rir.classes, rir.reflections, rir.distances)


# %%
# Laplacian Helper Scratch
# ------------------------
#

# First lets make a util to create a Laplacian from an Adjacency list
def make_laplacian(
    adj_list, wts=None, num_vertices=None, force_symmetric=False, dtype=np.float64
):
    """
    Given an adj_list construct a basic Laplacian, optionally using `wts`.

    `num_vertices` can be used to mask off the vertex set.
    In this case, that might control whether we include reflections in our data.
    """

    if num_vertices is None:
        num_vertices = len(adj_list)

    if wts is None:
        # Make a 2d thing, using one 1d object
        m = np.max(adj_list) + 1
        wts = [np.full(m, -1)] * m

    # adjacency entries (negated)
    L = np.zeros((num_vertices, num_vertices), dtype=dtype)
    # degree entries
    D = np.zeros(num_vertices, dtype=int)
    for r, row in enumerate(adj_list):
        vi = row[0]
        if vi >= num_vertices:
            continue
        D[vi] += 1
        for c, vj in enumerate(row[1:]):
            if vj >= num_vertices:
                continue
            # weight is inversely proportional to distance, note negated
            # print(num_vertices, vi,vj, r, c+1)
            L[vi][vj] -= wts[r][c + 1]
            if force_symmetric:
                L[vj][vi] -= wts[r][c + 1]
                D[vj] += 1

    # L = D - A
    # In this code we have L = D + L where L = -A currently.
    # We do this to operate in place.
    np.fill_diagonal(L, D)

    return L


# Let's take a look at the spectrum to see what we're working with.
adj_list, dist_list = rir.get_nn_graph()

# First the unweighted Laplacian.
# We'll symmetrize it, at least initially...
#   otherwise with reflections this might get too weird.
unweighted_L = make_laplacian(adj_list, num_vertices=2 * num_imgs, force_symmetric=True)

# What does the spectrum look like?
lamb = np.linalg.eigvalsh(unweighted_L)
# Plot the spectrum, descending
plt.semilogy(lamb[::-1])
plt.show()

# Cool, let's ignore the reflections
unrefl_unweighted_L = make_laplacian(
    adj_list, num_vertices=num_imgs, force_symmetric=True
)

# What does the spectrum look like?
unrefl_lamb = np.linalg.eigvalsh(unrefl_unweighted_L)
# Plot the spectrum, descending
plt.semilogy(unrefl_lamb[::-1])
plt.show()


# Now let's look at the directed weighted case,
# when weight is inversely proportional to distance.
L = make_laplacian(
    adj_list, wts=1.0 / rir.distances, num_vertices=num_imgs, force_symmetric=False
)

# What does this spectrum look like? (Note, we don't use `eigvalsh` here... and we should them sort)
lamb = np.sort(np.linalg.eigvals(L))
# Plot the spectrum, descending
plt.semilogy(lamb[::-1])
plt.show()

## From here look at various normalizations and kernels etc
##   until satisfied with complete graph.


# %%
# Truncated NN Graph
# ------------------
# Then repeat looking at actual (truncated) NN graph.
# The parameters will be modified to
# explore a larger number of images,
# return all images' classes,
# but with fewer neighbors in each class.

# %%
# Parameters
# ^^^^^^^^^^
# Configuration to compute a small experiment.

img_size = 32  # Downsample the volume to a desired resolution
num_imgs = 4000  # How many images in our source.
n_classes = num_imgs  # How many class averages to compute.
n_nbor = 200  # How many neighbors to stack

# %%
# Simulation Data
# ^^^^^^^^^^^^^^^
#

# Create the Simulation
src = Simulation(
    L=v.resolution,
    n=num_imgs,
    vols=v,
)

# Cache to memory for some speedup
src.cache()

# %%
# Class Averaging
# ^^^^^^^^^^^^^^^
# Now perform classification and averaging for each class.

logger.info("Begin Class Averaging")

rir = RIRClass2D(
    src,  # Source used for classification
    fspca_components=400,
    bispectrum_components=300,  # Compressed Features after last PCA stage.
    n_nbor=n_nbor,
    n_classes=n_classes,
    large_pca_implementation="legacy",
    nn_implementation="sklearn",
    bispectrum_implementation="legacy",
)


# Compute and populate the `rir` object with the class data
# needed to generate and inspect the spectrum.
rir.classify()

# %%
# Scratch
# -------
#

# Let's take a look at the spectrum to see what we're working with.
adj_list, distances = rir.get_nn_graph()

# Let's take a look at the spectrum to see what we're working with.
adj_list, dist_list = rir.get_nn_graph()

# First the unweighted Laplacian.
# We'll symmetrize it, at least initially...
#   otherwise with reflections this might get too weird.
unweighted_L = make_laplacian(adj_list, num_vertices=2 * num_imgs, force_symmetric=True)

# What does the spectrum look like?
lamb = np.linalg.eigvalsh(unweighted_L)
# Plot the spectrum, descending
plt.semilogy(lamb[::-1])
plt.show()

# Cool, let's ignore the reflections
unrefl_unweighted_L = make_laplacian(
    adj_list, num_vertices=num_imgs, force_symmetric=True
)

# What does the spectrum look like?
unrefl_lamb = np.linalg.eigvalsh(unrefl_unweighted_L)
# Plot the spectrum, descending
plt.semilogy(unrefl_lamb[::-1])
plt.show()


# Now let's look at the directed weighted case,
# when weight is inversely proportional to distance.
L = make_laplacian(
    adj_list, wts=1.0 / rir.distances, num_vertices=num_imgs, force_symmetric=False
)

# What does this spectrum look like? (Note, we don't use `eigvalsh` here... and we should them sort)
lamb = np.sort(np.linalg.eigvals(L))
# Plot the spectrum, descending
plt.semilogy(lamb[::-1])
plt.show()

# # From here extend look at various normalizations and kernels etc
# #   until satisfied.

# # Good luck!
