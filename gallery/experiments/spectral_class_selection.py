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
# Parameters
# ---------------
# Configuration to compute a small (but complete) NN graph.

img_size = 32  # Downsample the volume to a desired resolution
num_imgs = 1000  # How many images in our source.
n_classes = num_imgs  # How many class averages to compute.
n_nbor = 2 * num_imgs  # How many neighbors to stack


# %%
# Simulation Data
# ---------------
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
# ----------------------
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


def custom_class_selector(self, classes, reflections):

    adj_list, wts_list = self.get_nn_graph()

    # Insert Smart Algo Here
    # ... something something ...
    # selections =

    return classes[selection], reflections[selection]


# Overload the selector.
# For proofs of concept, this should be adequate.
# If this is a fruitful area, simply make a smart selector class
#   and assign/instantiate during RIRClass2D init.
rir.select_classes = custom_class_selector

# Continue on with averaging...
# avgs = rir.averages(rir.classes, rir.reflections, rir.distances)


# %%
# Scratch
# -------
#

# Let's take a look at the spectrum to see what we're working with.
adj_list, wts_list = rir.get_nn_graph()

# First the unweighted Laplacian.
# We'll symmetrize it, at least initially...
#   otherwise with reflections this might get too weird.
unweighted_L = np.zeros((2 * num_imgs, 2 * num_imgs), np.float64)
for row in adj_list:
    vi = row[0]
    for vj in row[1:]:
        unweighted_L[vi][vj] = -1
        # Sym
        unweighted_L[vj][vi] = -1

# Compute the degree entries (for complete graph this is a lot easier than I've made it ;) )
D = np.sum(unweighted_L, axis=1)
# L = D-A
np.fill_diagonal(unweighted_L, -D)

# What does the spectrum look like?
lamb = np.linalg.eigvalsh(unweighted_L)
# Plot the spectrum, descending
plt.semilogy(lamb[::-1])
plt.show()

# Cool, let's ignore reflections (we're missing half the set anyway, well, sort of)
unrefl_unweighted_L = unweighted_L[:num_imgs, :num_imgs]
# Reset diag to 0
np.fill_diagonal(unrefl_unweighted_L, 0)
# Compute the degree entries
D = np.sum(unrefl_unweighted_L, axis=1)
# L = D-A
np.fill_diagonal(unrefl_unweighted_L, -D)

# What does the spectrum look like?
unrefl_lamb = np.linalg.eigvalsh(unrefl_unweighted_L)
# Plot the spectrum, descending
plt.semilogy(unrefl_lamb[::-1])
plt.show()


# Now let's look at the weighted case.
L = np.zeros((2 * num_imgs, 2 * num_imgs), np.float64)
for r, row in enumerate(adj_list):
    vi = row[0]
    for c, vj in enumerate(row[1:]):
        # weight is inversely proportional to distance
        L[vi][vj] = 1 / wts_list[r][c + 1]
# Compute the degree entries
D = np.sum(unweighted_L, axis=1)
# L = D-A
np.fill_diagonal(L, -D)

# What does this spectrum look like?
lamb = np.linalg.eigvalsh(L)
# Plot the spectrum, descending
plt.semilogy(lamb[::-1])
plt.show()

## From here look at various normalizations and kernels etc
##   until satisfied with complete graph.
## Then repeat looking at actual (truncated) NN graph ...
## Good luck!
