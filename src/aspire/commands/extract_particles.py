import glob
import logging
import os

import click
from click import UsageError

from aspire.source.coordinates import CentersCoordinateSource, EmanCoordinateSource

logger = logging.getLogger(__name__)


@click.command()
@click.option(
    "--mrc_paths",
    required=True,
    help="Text file or glob expression containing paths to micrograph files",
)
@click.option(
    "--coord_paths",
    required=True,
    help="Text file or glob expression containing paths to coordinate files",
)
@click.option(
    "--starfile_out",
    required=True,
    help="Path to starfile of the particle stack to be created",
)
@click.option(
    "--particle_size",
    default=0,
    help="Desired box size (in pixels) of particles to be extracted",
)
@click.option(
    "--centers",
    is_flag=True,
    help="Set this flag if coordinate files contain (X,Y) particle centers",
)
@click.option(
    "--batch_size", default=512, help="Batch size to load images from .mrc files"
)
@click.option(
    "--save_mode",
    default="single",
    help="Option to save MRC file. If not single, saved to multiple files by batch size",
)
@click.option(
    "--overwrite", is_flag=True, help="Overwrite output if it already exists?"
)
def extract_particles(
    mrc_paths,
    coord_paths,
    starfile_out,
    particle_size,
    centers,
    batch_size,
    save_mode,
    overwrite,
):
    """
    Given a dataset of full micrographs and corresponding coordinate files
    containing the locations of picked particles in the .mrc, extract the
    particles into one or more .mrcs stacks and generate a STAR file.

    Example usage:

    aspire extract-particles --mrc_paths=my/data/*.mrc --coord_paths=my/data/coords/*.coord --starfile_out=my_dataset_stack.star --particle_size=256 --centers
    """

    # mrc_paths and coord_paths can be either paths to text files
    # listing the micrograph and coordinate file paths, or glob-type
    # expressions
    # first try interpreting them as files
    if os.path.exists(mrc_paths) and os.path.exists(coord_paths):
        with open(mrc_paths) as _mrc:
            mrc_files = _mrc.readlines()
        with open(coord_paths) as _coords:
            coord_files = _coords.readlines()
    elif glob.glob(mrc_paths) and glob.glob(coord_paths):
        mrc_files = sorted(glob.glob(mrc_paths))
        coord_files = sorted(glob.glob(coord_paths))
    else:
        raise UsageError(
            "--mrc_paths and --coord_paths must both be either filepaths or glob-type expressions"
        )

    # must have one-to-one micrographs and coordinate files
    if len(mrc_files) != len(coord_files):
        raise ValueError(
            f"Number of micrographs and coordinate files must match ({len(mrc_files)} micrographs and {len(coord_files)} coordinate files found)"
        )

    # this is the input to the CoordinateSource constructors
    files = list(zip(mrc_files, coord_files))

    if centers:
        if particle_size == 0:
            raise ValueError(
                "A --particle_size must be specified when loading from centers"
            )
        src = CentersCoordinateSource(
            files,
            particle_size=particle_size,
        )
    else:
        src = EmanCoordinateSource(
            files,
            particle_size=particle_size,
        )

    # saves to .mrcs and STAR file with column "_rlnImageName"
    src.save(
        starfile_out, batch_size=batch_size, save_mode=save_mode, overwrite=overwrite
    )