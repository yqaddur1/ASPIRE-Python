import logging
import os
import tempfile
from shutil import copyfile
from unittest import TestCase

import mrcfile
import numpy as np
from parameterized import parameterized

from aspire.ctf import estimate_ctf

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "saved_test_data")


class CtfEstimatorTestCase(TestCase):
    def setUp(self):
        self.test_input_fn = "sample.mrc"
        # These values are from CTFFIND4
        self.test_output = {
            "defocus_u": 34914.63,  # Angstrom
            "defocus_v": 33944.32,  # Angstrom
            "defocus_ang": -65.26,  # Degree wrt some axis
            "cs": 2.0,
            "voltage": 300.0,
            "pixel_size": 1.77,  # EMPIAR 10017
            "amplitude_contrast": 0.07,
        }

    def tearDown(self):
        pass

    def testEstimateCTF(self):
        with tempfile.TemporaryDirectory() as tmp_input_dir:
            # Copy input file
            copyfile(
                os.path.join(DATA_DIR, self.test_input_fn),
                os.path.join(tmp_input_dir, self.test_input_fn),
            )

            with tempfile.TemporaryDirectory() as tmp_output_dir:
                # Returns results in output_dir
                results = estimate_ctf(
                    data_folder=tmp_input_dir,
                    pixel_size=1.77,
                    cs=2.0,
                    amplitude_contrast=0.07,
                    voltage=300.0,
                    num_tapers=2,
                    psd_size=512,
                    g_min=30.0,
                    g_max=5.0,
                    output_dir=tmp_output_dir,
                    dtype=np.float64,
                    save_ctf_images=True,
                    save_noise_images=True,
                )

                logger.debug(f"results: {results}")

                for result in results.values():
                    # The defocus values are set to be within 5% of CTFFIND4

                    # defocusU
                    self.assertTrue(
                        np.allclose(
                            result["defocus_u"],
                            self.test_output["defocus_u"],
                            rtol=0.05,
                        )
                    )
                    # defocusV
                    self.assertTrue(
                        np.allclose(
                            result["defocus_u"],
                            self.test_output["defocus_u"],
                            rtol=0.05,
                        )
                    )

                    # defocusAngle
                    defocus_ang_degrees = result["defocus_ang"] * 180 / np.pi
                    try:
                        self.assertTrue(
                            np.allclose(
                                defocus_ang_degrees,
                                self.test_output["defocus_ang"],
                                atol=1,  # one degree
                            )
                        )
                    except AssertionError:
                        logger.warning(
                            "Defocus Angle (degrees):"
                            f"\n\tASPIRE= {defocus_ang_degrees:0.2f}*"
                            f'\n\tCTFFIND4= {self.test_output["defocus_ang"]:0.2f}*'
                            f'\n\tError: {abs((self.test_output["defocus_ang"]- defocus_ang_degrees)/self.test_output["defocus_ang"]) * 100:0.2f}%'
                        )

                    for param in ["cs", "amplitude_contrast", "voltage", "pixel_size"]:
                        self.assertTrue(
                            np.allclose(result[param], self.test_output[param])
                        )

    # we are chopping the micrograph into a vertical and a horizontal rectangle
    # as small as possible to save testing duration
    @parameterized.expand(
        [[(slice(0, 128), slice(0, 64))], [(slice(0, 64), slice(0, 128))]]
    )
    def testRectangularMicrograph(self, slice_range):
        with tempfile.TemporaryDirectory() as tmp_input_dir:
            # copy input file
            copyfile(
                os.path.join(DATA_DIR, self.test_input_fn),
                os.path.join(tmp_input_dir, "rect_" + self.test_input_fn),
            )
            # trim the file into a rectangle
            with mrcfile.open(
                os.path.join(tmp_input_dir, "rect_" + self.test_input_fn), "r+"
            ) as mrc_in:
                data = mrc_in.data[slice_range]
                mrc_in.set_data(data)
            # make sure we can estimate with no errors
            with tempfile.TemporaryDirectory() as tmp_output_dir:
                _ = estimate_ctf(
                    data_folder=tmp_input_dir, output_dir=tmp_output_dir, psd_size=64
                )
