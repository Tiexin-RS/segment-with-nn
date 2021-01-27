import unittest
import numpy as np

from segelectri.data_loader.tr_ds.tr_ds import get_tr_ds


class TestTrDs(unittest.TestCase):

    def get_tr_ds_demo(self):
        ds = get_tr_ds(original_pattern='/opt/dataset/tr2_cropped/*.png',
                       mask_pattern='/opt/dataset/tr2_cropped/*_class.png')
        for d in ds.take(1):
            self.assertEqual(np.array(d).shape[1:], (2, 1024, 1024, 3))