import unittest

from psh_pytorch import PerfectSpatialHash


class TestEmpty(unittest.TestCase):

    def test_initialization(self):
        psh = PerfectSpatialHash(None, 3,
                                 build_offset_table=False,
                                 dimensions=3, hash_table_size=25, offset_table_size=25)

        self.assertTrue(psh.offset_table.abs().sum() == 0)


if __name__ == '__main__':
    unittest.main()
