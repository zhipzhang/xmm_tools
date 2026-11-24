"""Basic package smoke tests."""

import unittest

import xmm_tools


class TestPackage(unittest.TestCase):
    def test_version(self):
        self.assertIsInstance(xmm_tools.__version__, str)
        self.assertEqual(xmm_tools.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
