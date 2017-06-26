import unittest

class ImportTest(unittest.TestCase):
  def test_imports(self):
    import crayimage
    import crayimage.nn
    import crayimage.cosmicGAN

    from crayimage.cosmicGAN.gannery import EPreservingUNet
