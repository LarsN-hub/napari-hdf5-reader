"""
Tests for the _reader module
"""

import h5py
import numpy as np

from napari_hdf5_reader import napari_get_reader


def test_reader(tmp_path):
    """
    Test that the HDF5 reader imports a test file correctly.
    """

    test_file: str = str(tmp_path / 'test_file.hdf5')
    test_data = np.random.rand(20, 20)
    with h5py.File(test_file, 'w') as h5_file:
        _ = h5_file.create_dataset('test_dset', data=test_data)
    reader = napari_get_reader(test_file)
    assert callable(reader)
    layer_data_list = reader(test_file)
    assert isinstance(layer_data_list, list) and len(layer_data_list) > 0
    layer_data_tuple = layer_data_list[0]
    assert isinstance(layer_data_tuple, tuple) and len(layer_data_tuple) > 0
    np.testing.assert_allclose(test_data, layer_data_tuple[0])


def test_get_reader_pass(tmp_path):
    """
    Test that the HDF5 reader is not used when the file is not an HDF5.
    """

    reader = napari_get_reader('fake.file')
    assert reader is None
    test_file = str(tmp_path / 'test_file.hdf5')
    h5_file: h5py.File = h5py.File(test_file, 'w')
    h5_file.close()
    reader = napari_get_reader(test_file)
    assert reader is None
