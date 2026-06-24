"""
Read datasets from HDF5 files into napari
"""

from collections.abc import Callable

import h5py
import numpy as np


def napari_get_reader(path: str | list[str]) -> Callable | None:
    """
    Check if the provided path is for an HDF5 file.

    Parameters
    ----------
    path : str or list of str
        Path to file, or list of paths.

    Returns
    -------
    Callable | None
        If the path is a recognized format, return a function that accepts the
        same path or list of paths, and returns a list of layer data tuples.
    """

    if isinstance(path, list):
        path = path[0]
    if path[path.rfind('.') :] not in ['.h5', '.hdf', '.hdf5']:
        return None
    else:
        try:
            h5_file: h5py.File = h5py.File(path)
            tree_list, dset_found = h5_expand_tree(
                h5_file=h5_file, break_on_dset=True
            )
            h5_file.close()
            if not dset_found:
                return None
        except OSError:
            return None
    return reader_function


def h5_expand_tree(
    path: str | None = None,
    *,
    h5_file: h5py.File | None = None,
    tree_list: list[str] = None,
    break_on_dset: bool = False,
    dset_found: bool = False,
    print_tree: bool = False,
) -> list[str] | bool:
    """
    Recursive function to expand the file tree of an h5 file.

    Parameters
    ----------
    file_path: str
        Path to file. This is ignored if h5_file is provided.
    h5_file: h5py.File (optional)
        An h5 file object created from h5py.File.
    tree_list: list[str] (optional)
        The current list of directories in the h5 file tree.
    break_on_dset: bool
        Whether or not to break the recursion upon finding the first dataset.
    dset_found: bool
        Indicates if a dataset was found during recursive expansion.
    print_tree: bool
        Whether or not to print the file tree.

    Returns
    -------
    tree_list: list[str]
        A list containing the h5 file's directory tree.
    dset_found: bool
        Indicates if a dataset was found during recursive expansion.
    """

    if not tree_list:
        tree_list: list = []
    if not break_on_dset or (break_on_dset and not dset_found):
        if not h5_file:
            h5_file: h5py.File = h5py.File(path)
        contents: list[str] = list(h5_file.keys())
        if h5_file.name == '/':
            tab_count: int = 0
        else:
            tab_count: int = h5_file.name.count('/')
        for item in contents:
            label = h5_file[item].name[h5_file[item].name.rfind('/') + 1 :]
            if isinstance(h5_file[item], h5py.Group):
                if print_tree:
                    print((tab_count * '  ') + '- ' + label)
                tree_list, dset_found = h5_expand_tree(
                    h5_file=h5_file[item],
                    tree_list=tree_list,
                    break_on_dset=break_on_dset,
                    dset_found=dset_found,
                    print_tree=print_tree,
                )
            elif isinstance(h5_file[item], h5py.Dataset):
                dset_found = True
                if len(h5_file[item].shape):
                    label += ' [shape: ' + str(h5_file[item].shape) + ']'
                    tree_list.append(h5_file[item].name)
                if print_tree:
                    print((tab_count * '  ') + '- ' + label)
            if dset_found and break_on_dset:
                return tree_list, dset_found
    return tree_list, dset_found


def h5_find_largest(file: str | h5py.File) -> str:
    """
    Automatically locate the largest dataset within an h5 file tree.

    Parameters
    ----------
    file: str | h5py.File
        Path to file or an h5py.File object.

    Returns
    -------
    largest_path: str
        The path within the h5 file to the largest item.
    """

    if isinstance(file, str):
        file: h5py.File = h5py.File(file)
    tree_list, _ = h5_expand_tree(h5_file=file)
    prev_bytes: int = 0
    for directory in tree_list:
        if file[directory].nbytes > prev_bytes:
            current_largest: str = directory
            prev_bytes = file[directory].nbytes
    return current_largest


def reader_function(path: str | list[str]) -> list[tuple]:
    """Take a path or list of paths and return a list of LayerData tuples.

    Readers are expected to return data as a list of tuples, where each tuple
    is (data, [add_kwargs, [layer_type]]), "add_kwargs" and "layer_type" are
    both optional.

    Parameters
    ----------
    path : str | list[str]
        Path to file, or list of paths.

    Returns
    -------
    layer_data : list[tuple]
        A list of LayerData tuples where each tuple in the list contains
        (data, metadata, layer_type), where data is a numpy array, metadata is
        a dict of keyword arguments for the corresponding viewer.add_* method
        in napari, and layer_type is a lower-case string naming the type of
        layer. Both "meta", and "layer_type" are optional. napari will
        default to layer_type=="image" if not provided
    """

    paths = [path] if isinstance(path, str) else path
    arrays: list[np.ndarray] = []
    for _path in paths:
        h5_file: h5py.File = h5py.File(_path)
        arrays.append(np.array(h5_file[h5_find_largest(h5_file)]))
        h5_file.close()
    if arrays[0].ndim > 2:
        data: np.ndarray = np.concat(arrays, 0)
    else:
        data: np.ndarray = np.squeeze(np.stack(arrays))
    return [(data, {}, 'image')]
