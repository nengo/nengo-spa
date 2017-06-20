"""Functions for handling resources."""

import os
from pkg_resources import resource_isdir, resource_listdir, resource_stream
import shutil


def extract_resource(resource_name, destination_path):
    """Extracts a nengo_spa resource.

    Parameters
    ----------
    resource_name : str
        Name of the resource to extract.
    destination_path : str
        Path to extract to. If the last component of the path does not exist,
        it will be created.
    """
    if not os.path.exists(destination_path):
        os.mkdir(destination_path)

    package = 'nengo_spa'
    for dirpath, dirnames, filenames in resource_walk(package, resource_name):
        path = os.path.join(
            destination_path, dirpath[(len(resource_name) + 1):])
        for dirname in dirnames:
            os.mkdir(os.path.join(path, dirname))
        for filename in filenames:
            src_path = os.path.join(dirpath, filename)
            dst_path = os.path.join(path, filename)
            with resource_stream(package, src_path) as src, \
                    open(dst_path, 'wb') as dst:
                shutil.copyfileobj(src, dst)


def resource_walk(package_or_requirement, resource_name):
    """Generate the file names in a resource tree.

    Parameters
    ----------
    package_or_requirement : str or Requirement
        Package or requirement that contains the resource.
    resource_name : str
        Name of the resource.

    Returns
    -------
    tuple
        For each directory in the tree rooted at the given resoruce a 3-tuple
        ``(dirpath, dirnames, filenames)`` is returned. *dirpath* is a string,
        the path to the directory starting with *resource_name*. *dirnames* is
        a list of the names of subdirectories in *dirpath*. *filenames* is a
        list of names of non-directory files in *dirpath*.
    """
    queue = [resource_name]
    while len(queue) > 0:
        dirpath = queue.pop()
        dirnames = []
        filenames = []
        for name in resource_listdir(package_or_requirement, dirpath):
            fullpath = os.path.join(dirpath, name)
            if resource_isdir(package_or_requirement, fullpath):
                dirnames.append(name)
                queue.append(fullpath)
            else:
                filenames.append(name)
        yield dirpath, dirnames, filenames
