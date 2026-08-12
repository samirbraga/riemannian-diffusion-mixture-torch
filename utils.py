import numpy as np
import glob
import os
import hashlib
from typing import Literal

def modulo_with_wrapped_range(
    vals, range_min: float = -np.pi, range_max: float = np.pi
):
    """
    Modulo with wrapped range -- capable of handing a range with a negative min

    >>> modulo_with_wrapped_range(3, -2, 2)
    -1
    """
    assert range_min <= 0.0
    assert range_min < range_max

    # Modulo after we shift values
    top_end = range_max - range_min
    # Shift the values to be in the range [0, top_end)
    vals_shifted = vals - range_min
    # Perform modulo
    vals_shifted_mod = vals_shifted % top_end
    # Shift back down
    retval = vals_shifted_mod + range_min

    # Checks
    # print("Mod return", vals, " --> ", retval)
    # if isinstance(retval, torch.Tensor):
    #     notnan_idx = ~torch.isnan(retval)
    #     assert torch.all(retval[notnan_idx] >= range_min)
    #     assert torch.all(retval[notnan_idx] < range_max)
    # else:
    #     assert (
    #         np.nanmin(retval) >= range_min
    #     ), f"Illegal value: {np.nanmin(retval)} < {range_min}"
    #     assert (
    #         np.nanmax(retval) <= range_max
    #     ), f"Illegal value: {np.nanmax(retval)} > {range_max}"
    return retval

def md5_all_py_files(dirname: str) -> str:
    """Create a single md5 sum for all given files"""
    # https://stackoverflow.com/questions/36099331/how-to-grab-all-files-in-a-folder-and-get-their-md5-hash-in-python
    fnames = glob.glob(os.path.join(dirname, "*.py"))
    hash_md5 = hashlib.md5()
    for fname in sorted(fnames):
        with open(fname, "rb") as f:
            for chunk in iter(lambda: f.read(2**20), b""):
                hash_md5.update(chunk)
    return hash_md5.hexdigest()

def tolerant_comparison_check(values, cmp: Literal[">=", "<="], v):
    """
    Compares values in a way that is tolerant of numerical precision
    >>> tolerant_comparison_check(-3.1415927410125732, ">=", -np.pi)
    True
    """
    if cmp == ">=":  # v is a lower bound
        minval = np.nanmin(values)
        diff = minval - v
        if np.isclose(diff, 0, atol=1e-5):
            return True  # Passes
        return diff > 0
    elif cmp == "<=":
        maxval = np.nanmax(values)
        diff = maxval - v
        if np.isclose(diff, 0, atol=1e-5):
            return True
        return diff < 0
    else:
        raise ValueError(f"Illegal comparator: {cmp}")