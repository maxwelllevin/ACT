"""
Bilinear interpolation transformation kernel and its xarray-facing wrapper.

Interpolates input samples onto target coordinate points using piecewise
linear interpolation, accounting for QC flags and range limits. The core
loop is numba-jitted for speed with a pure-Python fallback (see
:mod:`act.transform._numba_support`) in case numba is unavailable or fails
to compile.

"""

import numpy as np

from act.transform._numba_support import JitFallbackKernel
from act.transform.constants import (
    QC_ALL_BAD_INPUTS,
    QC_BAD,
    QC_EXTRAPOLATE,
    QC_INDETERMINATE,
    QC_INTERPOLATE,
    QC_OUTSIDE_RANGE,
)


def _bilinear_interpolate_kernel_impl(
    array,
    qc_array,
    qc_mask,
    index,
    ni,
    output,
    qc_output,
    target,
    nt,
    input_missing_value,
    output_missing_value,
    dist_1,
    dist_2,
    t_range,
):
    if ni < 2:
        for j in range(nt):
            output[j] = output_missing_value
            dist_1[j] = output_missing_value
            dist_2[j] = output_missing_value
            qc_output[j] |= QC_OUTSIDE_RANGE
            qc_output[j] |= QC_BAD
        return 0

    sign = 1
    if nt > 1:
        if (index[0] < index[1]) and (target[0] < target[1]):
            sign = 1
        elif (index[0] > index[1]) and (target[0] > target[1]):
            sign = -1
        else:
            return -5

    i = 0
    for j in range(nt):
        qc_output[j] = 0

        if sign * target[j] < sign * (index[0] - ((index[1] - index[0]) / 2.0)) or sign * target[
            j
        ] > sign * (index[ni - 1] + ((index[ni - 1] - index[ni - 2]) / 2.0)):
            output[j] = output_missing_value
            dist_1[j] = output_missing_value
            dist_2[j] = output_missing_value
            qc_output[j] |= QC_OUTSIDE_RANGE
            qc_output[j] |= QC_BAD
            continue

        while i < ni and sign * index[i] < sign * target[j]:
            i += 1

        if (
            i < ni
            and abs(target[j] - index[i]) < 1e-8
            and abs(array[i] - input_missing_value) > 1e-8
            and not (qc_array[i] & qc_mask)
            and np.isfinite(array[i])
        ):
            output[j] = array[i]
            dist_1[j] = 0.0
            dist_2[j] = 0.0
            continue

        if i == ni:
            n1 = ni - 2
            n2 = ni - 1
        elif i == 0:
            n1 = 0
            n2 = 1
        else:
            n1 = i - 1
            n2 = i

        while (n1 >= 0) and (
            abs(array[n1] - input_missing_value) < 1e-8
            or (qc_array[n1] & qc_mask)
            or not np.isfinite(array[n1])
        ):
            qc_output[j] |= QC_INTERPOLATE
            n1 -= 1

        while (n1 < ni) and (
            n1 < 0
            or n1 == n2
            or abs(array[n1] - input_missing_value) < 1e-8
            or (qc_array[n1] & qc_mask)
            or not np.isfinite(array[n1])
        ):
            qc_output[j] |= QC_INTERPOLATE
            n1 += 1

        if n1 >= ni:
            for k in range(nt):
                output[k] = output_missing_value
                dist_1[k] = output_missing_value
                dist_2[k] = output_missing_value
                qc_output[k] |= QC_ALL_BAD_INPUTS
                qc_output[k] |= QC_BAD
            return 2

        while (n2 < ni) and (
            n2 == n1
            or abs(array[n2] - input_missing_value) < 1e-8
            or (qc_array[n2] & qc_mask)
            or not np.isfinite(array[n2])
        ):
            qc_output[j] |= QC_INTERPOLATE
            n2 += 1
        while (n2 > 0) and (
            n2 == n1
            or n2 >= ni
            or abs(array[n2] - input_missing_value) < 1e-8
            or (qc_array[n2] & qc_mask)
            or not np.isfinite(array[n2])
        ):
            qc_output[j] |= QC_INTERPOLATE
            n2 -= 1

        if n2 >= ni or n2 <= 0 or n2 == n1:
            for k in range(nt):
                output[k] = output_missing_value
                dist_1[k] = output_missing_value
                dist_2[k] = output_missing_value
                qc_output[k] |= QC_ALL_BAD_INPUTS
                qc_output[k] |= QC_BAD
            return 2

        x = target[j]
        x1 = index[n1]
        x2 = index[n2]
        y1 = array[n1]
        y2 = array[n2]

        if abs(x - x1) > t_range or abs(x - x2) > t_range:
            output[j] = output_missing_value
            qc_output[j] |= QC_OUTSIDE_RANGE
            qc_output[j] |= QC_BAD
            continue

        u = (x - x1) / (x2 - x1)
        output[j] = u * y2 + (1 - u) * y1
        dist_1[j] = x1 - x
        dist_2[j] = x2 - x

        if u < 0 or u > 1:
            qc_output[j] |= QC_EXTRAPOLATE

        if abs(u - 1) > 1e-5 and (qc_array[n1] & ~qc_mask) != 0:
            qc_output[j] |= QC_INDETERMINATE

        if abs(u) > 1e-5 and (qc_array[n2] & ~qc_mask) != 0:
            qc_output[j] |= QC_INDETERMINATE

    return 0


_bilinear_interpolate_kernel = JitFallbackKernel(
    _bilinear_interpolate_kernel_impl, 'interpolate', cache=True
)


def _bilinear_interpolate_1d(
    array,
    qc_array,
    qc_mask,
    index,
    ni,
    output,
    qc_output,
    target,
    nt,
    input_missing_value,
    output_missing_value,
    rmet,
    t_range,
):
    """Bilinear-interpolate ``array`` onto ``target`` coordinate points.

    This is the raw-numpy kernel-level function used internally by
    :func:`act.transform.driver.transform_1d`. Most users should call
    :func:`act.transform.interpolate.interpolate` instead, which operates on
    :class:`xarray.DataArray` objects.

    Parameters
    ----------
    array : numpy.ndarray
        Input data values (length ``ni``).
    qc_array : numpy.ndarray
        Integer QC flags for each input sample (length ``ni``).
    qc_mask : int
        Bitmask; bits set here are treated as bad.
    index : numpy.ndarray
        Input coordinate values (length ``ni``).
    ni : int
        Number of input samples.
    output : numpy.ndarray
        Pre-allocated output array (length ``nt``); modified in place.
    qc_output : numpy.ndarray
        Pre-allocated integer QC output array (length ``nt``); modified in place.
    target : numpy.ndarray
        Target coordinate values (length ``nt``).
    nt : int
        Number of target points.
    input_missing_value : float
        Sentinel for missing input.
    output_missing_value : float
        Sentinel to fill missing output.
    rmet : list of numpy.ndarray
        ``[dist_1, dist_2]`` - pre-allocated distance metric arrays (each length ``nt``).
    t_range : float
        Maximum coordinate distance allowed for interpolation.

    Returns
    -------
    int
        0 on success, negative on error.

    """
    return _bilinear_interpolate_kernel(
        array,
        qc_array,
        qc_mask,
        index,
        ni,
        output,
        qc_output,
        target,
        nt,
        input_missing_value,
        output_missing_value,
        rmet[0],
        rmet[1],
        t_range,
    )


def interpolate(data, target, dim, qc=None, qc_mask=0, t_range=None):
    """Bilinearly interpolate ``data`` onto ``target`` coordinate values along ``dim``.

    Parameters
    ----------
    data : xarray.DataArray
        Input DataArray.
    target : xarray.DataArray or numpy.ndarray
        Target coordinate values to interpolate onto.
    dim : str
        Name of the dimension along which to apply the transform.
    qc : xarray.DataArray, optional
        Optional integer QC DataArray with same shape as ``data``.
    qc_mask : int
        Bitmask of QC bits that indicate bad data.
    t_range : float, optional
        Maximum distance from a target point to an input point for interpolation.
        Defaults to the median spacing of input coordinate values.

    Returns
    -------
    result : xarray.DataArray
        Interpolated DataArray on the target coordinate.
    result_qc : xarray.DataArray
        Integer QC DataArray with same shape as ``result``, with CF
        ``flag_masks``/``flag_meanings``/``flag_assessments`` attributes set.

    Examples
    --------
        .. code-block:: python

            result, result_qc = act.transform.interpolate(ds['temp'], target_time, dim='time')

    """
    from act.transform.driver import apply_transform

    return apply_transform(
        data,
        target,
        dim,
        'interpolate',
        qc=qc,
        qc_mask=qc_mask,
        t_range=t_range,
    )
