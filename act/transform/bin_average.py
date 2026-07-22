"""
Bin-average transformation kernel and its xarray-facing wrapper.

Computes a weighted average of input samples that fall within each output
bin, accounting for partial bin overlaps, QC flags, and coverage metrics.
The core loop is numba-jitted for speed with a pure-Python fallback (see
:mod:`act.transform._numba_support`) in case numba is unavailable or fails
to compile.

"""

import numpy as np

from act.transform._numba_support import JitFallbackKernel
from act.transform.constants import (
    QC_ALL_BAD_INPUTS,
    QC_BAD,
    QC_BAD_GOODFRAC,
    QC_BAD_STD,
    QC_INDETERMINATE,
    QC_INDETERMINATE_GOODFRAC,
    QC_INDETERMINATE_STD,
    QC_OUTSIDE_RANGE,
    QC_SOME_BAD_INPUTS,
    QC_ZERO_WEIGHT,
)


def _bin_average_kernel_impl(
    array,
    qc_array,
    qc_mask,
    index_start,
    index_end,
    weights,
    ni,
    output,
    qc_output,
    target_start,
    target_end,
    nt,
    input_missing_value,
    output_missing_value,
    stdev,
    coverage,
    std_bad_max,
    std_ind_max,
    goodfrac_bad_min,
    goodfrac_ind_min,
):
    if (ni == 1 or index_start[0] < index_start[1]) and (
        nt == 1 or target_start[0] < target_start[1]
    ):
        sign = 1
    elif (ni == 1 or index_start[0] > index_start[1]) and (
        nt == 1 or target_start[0] > target_start[1]
    ):
        sign = -1
    else:
        return -5

    i0 = 0
    for j in range(nt):
        sum_array = sum_weight = max_weight = 0.0
        sum_array2 = sum_weight2 = 0.0
        total_span = good_span = 0.0
        qco = 0
        qc_output[j] = 0
        i = i0

        while i < ni and sign * index_end[i] < sign * target_start[j]:
            i += 1

        i0 = i

        while i < ni and sign * index_start[i] < sign * target_end[j]:
            if sign * index_end[i] < sign * target_start[j]:
                i += 1
                continue

            w = 1.0
            bin_width = index_end[i] - index_start[i]

            if bin_width == 0.0:
                u = 0.0
                v = 0.0
            else:
                u = (target_start[j] - index_start[i]) / bin_width
                if u > 0:
                    w -= u
                v = (index_end[i] - target_end[j]) / bin_width
                if v > 0:
                    w -= v

            if u > 1.0 or v > 1.0 or u + v > 1.0 or w < 0.0:
                return -1

            if abs(bin_width) > 0:
                total_span += w * sign * bin_width
            else:
                total_span += 1.0

            if w > 0 and weights[i] > max_weight:
                max_weight = weights[i]

            if w > 0 and (
                array[i] == input_missing_value
                or (qc_array[i] & qc_mask)
                or not np.isfinite(array[i])
            ):
                qc_output[j] |= QC_SOME_BAD_INPUTS
                i += 1
                continue
            else:
                if abs(bin_width) > 0:
                    good_span += w * sign * bin_width
                else:
                    good_span += 1.0

            w *= weights[i]

            sum_array += w * array[i]
            sum_weight += w
            sum_array2 += w * array[i] * array[i]
            sum_weight2 += w * w

            if w > 0:
                qco |= qc_array[i]

            i += 1

        if max_weight == 0 and i > i0:
            output[j] = 0.0
            stdev[j] = 0.0
            coverage[j] = 0.0
            qc_output[j] |= QC_ZERO_WEIGHT
        elif i == i0:
            output[j] = output_missing_value
            stdev[j] = output_missing_value
            coverage[j] = 0.0
            qc_output[j] |= QC_OUTSIDE_RANGE
            qc_output[j] |= QC_BAD
        elif sum_weight == 0:
            output[j] = output_missing_value
            stdev[j] = output_missing_value
            coverage[j] = 0.0
            qc_output[j] |= QC_ALL_BAD_INPUTS
            qc_output[j] |= QC_BAD
        else:
            output[j] = sum_array / sum_weight

            # Weighted population variance: (s0*s2 - s1^2) / s0^2
            stdev[j] = sum_weight * sum_array2 - sum_array * sum_array
            stdev[j] /= sum_weight * sum_weight

            if abs(stdev[j]) < 1e-12:
                stdev[j] = 0.0
            elif stdev[j] < 0:
                stdev[j] = output_missing_value
            else:
                stdev[j] = np.sqrt(stdev[j])

            coverage[j] = good_span / total_span

            if (qco & ~qc_mask) != 0:
                qc_output[j] |= QC_INDETERMINATE

        if stdev[j] != output_missing_value:
            if stdev[j] > std_bad_max:
                qc_output[j] |= QC_BAD_STD
            elif stdev[j] > std_ind_max:
                qc_output[j] |= QC_INDETERMINATE_STD

        if coverage[j] != output_missing_value:
            if coverage[j] < goodfrac_bad_min:
                qc_output[j] |= QC_BAD_GOODFRAC
            elif coverage[j] < goodfrac_ind_min:
                qc_output[j] |= QC_INDETERMINATE_GOODFRAC

    return 0


_bin_average_kernel = JitFallbackKernel(_bin_average_kernel_impl, 'bin_average', cache=True)


def _bin_average_1d(
    array,
    qc_array,
    qc_mask,
    index_start,
    index_end,
    weights,
    ni,
    output,
    qc_output,
    target_start,
    target_end,
    nt,
    input_missing_value,
    output_missing_value,
    rmet,
    std_bad_max,
    std_ind_max,
    goodfrac_bad_min,
    goodfrac_ind_min,
):
    """Bin-average ``array`` onto the target bins defined by ``target_start``/``target_end``.

    This is the raw-numpy kernel-level function used internally by
    :func:`act.transform.driver.transform_1d`. Most users should call
    :func:`act.transform.bin_average.bin_average` instead, which operates on
    :class:`xarray.DataArray` objects.

    Parameters
    ----------
    array : numpy.ndarray
        Input data values (length ``ni``).
    qc_array : numpy.ndarray
        Integer QC flags for each input sample (length ``ni``).
    qc_mask : int
        Bitmask; bits set here are treated as bad.
    index_start : numpy.ndarray
        Start coordinate of each input bin (length ``ni``).
    index_end : numpy.ndarray
        End coordinate of each input bin (length ``ni``).
    weights : numpy.ndarray or None
        Per-sample weights, or None for uniform weights.
    ni : int
        Number of input samples.
    output : numpy.ndarray
        Pre-allocated output array (length ``nt``); modified in place.
    qc_output : numpy.ndarray
        Pre-allocated integer QC output array (length ``nt``); modified in place.
    target_start : numpy.ndarray
        Start coordinate of each output bin (length ``nt``).
    target_end : numpy.ndarray
        End coordinate of each output bin (length ``nt``).
    nt : int
        Number of output bins.
    input_missing_value : float
        Sentinel for missing input.
    output_missing_value : float
        Sentinel to fill missing output.
    rmet : list of numpy.ndarray
        ``[stdev, coverage]`` - pre-allocated metric arrays (each length ``nt``).
    std_bad_max : float
        Stdev threshold above which output is flagged ``QC_BAD_STD``.
    std_ind_max : float
        Stdev threshold above which output is flagged ``QC_INDETERMINATE_STD``.
    goodfrac_bad_min : float
        Coverage fraction below which output is flagged ``QC_BAD_GOODFRAC``.
    goodfrac_ind_min : float
        Coverage fraction below which output is flagged ``QC_INDETERMINATE_GOODFRAC``.

    Returns
    -------
    int
        0 on success, negative on error.

    """
    if weights is None:
        weights = np.ones(ni, dtype=np.float64)
    return _bin_average_kernel(
        array,
        qc_array,
        qc_mask,
        index_start,
        index_end,
        weights,
        ni,
        output,
        qc_output,
        target_start,
        target_end,
        nt,
        input_missing_value,
        output_missing_value,
        rmet[0],
        rmet[1],
        std_bad_max,
        std_ind_max,
        goodfrac_bad_min,
        goodfrac_ind_min,
    )


def bin_average(
    data,
    target,
    dim,
    qc=None,
    qc_mask=0,
    input_bounds=None,
    output_bounds=None,
    weights=None,
    std_bad_max=np.inf,
    std_ind_max=np.inf,
    goodfrac_bad_min=0.0,
    goodfrac_ind_min=0.0,
):
    """Bin-average ``data`` onto ``target`` coordinate values along ``dim``.

    Parameters
    ----------
    data : xarray.DataArray
        Input DataArray.
    target : xarray.DataArray or numpy.ndarray
        Target coordinate values to average onto.
    dim : str
        Name of the dimension along which to apply the transform.
    qc : xarray.DataArray, optional
        Optional integer QC DataArray with same shape as ``data``.
    qc_mask : int
        Bitmask of QC bits that indicate bad data.
    input_bounds : numpy.ndarray, optional
        Shape ``(ni, 2)`` array of [start, end] bounds for each input bin.
        Inferred from midpoints if None.
    output_bounds : numpy.ndarray, optional
        Shape ``(nt, 2)`` array of [start, end] bounds for each output bin.
        Inferred from midpoints if None.
    weights : numpy.ndarray, optional
        Per-input-sample weights. Defaults to ones.
    std_bad_max : float
        Stdev above which output is flagged ``QC_BAD_STD``.
    std_ind_max : float
        Stdev above which output is flagged ``QC_INDETERMINATE_STD``.
    goodfrac_bad_min : float
        Coverage fraction below which output is flagged ``QC_BAD_GOODFRAC``.
    goodfrac_ind_min : float
        Coverage fraction below which output is flagged ``QC_INDETERMINATE_GOODFRAC``.

    Returns
    -------
    result : xarray.DataArray
        Transformed DataArray on the target coordinate.
    result_qc : xarray.DataArray
        Integer QC DataArray with same shape as ``result``, with CF
        ``flag_masks``/``flag_meanings``/``flag_assessments`` attributes set.

    Examples
    --------
        .. code-block:: python

            result, result_qc = act.transform.bin_average(ds['temp'], target_time, dim='time')

    """
    from act.transform.driver import apply_transform

    return apply_transform(
        data,
        target,
        dim,
        'bin_average',
        qc=qc,
        qc_mask=qc_mask,
        input_bounds=input_bounds,
        output_bounds=output_bounds,
        weights=weights,
        std_bad_max=std_bad_max,
        std_ind_max=std_ind_max,
        goodfrac_bad_min=goodfrac_bad_min,
        goodfrac_ind_min=goodfrac_ind_min,
    )
