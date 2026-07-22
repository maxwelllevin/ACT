"""
QC flag bit constants shared by the ``act.transform`` kernels
(:mod:`act.transform.bin_average`, :mod:`act.transform.interpolate`,
:mod:`act.transform.subsample`).

Each flag is a bit position in a packed integer QC field. Multiple flags
can be set simultaneously via bitwise OR. ``QC_FLAG_MEANINGS`` and
``QC_FLAG_ASSESSMENTS`` give the CF-style ``flag_meanings``/``flag_assessments``
text for each flag, in the same order as ``QC_ALL_FLAGS``, so that output QC
variables can be annotated the same way :func:`act.qc.qcfilter.QCFilter.add_test`
annotates its QC variables.

"""

QC_BAD = 1 << 1
QC_INDETERMINATE = 1 << 2
QC_INTERPOLATE = 1 << 3
QC_EXTRAPOLATE = 1 << 4
QC_NOT_USING_CLOSEST = 1 << 5
QC_SOME_BAD_INPUTS = 1 << 6
QC_ZERO_WEIGHT = 1 << 7
QC_OUTSIDE_RANGE = 1 << 8
QC_ALL_BAD_INPUTS = 1 << 9
QC_BAD_STD = 1 << 10
QC_INDETERMINATE_STD = 1 << 11
QC_BAD_GOODFRAC = 1 << 12
QC_INDETERMINATE_GOODFRAC = 1 << 13

# Ordered (flag_mask, flag_meaning, flag_assessment) triples used to populate
# CF-style flag_masks/flag_meanings/flag_assessments attributes on QC
# variables produced by act.transform. Order matches historical act-transform
# bit ordering; any subset of these bits may appear in a given output QC
# variable depending on which transform produced it.
QC_ALL_FLAGS = [
    QC_BAD,
    QC_INDETERMINATE,
    QC_INTERPOLATE,
    QC_EXTRAPOLATE,
    QC_NOT_USING_CLOSEST,
    QC_SOME_BAD_INPUTS,
    QC_ZERO_WEIGHT,
    QC_OUTSIDE_RANGE,
    QC_ALL_BAD_INPUTS,
    QC_BAD_STD,
    QC_INDETERMINATE_STD,
    QC_BAD_GOODFRAC,
    QC_INDETERMINATE_GOODFRAC,
]

QC_FLAG_MEANINGS = [
    'Transformed value is bad',
    'Transformed value is indeterminate',
    'Transformed value is interpolated from neighboring values',
    'Transformed value is extrapolated beyond the input range',
    'Transformed value did not use the closest available input sample',
    'Some input samples used for this transformed value were flagged bad and excluded',
    'Transformed value has zero total weight from contributing input samples',
    'Transformed value is outside the range of the input coordinate',
    'All input samples for this transformed value were flagged bad',
    'Standard deviation of input samples exceeds the bad threshold',
    'Standard deviation of input samples exceeds the indeterminate threshold',
    'Fraction of good input coverage is below the bad threshold',
    'Fraction of good input coverage is below the indeterminate threshold',
]

QC_FLAG_ASSESSMENTS = [
    'Bad',
    'Indeterminate',
    'Indeterminate',
    'Indeterminate',
    'Indeterminate',
    'Indeterminate',
    'Bad',
    'Bad',
    'Bad',
    'Bad',
    'Indeterminate',
    'Bad',
    'Indeterminate',
]


def add_cf_qc_attrs(qc_da):
    """Populate CF-style QC attributes on a transform output QC DataArray.

    Sets ``flag_masks``, ``flag_meanings``, ``flag_assessments``, and
    ``standard_name='quality_flag'`` on ``qc_da`` following the same
    convention used by :meth:`act.qc.qcfilter.QCFilter.create_qc_variable`,
    so that transform output composes cleanly with ``ds.qcfilter``/``ds.clean``.

    Parameters
    ----------
    qc_da : xarray.DataArray
        Integer QC DataArray to annotate in place.

    Returns
    -------
    qc_da : xarray.DataArray
        The same DataArray, with QC attributes populated.

    """
    qc_da.attrs['flag_masks'] = list(QC_ALL_FLAGS)
    qc_da.attrs['flag_meanings'] = list(QC_FLAG_MEANINGS)
    qc_da.attrs['flag_assessments'] = list(QC_FLAG_ASSESSMENTS)
    qc_da.attrs['standard_name'] = 'quality_flag'
    return qc_da
