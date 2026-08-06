"""
Propagating quality control through a resample
----------------------------------------------

This example shows the behavior that distinguishes ``act.transform`` from a
plain ``xarray`` resample: input quality control is honored, so flagged samples
are excluded from the average, and output quality control is generated, so the
result records which output points were affected.

The corrected tipping-bucket precipitation variable in this file contains
7999 mm spikes that trip the datastream's own ``fail_max`` test. Averaging
without consulting QC carries those spikes into the result.

"""

import matplotlib.pyplot as plt
import numpy as np
from arm_test_data import DATASETS

import act
from act.transform.constants import QC_SOME_BAD_INPUTS

filename = DATASETS.fetch('gucmetM1.b1.20230301.000000.cdf')
ds = act.io.arm.read_arm_netcdf(filename, cleanup_qc=True)

var_name = 'tbrg_precip_total_corr'
qc_var_name = 'qc_' + var_name

# Build the qc_mask from the QC variable's own CF flag attributes rather than
# hardcoding a bit value, so the same code works across datastreams. Here every
# declared test is assessed "Bad".
qc_da = ds[qc_var_name]
qc_mask = 0
for mask, assessment in zip(qc_da.attrs['flag_masks'], qc_da.attrs['flag_assessments']):
    if assessment == 'Bad':
        qc_mask |= mask

print(f'flag_meanings: {qc_da.attrs["flag_meanings"]}')
print(f'derived qc_mask = {qc_mask}')

target = act.transform.make_coord('2023-03-01', '2023-03-02', '30min', name='time')

# Without QC: every sample is averaged, spikes included.
plain, plain_qc = act.transform.bin_average(ds[var_name], target, dim='time')

# With QC: samples matching qc_mask are excluded from the computation, and the
# output QC records which bins lost input.
filtered, filtered_qc = act.transform.bin_average(
    ds[var_name], target, dim='time', qc=qc_da, qc_mask=qc_mask
)

print(f'\nInput maximum:            {float(ds[var_name].max()):.1f} mm')
print(f'Output maximum without QC: {float(plain.max()):.1f} mm')
print(f'Output maximum with QC:    {float(filtered.max()):.1f} mm')

# Every sample that survives the QC mask is 0.0 mm, so the entire signal in the
# unfiltered result was spurious.
kept = ds[var_name].values[~(qc_da.values & qc_mask).astype(bool)]
print(f'Range of the samples QC kept: {kept.min():.1f} to {kept.max():.1f} mm')

some_bad = (filtered_qc.values & QC_SOME_BAD_INPUTS).astype(bool)
print(f'\nBins flagged QC_SOME_BAD_INPUTS: {some_bad.sum()} of {filtered.size}')

# Count how many input samples each output bin lost, which is what drives the
# QC_SOME_BAD_INPUTS bit above.
flagged_input = (qc_da.values & qc_mask).astype(bool)
bin_edges = target.values
lost_per_bin = np.array(
    [
        flagged_input[
            (ds['time'].values >= bin_edges[i]) & (ds['time'].values < bin_edges[i + 1])
        ].sum()
        for i in range(len(bin_edges) - 1)
    ]
)

fig, (ax0, ax1, ax2) = plt.subplots(
    3, 1, figsize=(11, 9), sharex=True, gridspec_kw={'height_ratios': [2, 2, 1.4]}
)

# Panel 1: the flagged input. Plotted on a log scale because the spikes are four
# orders of magnitude above the real signal.
flagged = (ds[qc_var_name].values & qc_mask).astype(bool)
ax0.plot(ds['time'].values, ds[var_name].values, color='0.6', lw=0.8, label='Input')
ax0.plot(
    ds['time'].values[flagged],
    ds[var_name].values[flagged],
    'rx',
    ms=7,
    label=f'Flagged by {qc_var_name} ({flagged.sum()} samples)',
)
ax0.set_yscale('symlog', linthresh=1)
ax0.set_ylabel(f'{var_name}\n({ds[var_name].attrs["units"]})')
ax0.set_title('Input: real ARM data with samples its own QC variable flags as bad')
ax0.legend(loc='upper left', fontsize=8)
ax0.grid(alpha=0.3)

# Panel 2: the two results. Ignoring QC lets the spikes dominate.
ax1.plot(
    plain['time'].values,
    plain.values,
    color='tab:red',
    marker='o',
    ms=4,
    lw=1.5,
    label='bin_average without QC (spikes averaged in)',
)
ax1.plot(
    filtered['time'].values,
    filtered.values,
    color='tab:blue',
    marker='o',
    ms=4,
    lw=1.5,
    label='bin_average with qc_mask (spikes excluded)',
)
ax1.set_yscale('symlog', linthresh=0.1)
ax1.set_ylabel(f'30-min mean\n({ds[var_name].attrs["units"]})')
ax1.set_title('Output: the same transform, with and without the QC variable')
ax1.legend(loc='upper left', fontsize=8)
ax1.grid(alpha=0.3)

# Panel 3: the generated output QC, so the propagation is visible. Bars show how
# many input samples each bin lost; the markers show where the transform set
# QC_SOME_BAD_INPUTS on its output. They line up exactly, which is the point.
ax2.bar(
    bin_edges[:-1],
    lost_per_bin,
    width=np.diff(bin_edges),
    align='edge',
    color='0.75',
    edgecolor='0.5',
    label='Input samples excluded by qc_mask',
)
marker_level = lost_per_bin.max() + 0.7
ax2.plot(
    filtered['time'].values[some_bad],
    np.full(some_bad.sum(), marker_level),
    's',
    color='tab:orange',
    ms=6,
    label='Output bin flagged QC_SOME_BAD_INPUTS',
)
ax2.set_ylim(0, marker_level + 0.9)
ax2.set_ylabel('Samples\nexcluded')
ax2.set_xlabel('Time (UTC)')
ax2.set_title('Generated output QC: excluded input per bin, and the resulting QC bit')
ax2.legend(loc='upper left', fontsize=8)
ax2.grid(alpha=0.3, axis='x')

fig.tight_layout()
plt.show()

ds.close()
