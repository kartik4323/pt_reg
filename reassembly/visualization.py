"""Bounded qualitative plots; visualization is never an inference input."""
from pathlib import Path

import numpy as np


def plot_scaffold(field: dict, output: Path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    distance, sigma = field['distance'], field['uncertainty']
    bounds = np.asarray(field['bounds'])
    figure, axes = plt.subplots(2, 3, figsize=(10, 6), constrained_layout=True)
    for axis in range(3):
        remaining = [k for k in range(3) if k != axis]
        extent = [bounds[0, remaining[0]], bounds[1, remaining[0]],
                  bounds[0, remaining[1]], bounds[1, remaining[1]]]
        for row, (values, name, cmap) in enumerate(((distance, 'Signed distance', 'coolwarm'),
                                                  (sigma, 'Uncertainty', 'magma'))):
            data = np.take(values, values.shape[axis] // 2, axis=axis).T
            limits = {'vmin': -field['truncation'], 'vmax': field['truncation']} if row == 0 else {'vmin': float(sigma.min()), 'vmax': float(sigma.max()) + 1e-9}
            image = axes[row, axis].imshow(data, origin='lower', extent=extent, cmap=cmap, **limits)
            axes[row, axis].set(title=f'{name}: {"xyz"[axis]} mid-slice',
                                xlabel='xyz'[remaining[0]], ylabel='xyz'[remaining[1]])
            figure.colorbar(image, ax=axes[row, axis], shrink=.8)
    figure.suptitle('Coarse scaffold in normalized reference coordinates')
    figure.savefig(output, dpi=130)
    plt.close(figure)
