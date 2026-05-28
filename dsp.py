import numpy as np
from scipy.stats import kurtosis, skew

def generate_features(implementation_version, draw_graphs, raw_data, axes, sampling_freq):
    N_SUBWINDOWS = 20
    MIN_SUBWINDOW_SAMPLES = 32

    n_axes    = len(axes)
    n_samples = len(raw_data) // n_axes
    data      = np.array(raw_data, dtype=float).reshape(n_samples, n_axes)

    subwindow_size    = max(n_samples // N_SUBWINDOWS, MIN_SUBWINDOW_SAMPLES)
    actual_subwindows = n_samples // subwindow_size

    features = []

    for sw in range(actual_subwindows):
        start = sw * subwindow_size
        end   = start + subwindow_size
        chunk = data[start:end, :]

        for ax in range(n_axes):
            sig   = chunk[:, ax]
            rms   = float(np.sqrt(np.mean(sig**2)))
            peak  = float(np.max(np.abs(sig)))
            crest = peak / (rms + 1e-12)
            kurt  = float(kurtosis(sig, fisher=True))
            skw   = float(skew(sig))
            features.extend([rms, peak, crest, kurt, skw])

    graphs = []
    if draw_graphs:
        for ax_idx, ax_name in enumerate(axes):
            rms_curve = []
            for sw in range(actual_subwindows):
                start = sw * subwindow_size
                sig   = data[start:start + subwindow_size, ax_idx]
                rms_curve.append(float(np.sqrt(np.mean(sig**2))))
            graphs.append({
                "name": f"Rolling RMS — {ax_name}",
                "X":    list(range(actual_subwindows)),
                "y":    rms_curve,
                "type": "line"
            })

    return { "features": features, "graphs": graphs }
