import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def calibration_by_bins(y_true, y_pred, n_bins=10, method='quantile', show_ci=True,
                        log_transform=False, use_log1p=True, eps=1e-12):
    """
    Return aggregated DataFrame for calibration plot:
      - mean_pred: mean predicted vol in bin (original scale)
      - mean_true: mean realized vol in bin (original scale)
      - count: samples per bin
      - se_true: standard error of realized vol in bin (original scale)
    method: 'quantile' (equal population per bin) or 'uniform' (fixed-width bins)

    log_transform: if True, binning is performed on the log-transformed predictions.
                   Aggregates (means, std, se) are computed on the original scale.
    use_log1p: whether to use log1p/expm1 pair (recommended if zeros possible).
    eps: small constant added when using plain log to avoid log(0).
    """
    df = pd.DataFrame({'y_true': np.ravel(y_true), 'y_pred': np.ravel(y_pred)})
    # Determine the column used for binning (transformed or raw predicted)
    if log_transform:
        if use_log1p:
            df['y_pred_for_bin'] = np.log1p(df['y_pred'])
        else:
            df['y_pred_for_bin'] = np.log(df['y_pred'] + eps)
    else:
        df['y_pred_for_bin'] = df['y_pred']

    if method == 'quantile':
        df['bin'] = pd.qcut(df['y_pred_for_bin'], q=n_bins, duplicates='drop')
    else:
        df['bin'] = pd.cut(df['y_pred_for_bin'], bins=n_bins)

    # Aggregate on original-scale columns so returned means are easy to interpret
    agg = df.groupby('bin').agg(
        mean_pred=('y_pred', 'mean'),
        mean_true=('y_true', 'mean'),
        count=('y_true', 'size'),
        std_true=('y_true', 'std')
    ).reset_index(drop=True)

    agg['se_true'] = agg['std_true'] / np.sqrt(agg['count'])
    # also keep the mean of the binning key (useful for diagnostics)
    agg['mean_pred_for_bin'] = df.groupby('bin')['y_pred_for_bin'].mean().values
    return agg

def plot_calibration(agg, figsize=(6,6), show_counts=True, show_ci=True):
    x = agg['mean_pred']
    y = agg['mean_true']
    counts = agg['count']

    fig, ax = plt.subplots(figsize=figsize)
    # identity line
    mn = min(x.min(), y.min())
    mx = max(x.max(), y.max())
    ax.plot([mn, mx], [mn, mx], '--', color='gray', label='y=x')

    # points sized by count
    sns.scatterplot(x=x, y=y, size=counts, sizes=(20, 300), ax=ax, legend=False)

    # error bars (CI on mean of y_true)
    if show_ci and 'se_true' in agg.columns:
        ax.errorbar(x, y, yerr=agg['se_true']*1.96, fmt='none', ecolor='gray', alpha=0.6)

    ax.set_xlabel('Mean predicted vol (bin)')
    ax.set_ylabel('Mean realized vol (bin)')
    ax.set_title('Calibration: predicted vs realized (binned)')
    if show_counts:
        # annotate counts
        for xi, yi, ci in zip(x, y, counts):
            ax.annotate(int(ci), (xi, yi), textcoords="offset points", xytext=(5, -5), fontsize=8)
    plt.tight_layout()
    return fig, ax

# Example usage:
# agg = calibration_by_bins(y_test, y_pred, n_bins=10, method='quantile', log_transform=True)
# fig, ax = plot_calibration(agg)
# plt.show()