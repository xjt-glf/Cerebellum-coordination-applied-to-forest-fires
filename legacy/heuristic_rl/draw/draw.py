import os
import pandas as pd
import matplotlib.pyplot as plt


def load_series(path):
	df = pd.read_csv(path, header=0)
	# prefer 'fusion_metric' if present
	if 'fusion_metric' in df.columns:
		y = pd.to_numeric(df['fusion_metric'], errors='coerce').dropna().reset_index(drop=True)
	else:
		# try several common names
		for col in ('value', 'reward', 'step_reward', 'y', 'score'):
			if col in df.columns:
				y = pd.to_numeric(df[col], errors='coerce').dropna().reset_index(drop=True)
				break
		else:
			# fallback: find first numeric column that is not 'episode'
			y = None
			for col in df.columns:
				if col == 'episode':
					continue
				if pd.api.types.is_numeric_dtype(df[col]):
					y = pd.to_numeric(df[col], errors='coerce').dropna().reset_index(drop=True)
					break
			if y is None:
				# last resort: take first column parsed as numbers
				y = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().reset_index(drop=True)

	# x values: use 'episode' if present, otherwise use 1-based index
	if 'episode' in df.columns:
		x = pd.to_numeric(df['episode'], errors='coerce').dropna().reset_index(drop=True)
	else:
		x = pd.Series(range(1, len(y) + 1))

	return x, y


def ma(series, window=20):
	return series.rolling(window, min_periods=1).mean()



# legacy helper, kept for compatibility but not used for the new task
# (it plots a single fusion metric with a fixed rolling window)
def plot_three(files, labels=None, out_path=None):
    data = []
    for f in files:
        x, y = load_series(f)
        data.append((x, y))

    plt.figure(figsize=(10, 6))
    x_max = 1200
    styles = [dict(color='#1f77b4', lw=1.5), dict(color='#ff7f0e', lw=1.5), dict(color='#2ca02c', lw=1.5)]

    for idx, (x, y) in enumerate(data):
        y_ma = ma(y, window=20)
        # take first 1200 points by position
        y_plot = y_ma.iloc[:x_max]
        x_plot = x.iloc[:len(y_plot)]
        if x_plot.empty:
            x_plot = pd.Series(range(1, len(y_plot) + 1))
        label = labels[idx] if labels and idx < len(labels) else os.path.splitext(os.path.basename(files[idx]))[0]
        plt.plot(x_plot, y_plot, label=label, **styles[idx % len(styles)])

    plt.xlabel('Episode')
    plt.ylabel('Fusion Metric')
    plt.title('Fusion Metric')
    plt.legend()
    plt.grid(alpha=0.3)

    if out_path:
        plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.show()


def smooth(series, window=50):
    """Return a rolling mean with specified window (min_periods=1)."""
    return series.rolling(window, min_periods=1).mean()


def plot_two_files(file1, file2):
    """Create two figures comparing avg_coverage, avg_explore and max_aoi.

    Each metric is drawn twice per series: raw (high transparency) and smoothed
    with a window of 50 (opaque) so that the smoothed data is highlighted.
    """
    df1 = pd.read_csv(file1)
    df2 = pd.read_csv(file2)

    # ensure we have an episode column
    x1 = df1['episode'] if 'episode' in df1.columns else pd.Series(range(len(df1)))
    x2 = df2['episode'] if 'episode' in df2.columns else pd.Series(range(len(df2)))

    colors = ('#1f77b4', '#ff7f0e')
    labels = ('PRS-HRL', 'Ablation')

    # first figure: avg_coverage and avg_explore
    plt.figure(figsize=(6, 4))

    # plt.rcParams.update({
    # "font.size": 14,
    # # "axes.titlesize": 16,
    # # "axes.labelsize": 10,
    # # "xtick.labelsize": 12,
    # # "ytick.labelsize": 12,
    # # "legend.fontsize": 12,
    # })
    for df, label, color in ((df1, labels[0], colors[0]), (df2, labels[1], colors[1])):
        x = df['episode'] if 'episode' in df.columns else pd.Series(range(len(df)))
        cov = df['avg_coverage']
        exp = df['avg_explore']
        # plt.plot(x, cov, alpha=0.2, color=color)
        plt.plot(x, smooth(cov), color=color, label=f"{label} coverage")
        # plt.plot(x, exp, alpha=0.2, color=color, linestyle='--')
        plt.plot(x, smooth(exp), color=color, linestyle='--', label=f"{label} explore")

    plt.xlabel('Episode')
    plt.ylabel('Rate')
    plt.title('Average Coverage and Exploration Rates')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()

    # 保存为PDF矢量图
    plt.savefig("coverage_explore_curve_2.pdf", format="pdf", bbox_inches="tight")

    plt.show()

    # second figure: max_aoi
    plt.figure(figsize=(10, 6))
    for df, label, color in ((df1, labels[0], colors[0]), (df2, labels[1], colors[1])):
        x = df['episode'] if 'episode' in df.columns else pd.Series(range(len(df)))
        aoi = df['max_aoi']
        plt.plot(x, aoi, alpha=0.2, color=color)
        plt.plot(x, smooth(aoi), color=color, label=f"{label} max_aoi")

    plt.xlabel('Episode')
    plt.ylabel('Max AOI')
    plt.title('Maximum AOI Comparison')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.show()


def main():
    base = os.path.dirname(__file__)
    f1 = os.path.join(base, 'shuju', 'training_data_topology.csv')
    f2 = os.path.join(base, 'shuju', 'training_without_qiehuan.csv')
    plot_two_files(f1, f2)


if __name__ == '__main__':
	main()

