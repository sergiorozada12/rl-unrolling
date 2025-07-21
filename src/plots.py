import matplotlib.pyplot as plt
import numpy as np


def plot_policy_and_value(
    q,
    Pi,
    shape,
    goal_pos,
    cliff_row=None,
    cliff_cols=None,
    min_prob=0.02,
    plot_all_trans=False
):
    """
    Plot policy and value function for gridworld-like environment.

    Args:
        q: Tensor of shape [nS, nA] — action-value function.
        Pi: Tensor of shape [nS, nA] — policy.
        shape: (rows, cols) — grid dimensions.
        goal_pos: (r, c) — position of the goal state.
        cliff_row: int — row index of cliff.
        cliff_cols: iterable[int] — column indices of cliff cells.
        min_prob: float — minimum prob threshold for soft arrows.
        plot_all_trans: bool — draw all action probs if True.
    """
    q = q.detach().cpu().numpy()
    Pi = Pi.detach().cpu().numpy()
    nS, nA = Pi.shape
    rows, cols = shape

    assert nS == rows * cols, "State count does not match grid shape"

    V = q.max(axis=1).reshape(rows, cols)
    greedy_actions = Pi.argmax(axis=1).reshape(rows, cols)

    # Infer cliff cells from row and column range
    cliff_cells = []
    if cliff_row is not None and cliff_cols is not None:
        cliff_cells = [(cliff_row, c) for c in cliff_cols]

    # Mask goal and cliff for color map
    V_masked = V.copy()
    for (r, c) in cliff_cells:
        V_masked[r, c] = float('nan')
    V_masked = np.ma.masked_invalid(V_masked)

    action_arrows = {
        0: (0, 1),   # →
        1: (1, 0),   # ↓
        2: (0, -1),  # ←
        3: (-1, 0),  # ↑
    }

    fig, ax = plt.subplots(figsize=(cols * 0.6, rows * 0.6))
    im = ax.imshow(V_masked, cmap="viridis")

    # Draw cliff and goal
    for (r, c) in cliff_cells:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color="black"))

    # Draw policy arrows
    max_lw = 3.5
    min_lw = 1
    for s in range(nS):
        r, c = divmod(s, cols)
        if (r, c) in cliff_cells or (r, c) == goal_pos:
            continue

        if plot_all_trans:
            for a, prob in enumerate(Pi[s]):
                if prob < min_prob:
                    continue
                dx, dy = action_arrows[a]
                ax.arrow(
                    c, r, dx * 0.3, -dy * 0.3,
                    linewidth=min_lw + (max_lw - min_lw) * prob,
                    alpha=prob,
                    head_width=0.2, head_length=0.2,
                    fc="white", ec="white")
        else:
            a = greedy_actions[r, c]
            dx, dy = action_arrows[a]
            ax.arrow(
                c, r, dx * 0.3, -dy * 0.3,
                head_width=0.2, head_length=0.2,
                fc='white', ec='white')

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("Mean Max Trans Prob = {:.3f}".format(Pi.max(axis=1).mean()))
    fig.colorbar(im, ax=ax, label='Value')

    return fig


def plot_Pi(Pi, figsize=(12, 4), title='Prob'):
    fig, ax = plt.subplots(figsize=figsize)
    cax = ax.imshow(Pi, aspect='auto')
    fig.colorbar(cax, ax=ax)

    ax.set_xlabel("Next (a')")
    ax.set_ylabel("Current (s)")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_Pi_train(list_Pi, ncols=5, freq_plots=10, figsize_per_plot=(12, 4)):
    num = len(list_Pi)
    nrows = int(np.ceil(num / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(figsize_per_plot[0]*ncols, figsize_per_plot[1]*nrows))

    # Flatten axes array for easy indexing
    axes = np.array(axes).reshape(-1)

    for i, Pi in enumerate(list_Pi):
        ax = axes[i]

        cax = ax.imshow(Pi, aspect='auto')
        fig.colorbar(cax, ax=ax)
        ax.set_title(fr'Prob (step {i*freq_plots})')
        ax.set_xlabel("Next (a')")
        ax.set_ylabel("Current (s)")

    # Hide unused subplots if any
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    return fig


def plot_filter_coefs(h_values):
    fig = plt.figure()
    plt.stem(range(h_values.size), h_values)
    plt.title("Shared filter coefficients h")
    plt.xlabel("k")
    plt.ylabel("h[k]")
    plt.grid(True)
    plt.tight_layout()
    return fig
            