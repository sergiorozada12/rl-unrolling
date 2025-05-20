import matplotlib.pyplot as plt
import numpy as np

def plot_policy_and_value(q, Pi, shape=(4, 12)):
    q = q.detach().cpu().numpy()
    Pi = Pi.detach().cpu().numpy()
    nS, nA = Pi.shape
    rows, cols = shape

    assert nS == rows * cols, "State count does not match grid shape"

    V = q.max(axis=1).reshape(rows, cols)
    greedy_actions = Pi.argmax(axis=1).reshape(rows, cols)

    action_arrows = {
        0: (0, 1),
        1: (1, 0),
        2: (0, -1),
        3: (-1, 0),
    }

    fig, ax = plt.subplots(figsize=(12, 4))
    im = ax.imshow(V, cmap='viridis')

    cliff_cells = [(3, c) for c in range(1, 11)]
    goal_cell = (3, 11)

    for r in range(rows):
        for c in range(cols):
            if (r, c) in cliff_cells:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, color='black'))
                continue
            if (r, c) == goal_cell:
                continue
            a = greedy_actions[r, c]
            dx, dy = action_arrows[a]
            ax.arrow(c, r, dx * 0.3, -dy * 0.3, head_width=0.2, head_length=0.2, fc='white', ec='white')

    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, label='Value')
    return fig
