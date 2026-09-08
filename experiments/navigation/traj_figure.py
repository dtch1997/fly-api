import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
AMBER, INK, EYE = '#B47B12', '#55575c', '#A93B2E'
fig, ax = plt.subplots(figsize=(7, 5))
for which, color, style in [('naive', INK, '--'), ('trained', AMBER, '-')]:
    t = np.load(f'runs/film/{which}_traj.npy')
    ax.plot(t[:, 0], t[:, 1], style, color=color, lw=2, label=f'{which} brain')
    ax.plot(t[-1, 0], t[-1, 1], 'o', color=color, ms=8)
ax.plot(*[14, 7], marker='o', ms=18, color=AMBER, mec='k'); ax.annotate('odor A (rewarded)', (14, 7), xytext=(16.5, 6.6), fontsize=10)
ax.plot(*[14, -7], marker='o', ms=18, color='#8a8a8e', mec='k'); ax.annotate('odor B (control)', (14, -7), xytext=(16.5, -7.4), fontsize=10)
ax.plot(0, 0, marker='*', ms=14, color=EYE); ax.annotate(' start', (0, 0), fontsize=9)
ax.set_xlabel('x (mm)'); ax.set_ylabel('y (mm)')
ax.set_title('Same fly, same arena: only the synapses differ')
ax.annotate('keeps walking →', (40, 9.6), fontsize=9, color='#55575c')
ax.legend(frameon=False, loc='lower right'); ax.set_aspect('equal')
ax.set_xlim(-3, 42)
fig.tight_layout(); fig.savefig('figures/n1_trajectories.png', dpi=150)
print('saved figures/n1_trajectories.png')
