#!/usr/bin/env python3
"""Visualise the Stage-1 world model: emergent attack effect vs the real system."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent.parent / 'results' / 'pilot_v3_worldmodel'
m = json.loads((OUT / 'metrics.json').read_text())
ev = m['emergent_vulnerability']; tr = ev['trajectories']
real = np.array(tr['real']); corr = np.array(tr['corrupted']); clean = np.array(tr['clean'])
truth = np.array(tr['truth_aligned']); att = np.array(tr['attack_active']); first = tr['first']
sl = slice(first, len(real))

fig, ax = plt.subplots(1, 3, figsize=(15, 4.6))

# Panel 1: XY trajectories
ax[0].plot(truth[sl, 0], truth[sl, 1], color='0.6', lw=2.5, label='ground truth')
ax[0].plot(real[sl, 0], real[sl, 1], color='#111', lw=1.6, label='real slam_toolbox (attacked)')
ax[0].plot(clean[sl, 0], clean[sl, 1], color='#4C78A8', lw=1.4, label='world model (clean scans)')
ax[0].plot(corr[sl, 0], corr[sl, 1], color='#E45756', lw=1.4, label='world model (attacked scans)')
a0 = np.flatnonzero(att)
if len(a0):
    ax[0].scatter(truth[a0, 0], truth[a0, 1], s=6, color='#F58518', zorder=5, label='attack active (truth)')
ax[0].set_aspect('equal'); ax[0].set_xlabel('x (m)'); ax[0].set_ylabel('y (m)')
ax[0].set_title('Trajectories'); ax[0].legend(fontsize=7)

# Panel 2: localization error vs time
t = np.arange(len(real))
err_real = np.linalg.norm(real[:, :2] - truth[:, :2], axis=1) * 1000
err_corr = np.linalg.norm(corr[:, :2] - truth[:, :2], axis=1) * 1000
err_clean = np.linalg.norm(clean[:, :2] - truth[:, :2], axis=1) * 1000
for y, c, l in [(err_real, '#111', 'real slam (attacked)'), (err_corr, '#E45756', 'world model (attacked)'), (err_clean, '#4C78A8', 'world model (clean)')]:
    ax[1].plot(t[sl], y[sl], color=c, lw=1.4, label=l)
if len(a0):
    ax[1].axvspan(a0.min(), a0.max(), color='#F58518', alpha=0.12, label='attack window')
ax[1].set_xlabel('scan index'); ax[1].set_ylabel('localization error vs truth (mm)')
ax[1].set_title('Error over time'); ax[1].legend(fontsize=7)

# Panel 3: predicted attack effect by phase, world model vs physics vs real growth
phases = ['nominal', 'attack', 'recovery']
wm_eff = [ev['worldmodel_predicted_attack_effect_m'].get(p, 0) * 1000 for p in phases]
base = ev['real_slam_vs_truth']['nominal']['rmse_m'] * 1000
real_growth = [max(0, ev['real_slam_vs_truth'].get(p, {'rmse_m': base / 1000})['rmse_m'] * 1000 - base) for p in phases]
x = np.arange(3); w = 0.28
ax[2].bar(x - w, real_growth, w, color='#111', label='real slam error growth')
ax[2].bar(x, wm_eff, w, color='#E45756', label='world model predicted effect')
ax[2].bar(x + w, [0, 0, 0], w, color='#4C78A8', label='calibrated physics (= 0)')
ax[2].set_xticks(x); ax[2].set_xticklabels(phases)
ax[2].set_ylabel('attack effect (mm)'); ax[2].set_title('Attack effect: emergent vs physics')
ax[2].legend(fontsize=7)

fig.tight_layout(); fig.savefig(OUT / 'worldmodel_emergent.png', dpi=130)
print('wrote', OUT / 'worldmodel_emergent.png')
