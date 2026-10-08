#!/usr/bin/env python3
"""Stage 1: an absolute-pose, drift-correcting recurrent SLAM world model.

Why this and not v1/v2. Nominal analysis showed slam_toolbox's scan dependence
lives in the ABSOLUTE/accumulated pose, not the per-step increment: calibrated
command integration drifts 0.4-4.8 m over a 60 s run while slam_toolbox stays
locked to ground truth within ~5 cm. v1/v2 regressed the per-step increment,
where the command already suffices to ~mm, so they learned a command shortcut,
ignored the scan, and were not vulnerable to scan attacks.

This model instead maintains an absolute pose estimate and a recurrent memory
(the "map"), and must use the scan to correct the drifting command prior. It is
trained by rolling its OWN pose across a window (no per-step teacher forcing), so
without a scan-driven correction the estimate drifts and the absolute-pose loss is
large -- this structurally forces scan dependence, the way real SLAM depends on
scan matching. The correction head sees only scan-derived features, never the
command, so a corrupted scan mechanically corrupts localization.

Training is NOMINAL ONLY. The attack is never seen in training; it is applied at
inference as an input perturbation. The question is whether the vulnerability
EMERGES, as it would in a faithful physics simulator -- not whether it was learned.
"""
import argparse, json, math, random, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader

CORR = np.array([.05, .05, .10], dtype=np.float32)   # per-step correction scale (m, m, rad)
POS_SCALE, YAW_SCALE = 0.10, 0.20                     # loss normalisers


def wrapped(x):
    return torch.atan2(torch.sin(x), torch.cos(x))


def wrapped_np(x):
    return np.arctan2(np.sin(x), np.cos(x))


def body_delta_np(p, q):
    d = q - p; c = np.cos(p[..., 2]); s = np.sin(p[..., 2])
    return np.stack([c * d[..., 0] + s * d[..., 1], -s * d[..., 0] + c * d[..., 1], wrapped_np(d[..., 2])], -1)


def fit_gains(episodes):
    """Two motion gains fitted to independent Gazebo ground truth, nominal train only."""
    r = [[], []]
    for a in episodes:
        gt = body_delta_np(a['ground_truth_xyyaw'][:-1], a['ground_truth_xyyaw'][1:])
        c = a['command_vw'][:-1]; dt = np.diff(a['scan_stamp_ns']) / 1e9
        for j, k in [(0, 0), (1, 2)]:
            den = c[:, j] * dt
            m = a['command_observed'][:-1] & np.isfinite(gt[:, k]) & (np.abs(den) > .005)
            r[j].extend((gt[m, k] / den[m]).tolist())
    return [float(np.clip(np.median(x), .5, 1.5)) if x else 1. for x in r]


def command_increment(a, g):
    """Calibrated differential-drive body-frame increment per step (the drifting prior)."""
    dt = np.r_[.2, np.diff(a['scan_stamp_ns']) / 1e9].astype(np.float32)
    cmd = np.nan_to_num(a['command_vw']).astype(np.float32); obs = a['command_observed']
    v = cmd[:, 0] * g[0]; w = cmd[:, 1] * g[1]; angle = w * dt
    dx = v * dt * np.sinc(angle / np.pi)
    dy = v * dt * (angle / 2) * np.sinc(angle / (2 * np.pi)) ** 2
    inc = np.stack([dx, dy, angle], 1).astype(np.float32); inc[~obs] = 0
    return inc


def load_episode(path, gains=None):
    a = dict(np.load(path, allow_pickle=False))
    if not np.all(np.diff(a['scan_stamp_ns']) > 0):
        raise ValueError('Episode must have increasing scan timestamps')
    a['beams'] = a['scan_normalized'].shape[1]
    if gains is not None:
        a['cmd_inc'] = command_increment(a, gains)
    return a


def clean_normalized(a):
    """Normalised ranges rebuilt from the paired clean sensor scan (counterfactual input)."""
    raw = a['clean_scan_ranges']; lo = a['scan_range_min'][:, None]; hi = a['scan_range_max'][:, None]
    v = np.isfinite(raw) & (raw >= lo) & (raw <= hi)
    return np.where(v, (raw - lo) / (hi - lo), 0).astype(np.float32), v


class Windows(Dataset):
    def __init__(self, episodes, length, stride):
        self.episodes = episodes; self.length = length; self.items = []
        for e, a in enumerate(episodes):
            n = len(a['scan_normalized'])
            for s in range(0, n - length, stride):
                if a['slam_pose_valid'][s:s + length].all():
                    self.items.append((e, s))
        if not self.items:
            raise ValueError('No valid windows')

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        e, s = self.items[k]; a = self.episodes[e]; L = self.length
        return (a['scan_normalized'][s:s + L].astype(np.float32),
                a['scan_valid'][s:s + L].astype(np.float32),
                a['cmd_inc'][s:s + L].astype(np.float32),
                a['slam_pose_xyyaw'][s:s + L].astype(np.float32))


class WorldModel(nn.Module):
    """Recurrent filter: drifting command prior + scan-driven absolute correction."""

    def __init__(self, beams):
        super().__init__(); self.beams = beams
        self.register_buffer('corr', torch.tensor(CORR))
        self.encoder = nn.Sequential(nn.Linear(2 * beams, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.cell = nn.LSTMCell(128, 256)                 # memory ~ accumulated map / pose graph
        self.correction = nn.Linear(256 + 128, 3)         # sees only scan-derived features
        nn.init.zeros_(self.correction.weight); nn.init.zeros_(self.correction.bias)

    def step(self, est, scan, valid, cmd_inc, hc):
        """One filter step: advance by command prior, then correct from the scan."""
        c = torch.cos(est[:, 2]); s = torch.sin(est[:, 2])
        prior = torch.stack([
            est[:, 0] + c * cmd_inc[:, 0] - s * cmd_inc[:, 1],
            est[:, 1] + s * cmd_inc[:, 0] + c * cmd_inc[:, 1],
            wrapped(est[:, 2] + cmd_inc[:, 2])], 1)
        feat = self.encoder(torch.cat([scan, valid], 1))
        h, cell = self.cell(feat, hc)
        delta = torch.tanh(self.correction(torch.cat([h, feat], 1))) * self.corr
        est = torch.stack([prior[:, 0] + delta[:, 0], prior[:, 1] + delta[:, 1], wrapped(prior[:, 2] + delta[:, 2])], 1)
        return est, (h, cell), prior, delta

    def rollout(self, scan, valid, cmd_inc, init_pose, correct=True, teacher=None, ss_prob=0.0):
        """Run the filter over a (B,T,...) window from init_pose; return estimates (B,T,3).

        Scheduled sampling: with probability ss_prob the next step's carried pose is
        replaced by the teacher (real SLAM) pose, which stabilises early training; it
        anneals to 0 so the model ends up fully self-integrated and must handle drift.
        Returns (estimates, correction_reg) where the reg is mean squared correction.
        """
        B, T, _ = scan.shape; dev = scan.device
        est = init_pose.clone(); h = torch.zeros(B, 256, device=dev); cell = torch.zeros(B, 256, device=dev)
        outs = []; reg = 0.0
        for t in range(T):
            est, (h, cell), prior, delta = self.step(est, scan[:, t], valid[:, t], cmd_inc[:, t], (h, cell))
            if not correct:
                est = prior   # calibrated-physics-only baseline (scan ignored)
            outs.append(est)
            reg = reg + (delta ** 2).mean()
            if teacher is not None and ss_prob > 0 and t + 1 < T:
                use = (torch.rand(B, 1, device=dev) < ss_prob).float()
                est = use * teacher[:, t] + (1 - use) * est
        return torch.stack(outs, 1), reg / T


def pose_loss(pred, target):
    pos = ((pred[..., :2] - target[..., :2]) ** 2).sum(-1).mean() / POS_SCALE ** 2
    yaw = (wrapped(pred[..., 2] - target[..., 2]) ** 2).mean() / YAW_SCALE ** 2
    return pos + yaw


def track_episode(model, a, scan_norm, valid, device, correct=True):
    """Full-episode open-loop filter run (teacher init pose, then self-integrated)."""
    first = int(np.flatnonzero(a['slam_pose_valid'])[0])
    scan = torch.tensor(scan_norm[None], device=device)
    val = torch.tensor(valid.astype(np.float32)[None], device=device)
    cmd = torch.tensor(a['cmd_inc'][None], device=device)
    init = torch.tensor(a['slam_pose_xyyaw'][first][None].astype(np.float32), device=device)
    with torch.no_grad():
        est = model.rollout(scan[:, first:], val[:, first:], cmd[:, first:], init, correct=correct)[0][0].cpu().numpy()
    out = np.full((len(scan_norm), 3), np.nan, np.float32); out[first:] = est
    return out, first


def align_truth(a, ref_pose, first):
    """Map truth into the estimate frame via one SE(2) alignment at the first valid pose."""
    gt = a['ground_truth_xyyaw']; rot = ref_pose[first, 2] - gt[first, 2]
    c, s = np.cos(rot), np.sin(rot)
    return (gt[:, :2] - gt[first, :2]) @ np.array([[c, s], [-s, c]]) + ref_pose[first, :2]


def phase_errors(est, aligned_gt, a, valid_pose):
    v = valid_pose & np.isfinite(est).all(1) & np.isfinite(aligned_gt).all(1)
    err = np.linalg.norm(est[:, :2] - aligned_gt[:, :2], axis=1)
    phases = np.where(a['attack_active'], 'attack',
                      np.where(a['attack_active'].any() & (a['elapsed_seconds'] >= 40), 'recovery', 'nominal'))
    out = {}
    for ph in ['nominal', 'attack', 'recovery']:
        m = v & (phases == ph)
        if m.any():
            out[ph] = {'rmse_m': float(np.sqrt(np.mean(err[m] ** 2))), 'max_m': float(np.max(err[m]))}
    return out


def emergent_vulnerability(model, attack_ep, device):
    """Apply the (never-trained) attack as an input perturbation; does the failure emerge?

    Runs the same nominal-trained filter on (a) the corrupted delivered scans and
    (b) the paired clean scans, plus the calibrated-physics-only baseline, and
    compares localization error by phase against the REAL slam_toolbox trajectory.
    """
    a = attack_ep
    corr_est, first = track_episode(model, a, a['scan_normalized'].astype(np.float32), a['scan_valid'], device, correct=True)
    cn, cv = clean_normalized(a)
    clean_est, _ = track_episode(model, a, cn, cv, device, correct=True)
    phys_est, _ = track_episode(model, a, a['scan_normalized'].astype(np.float32), a['scan_valid'], device, correct=False)
    real = a['slam_pose_xyyaw']

    gt_corr = align_truth(a, corr_est, first)   # model (corrupted) shares SLAM's init frame
    gt_real = align_truth(a, real, first)
    vp = a['slam_pose_valid']
    result = {
        'real_slam_vs_truth': phase_errors(real, gt_real, a, vp),
        'worldmodel_corrupted_vs_truth': phase_errors(corr_est, gt_corr, a, vp),
        'worldmodel_clean_vs_truth': phase_errors(clean_est, gt_corr, a, vp),
        'physics_only_vs_truth': phase_errors(phys_est, gt_corr, a, vp),
    }
    # Model's PREDICTED attack effect = corrupted vs its own clean counterfactual run.
    eff = np.linalg.norm(corr_est[:, :2] - clean_est[:, :2], axis=1)
    phases = np.where(a['attack_active'], 'attack', np.where(a['attack_active'].any() & (a['elapsed_seconds'] >= 40), 'recovery', 'nominal'))
    vmask = vp & np.isfinite(corr_est).all(1) & np.isfinite(clean_est).all(1)
    result['worldmodel_predicted_attack_effect_m'] = {
        ph: float(np.sqrt(np.mean(eff[vmask & (phases == ph)] ** 2)))
        for ph in ['nominal', 'attack', 'recovery'] if (vmask & (phases == ph)).any()}
    result['trajectories'] = {'first': first, 'real': real.tolist(), 'corrupted': corr_est.tolist(),
                              'clean': clean_est.tolist(), 'physics': phys_est.tolist(),
                              'truth_aligned': gt_corr.tolist(), 'attack_active': a['attack_active'].tolist()}
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--length', type=int, default=48)
    p.add_argument('--stride', type=int, default=6)
    p.add_argument('--batch-size', type=int, default=32)
    p.add_argument('--device', default='cpu')
    p.add_argument('--seed', type=int, default=7)
    a = p.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    torch.set_num_threads(4); torch.manual_seed(a.seed); np.random.seed(a.seed); random.seed(a.seed)
    if a.device.startswith('cuda'):
        torch.cuda.init(); torch.cuda.set_device(a.device); torch.cuda.reset_peak_memory_stats(a.device)
    spec = json.loads(a.manifest.read_text()); root = a.manifest.parent.parent
    paths = {k: [(root / x).resolve() for x in v] for k, v in spec.items()}
    flat = [str(x) for v in paths.values() for x in v]
    if len(flat) != len(set(flat)):
        raise ValueError('Episode leakage between splits')

    raw = {k: [load_episode(x) for x in v] for k, v in paths.items()}
    for k in ['train', 'validation']:
        if any(ep.get('attack_active', np.array([False])).any() for ep in raw[k]):
            raise ValueError('Nominal training/selection only')
    gains = fit_gains(raw['train'])
    data = {k: [load_episode(x, gains) for x in v] for k, v in paths.items()}
    train_ds = Windows(data['train'], a.length, a.stride)
    val_ds = Windows(data['validation'], a.length, a.stride)
    beams = data['train'][0]['beams']

    a.output.mkdir(parents=True)
    ckpt_dir = (root / 'checkpoints' / a.output.name).resolve(); ckpt_dir.mkdir(parents=True, exist_ok=False)
    model = WorldModel(beams).to(a.device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=a.epochs)
    loader = DataLoader(train_ds, batch_size=a.batch_size, shuffle=True)
    best = float('inf'); trace = []; start = time.monotonic()
    for epoch in range(a.epochs):
        ss = max(0.0, 0.9 * (1 - epoch / (0.7 * a.epochs)))   # anneal teacher forcing 0.9 -> 0
        model.train(); tot = 0; cnt = 0
        for scan, val, cmd, pose in loader:
            scan, val, cmd, pose = [t.to(a.device) for t in [scan, val, cmd, pose]]
            est, reg = model.rollout(scan, val, cmd, pose[:, 0], correct=True, teacher=pose, ss_prob=ss)
            loss = pose_loss(est, pose) + 0.1 * reg   # parsimony: correct only when the scan demands it
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.); opt.step()
            tot += loss.item() * len(scan); cnt += len(scan)
        sched.step()
        model.eval(); vt = 0; vc = 0
        with torch.no_grad():
            for scan, val, cmd, pose in DataLoader(val_ds, batch_size=64):
                scan, val, cmd, pose = [t.to(a.device) for t in [scan, val, cmd, pose]]
                vt += pose_loss(model.rollout(scan, val, cmd, pose[:, 0])[0], pose).item() * len(scan); vc += len(scan)
        val = vt / vc; trace.append({'epoch': epoch + 1, 'train': tot / cnt, 'validation': val})
        ckpt = {'model': model.state_dict(), 'beams': beams, 'gains': gains, 'epoch': epoch + 1}
        torch.save(ckpt, ckpt_dir / 'worldmodel_latest.pt')
        if val < best:
            best = val; torch.save(ckpt, ckpt_dir / 'worldmodel_best.pt')
        if (epoch + 1) % 5 == 0:
            print(f'epoch {epoch+1} train {tot/cnt:.4f} val {val:.4f}', flush=True)
    model.load_state_dict(torch.load(ckpt_dir / 'worldmodel_best.pt', map_location=a.device, weights_only=False)['model'])

    # Nominal faithfulness: does the filter localize (kill command drift) on the held-out test episode?
    results = {'research_question': 'Does a nominal-only SLAM world model inherit the real system vulnerability to scan attacks (emergent, not trained)?',
               'approach': 'absolute-pose recurrent filter; drifting command prior + scan-only correction; trained by self-integrated rollout to force scan dependence',
               'calibration_train_only': gains, 'checkpoint_dir': str(ckpt_dir), 'manifest': spec,
               'parameters': sum(pp.numel() for pp in model.parameters()), 'seconds': time.monotonic() - start, 'trace': trace,
               'nominal_faithfulness': {}}
    for nm, ep in [('test', data['test'][0])]:
        est, first = track_episode(model, ep, ep['scan_normalized'].astype(np.float32), ep['scan_valid'], a.device, correct=True)
        phys, _ = track_episode(model, ep, ep['scan_normalized'].astype(np.float32), ep['scan_valid'], a.device, correct=False)
        gt = align_truth(ep, est, first); vp = ep['slam_pose_valid']
        results['nominal_faithfulness'][nm] = {
            'worldmodel_vs_truth': phase_errors(est, gt, ep, vp),
            'physics_only_vs_truth': phase_errors(phys, gt, ep, vp),
            'worldmodel_vs_real_slam_rmse_m': float(np.sqrt(np.nanmean(np.sum((est[vp, :2] - ep['slam_pose_xyyaw'][vp, :2]) ** 2, 1))))}
    results['emergent_vulnerability'] = emergent_vulnerability(model, data['attack_test'][0], a.device)
    results['max_cuda_memory_bytes'] = torch.cuda.max_memory_allocated(a.device) if a.device.startswith('cuda') else 0
    (a.output / 'metrics.json').write_text(json.dumps(results, indent=2))
    print('Saved', a.output, flush=True)


if __name__ == '__main__':
    main()
