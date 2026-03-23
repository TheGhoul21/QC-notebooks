"""Shared visualization utilities for the QC-Intro notebook series.

Provides functions for Bloch spheres, amplitude bar charts,
interference arrow diagrams, and circuit state evolution plots.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import Axes3D

# ── Color palette (consistent with Visual QFT series) ────────────────────────

EDGE_CMAP = plt.cm.tab10
PRIMARY_COLOR = "steelblue"
ACCENT_COLOR = "crimson"
GHOST_COLOR = "lightgray"
BG_COLOR = "white"

# ── Matplotlib defaults ──────────────────────────────────────────────────────

MPL_DEFAULTS = {'figure.dpi': 120, 'figure.facecolor': 'white'}


def apply_style():
    """Apply consistent matplotlib style for all notebooks."""
    plt.rcParams.update(MPL_DEFAULTS)


# ── Phase ↔ Color ────────────────────────────────────────────────────────────

def phase_to_color(phase):
    """Map a phase angle in [-π, π] to an HSV color.

    0 = red, π/2 = green-yellow, π = cyan, -π/2 = magenta.
    """
    hue = (phase / (2 * np.pi)) % 1.0
    return mcolors.hsv_to_rgb([hue, 0.85, 0.9])


def phase_colorbar(ax, label="Phase"):
    """Add a phase-wheel colorbar to an axes."""
    import matplotlib.cm as cm
    from matplotlib.colors import Normalize
    norm = Normalize(vmin=-np.pi, vmax=np.pi)
    sm = cm.ScalarMappable(cmap=plt.cm.hsv, norm=norm)
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, label=label, ticks=[-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
    cb.ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'$0$', r'$\pi/2$', r'$\pi$'])
    return cb


# ── Task 2: Core visualization functions ─────────────────────────────────────

def draw_bloch_sphere(theta, phi, ax=None, title=None, show_axes_labels=True):
    """Draw a Bloch sphere with a state vector at (theta, phi).

    Parameters
    ----------
    theta : float  — polar angle [0, π]
    phi : float    — azimuthal angle [0, 2π]
    ax : Axes3D or None — if None, creates a new figure
    title : str or None
    show_axes_labels : bool

    Returns
    -------
    ax : Axes3D
    """
    if ax is None:
        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(111, projection='3d')

    # Wireframe sphere
    u = np.linspace(0, 2 * np.pi, 60)
    v = np.linspace(0, np.pi, 30)
    xs = np.outer(np.cos(u), np.sin(v))
    ys = np.outer(np.sin(u), np.sin(v))
    zs = np.outer(np.ones_like(u), np.cos(v))
    ax.plot_surface(xs, ys, zs, alpha=0.05, color='lightblue')

    # Equator + meridians
    ax.plot(np.cos(u), np.sin(u), np.zeros_like(u), color=GHOST_COLOR, lw=0.5)
    ax.plot(np.zeros_like(u), np.sin(u), np.cos(u), color=GHOST_COLOR, lw=0.5)
    ax.plot(np.sin(u), np.zeros_like(u), np.cos(u), color=GHOST_COLOR, lw=0.5)

    # Axes
    for axis, label in zip(np.eye(3), ['X', 'Y', 'Z']):
        ax.plot(*[[-a, a] for a in axis], color='gray', lw=0.5)
        if show_axes_labels:
            ax.text(*(axis * 1.15), label, fontsize=9, ha='center', color='gray')

    # |0⟩ and |1⟩ poles
    ax.text(0, 0, 1.2, r'$|0\rangle$', fontsize=10, ha='center')
    ax.text(0, 0, -1.25, r'$|1\rangle$', fontsize=10, ha='center')

    # State vector
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)
    ax.quiver(0, 0, 0, x, y, z, color=ACCENT_COLOR, arrow_length_ratio=0.1, lw=2)
    ax.scatter([x], [y], [z], color=ACCENT_COLOR, s=40, zorder=10)

    # Formatting
    ax.set_xlim([-1.3, 1.3])
    ax.set_ylim([-1.3, 1.3])
    ax.set_zlim([-1.3, 1.3])
    ax.set_box_aspect([1, 1, 1])
    ax.axis('off')
    if title:
        ax.set_title(title, fontsize=11, pad=10)

    return ax


def draw_amplitude_bars(statevector, ax=None, title=None, show_phase_colorbar=True):
    """Bar chart of a statevector: height = |amplitude|, color = phase.

    Parameters
    ----------
    statevector : array-like of complex
    ax : Axes or None
    title : str or None
    show_phase_colorbar : bool

    Returns
    -------
    ax : Axes
    """
    sv = np.asarray(statevector, dtype=complex)
    n = len(sv)
    labels = [f'|{i}⟩' for i in range(n)]

    magnitudes = np.abs(sv)
    phases = np.angle(sv)
    colors = [phase_to_color(p) if m > 1e-10 else (0.85, 0.85, 0.85)
              for m, p in zip(magnitudes, phases)]

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(n * 0.8, 4), 4))

    bars = ax.bar(range(n), magnitudes, color=colors, edgecolor='gray', lw=0.5)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('|amplitude|', fontsize=10)
    ax.set_ylim(0, max(magnitudes.max() * 1.15, 0.1))

    if title:
        ax.set_title(title, fontsize=11)

    if show_phase_colorbar and n <= 32:
        phase_colorbar(ax)

    return ax


def simulate_measurement(statevector, n_shots, ax=None, title=None):
    """Monte Carlo measurement simulation with histogram.

    Parameters
    ----------
    statevector : array-like of complex
    n_shots : int
    ax : Axes or None
    title : str or None

    Returns
    -------
    counts : dict, ax : Axes
    """
    sv = np.asarray(statevector, dtype=complex)
    probs = np.abs(sv) ** 2
    probs = probs / probs.sum()
    n = len(sv)

    outcomes = np.random.choice(n, size=n_shots, p=probs)
    counts = {i: int((outcomes == i).sum()) for i in range(n) if (outcomes == i).sum() > 0}

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(n * 0.8, 4), 4))

    labels = [f'|{i}⟩' for i in range(n)]
    freqs = [counts.get(i, 0) / n_shots for i in range(n)]
    ax.bar(range(n), freqs, color=PRIMARY_COLOR, alpha=0.7, label='Measured')
    ax.bar(range(n), probs, color=ACCENT_COLOR, alpha=0.3, width=0.4, label='Theory')
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.legend(fontsize=9)

    if title:
        ax.set_title(title, fontsize=11)

    return counts, ax


def draw_complex_plane(alpha, beta, ax=None, title=None):
    """Plot two complex amplitudes in the complex plane with normalization circle.

    Parameters
    ----------
    alpha, beta : complex
    ax : Axes or None
    title : str or None

    Returns
    -------
    ax : Axes
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 5))

    # Unit circle
    t = np.linspace(0, 2 * np.pi, 100)
    ax.plot(np.cos(t), np.sin(t), color=GHOST_COLOR, lw=1, ls='--')

    # Amplitudes as arrows from origin
    for amp, label, color in [(alpha, r'$\alpha$', 'steelblue'),
                               (beta, r'$\beta$', 'darkorange')]:
        ax.annotate("", xy=(amp.real, amp.imag), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=2))
        ax.plot(amp.real, amp.imag, 'o', color=color, markersize=8, zorder=10)
        offset = 0.08 * np.exp(1j * np.angle(amp))
        ax.text(amp.real + offset.real, amp.imag + offset.imag, label,
                fontsize=12, color=color, ha='center')

    # Axes
    lim = max(abs(alpha), abs(beta), 1.0) * 1.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.3)
    ax.axvline(0, color='gray', lw=0.3)
    ax.set_xlabel('Re', fontsize=10)
    ax.set_ylabel('Im', fontsize=10)

    # Probability annotation
    ax.text(0.02, 0.98, f'|α|² = {abs(alpha)**2:.3f}\n|β|² = {abs(beta)**2:.3f}',
            transform=ax.transAxes, fontsize=9, va='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    if title:
        ax.set_title(title, fontsize=11)

    return ax


# ── Task 3: Interference and evolution functions ──────────────────────────────

def draw_interference_arrows(amplitudes, ax=None, title=None, show_sum=True):
    """Draw complex amplitudes as tip-to-tail arrows, showing interference.

    Parameters
    ----------
    amplitudes : array-like of complex
    ax : Axes or None
    title : str or None
    show_sum : bool — mark the endpoint (total sum)

    Returns
    -------
    ax : Axes
    """
    amps = np.asarray(amplitudes, dtype=complex)
    n = len(amps)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    vertices = np.zeros(n + 1, dtype=complex)
    vertices[1:] = np.cumsum(amps)

    for k in range(n):
        start = vertices[k]
        end = vertices[k + 1]
        dx = end - start
        if abs(dx) < 1e-12:
            continue
        c = EDGE_CMAP(k / max(n - 1, 1))
        ax.annotate("", xy=(end.real, end.imag), xytext=(start.real, start.imag),
                    arrowprops=dict(arrowstyle="-|>", color=c, lw=2, mutation_scale=18))
        mid = (start + end) / 2
        ax.text(mid.real, mid.imag + 0.05, f'$a_{k}$', fontsize=9, ha='center', color=c)

    ax.plot(0, 0, 'ko', markersize=6, zorder=6)

    if show_sum:
        total = vertices[-1]
        ax.plot(total.real, total.imag, 's', color=ACCENT_COLOR, markersize=10, zorder=10)
        ax.text(total.real + 0.05, total.imag + 0.05,
                f'sum = {abs(total):.2f}', fontsize=9, color=ACCENT_COLOR)

    all_pts = vertices
    lim = max(np.max(np.abs(all_pts.real)), np.max(np.abs(all_pts.imag)), 0.5) * 1.3
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.axhline(0, color='gray', lw=0.3)
    ax.axvline(0, color='gray', lw=0.3)

    if title:
        ax.set_title(title, fontsize=11)

    return ax


def draw_circuit_amplitude_evolution(stages, figsize=None):
    """Show amplitude bar charts at each stage of a circuit.

    Parameters
    ----------
    stages : list of (str, array-like)
        Each element is (label, statevector) for one circuit stage.
    figsize : tuple or None

    Returns
    -------
    fig, axes
    """
    n_stages = len(stages)
    if figsize is None:
        n_amps = len(stages[0][1])
        figsize = (max(n_amps * 0.6, 3) * n_stages, 4)

    fig, axes = plt.subplots(1, n_stages, figsize=figsize, sharey=True)
    if n_stages == 1:
        axes = [axes]

    for ax, (label, sv) in zip(axes, stages):
        draw_amplitude_bars(sv, ax=ax, title=label, show_phase_colorbar=False)

    plt.tight_layout()
    return fig, axes


def draw_phase_pattern(statevector, ax=None, title=None):
    """Display amplitudes as a bar chart where bar color = phase, height = magnitude,
    and phase value is annotated on each bar.

    Parameters
    ----------
    statevector : array-like of complex
    ax : Axes or None
    title : str or None

    Returns
    -------
    ax : Axes
    """
    sv = np.asarray(statevector, dtype=complex)
    n = len(sv)
    magnitudes = np.abs(sv)
    phases = np.angle(sv)

    if ax is None:
        fig, ax = plt.subplots(figsize=(max(n * 0.8, 4), 4))

    colors = [phase_to_color(p) if m > 1e-10 else (0.85, 0.85, 0.85)
              for m, p in zip(magnitudes, phases)]

    ax.bar(range(n), magnitudes, color=colors, edgecolor='gray', lw=0.5)

    for i, (m, p) in enumerate(zip(magnitudes, phases)):
        if m > 1e-10:
            ax.text(i, m + 0.02, f'{p:.1f}', ha='center', fontsize=7, color='gray')

    ax.set_xticks(range(n))
    ax.set_xticklabels([f'|{i}⟩' for i in range(n)], fontsize=9)
    ax.set_ylabel('|amplitude|', fontsize=10)

    if title:
        ax.set_title(title, fontsize=11)

    phase_colorbar(ax, label='Phase')
    return ax


# ── Task 4: Gate Visualization + Grover + Deutsch-Jozsa ──────────────────────

def draw_gate_on_bloch(gate_matrix, theta_in, phi_in, ax=None, title=None):
    """Show a single-qubit gate's effect on the Bloch sphere: before and after.

    Parameters
    ----------
    gate_matrix : 2x2 array — the unitary gate
    theta_in, phi_in : float — input state angles
    ax : Axes3D or None — if None, creates a 1×2 figure
    title : str or None

    Returns
    -------
    fig, (ax_before, ax_after)
    """
    state_in = np.array([np.cos(theta_in / 2),
                         np.exp(1j * phi_in) * np.sin(theta_in / 2)])
    state_out = gate_matrix @ state_in

    theta_out = 2 * np.arccos(np.clip(np.abs(state_out[0]), 0, 1))
    if np.abs(state_out[1]) > 1e-10:
        phi_out = np.angle(state_out[1]) - np.angle(state_out[0]) if np.abs(state_out[0]) > 1e-10 else np.angle(state_out[1])
    else:
        phi_out = 0.0

    fig = plt.figure(figsize=(10, 5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')

    draw_bloch_sphere(theta_in, phi_in, ax=ax1, title='Before')
    draw_bloch_sphere(theta_out, phi_out, ax=ax2, title='After')

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    plt.tight_layout()
    return fig, (ax1, ax2)


def draw_grover_iteration(n_qubits, target, n_iters, ax=None):
    """Show amplitude bars after each Grover iteration.

    Parameters
    ----------
    n_qubits : int (2-5)
    target : int — the marked state index
    n_iters : int — number of Grover iterations
    ax : Axes or None — if None, creates a 1×(n_iters+1) figure

    Returns
    -------
    fig, axes, amplitudes_history : list of arrays
    """
    N = 2 ** n_qubits
    amps = np.ones(N, dtype=complex) / np.sqrt(N)

    history = [amps.copy()]
    for _ in range(n_iters):
        amps[target] *= -1
        mean = amps.mean()
        amps = 2 * mean - amps
        history.append(amps.copy())

    n_plots = len(history)
    fig, axes = plt.subplots(1, n_plots, figsize=(3 * n_plots, 4), sharey=True)
    if n_plots == 1:
        axes = [axes]

    for i, (ax_i, state) in enumerate(zip(axes, history)):
        mags = np.abs(state)
        colors = [ACCENT_COLOR if j == target else PRIMARY_COLOR for j in range(N)]
        ax_i.bar(range(N), mags, color=colors, edgecolor='gray', lw=0.5)
        ax_i.set_title(f'Iter {i}', fontsize=10)
        ax_i.set_xticks(range(N))
        ax_i.set_xticklabels([f'|{j}⟩' for j in range(N)], fontsize=7, rotation=45)
        ax_i.axhline(1 / np.sqrt(N), color='gray', ls='--', lw=0.5)

    axes[0].set_ylabel('|amplitude|', fontsize=10)
    plt.tight_layout()
    return fig, axes, history


def draw_grover_2d_plane(n_qubits, target, n_iters, ax=None):
    """Geometric 2D interpretation of Grover: rotations in |target⟩, |rest⟩ plane.

    Parameters
    ----------
    n_qubits : int
    target : int
    n_iters : int
    ax : Axes or None

    Returns
    -------
    ax
    """
    N = 2 ** n_qubits
    theta_0 = np.arcsin(1 / np.sqrt(N))
    delta = 2 * theta_0

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))

    t = np.linspace(0, np.pi / 2, 100)
    ax.plot(np.cos(t), np.sin(t), color=GHOST_COLOR, lw=1, ls='--')

    ax.set_xlabel(r'$|\mathrm{rest}\rangle$ component', fontsize=11)
    ax.set_ylabel(r'$|\mathrm{target}\rangle$ component', fontsize=11)

    for i in range(n_iters + 1):
        angle = (2 * i + 1) * theta_0
        x, y = np.cos(angle), np.sin(angle)
        alpha = 0.3 + 0.7 * (i / max(n_iters, 1))
        ax.annotate("", xy=(x, y), xytext=(0, 0),
                    arrowprops=dict(arrowstyle="-|>", color=ACCENT_COLOR,
                                    lw=2, alpha=alpha))
        ax.text(x + 0.03, y + 0.03, f'{i}', fontsize=9, color=ACCENT_COLOR, alpha=alpha)

    ax.plot([0, 0], [0, 1], color='gray', lw=0.5)
    ax.plot([0, 1], [0, 0], color='gray', lw=0.5)

    ax.set_xlim(-0.1, 1.15)
    ax.set_ylim(-0.1, 1.15)
    ax.set_aspect('equal')
    ax.set_title(f'Grover rotation: {n_qubits} qubits, {n_iters} iters\n'
                 f'θ₀ = {theta_0:.3f}, Δ = {delta:.3f}', fontsize=10)

    return ax


def draw_circuit_step_by_step(gates, initial_state, labels=None):
    """Show state evolution through a sequence of single-qubit gates.

    Parameters
    ----------
    gates : list of 2x2 arrays — the gate matrices in order
    initial_state : array of 2 complex — the initial statevector
    labels : list of str or None — gate names for titles

    Returns
    -------
    fig, axes
    """
    states = [np.array(initial_state, dtype=complex)]
    for g in gates:
        states.append(g @ states[-1])

    if labels is None:
        labels = [f'Gate {i+1}' for i in range(len(gates))]
    stage_labels = ['Initial'] + [f'After {l}' for l in labels]

    n_stages = len(states)
    fig, axes = plt.subplots(2, n_stages, figsize=(4 * n_stages, 8))

    for i, (state, label) in enumerate(zip(states, stage_labels)):
        # Top row: amplitude bars
        draw_amplitude_bars(state, ax=axes[0, i], title=label, show_phase_colorbar=False)

        # Bottom row: Bloch sphere
        theta = 2 * np.arccos(np.clip(np.abs(state[0]), 0, 1))
        phi = np.angle(state[1]) - np.angle(state[0]) if np.abs(state[0]) > 1e-10 and np.abs(state[1]) > 1e-10 else 0.0
        ax3d = fig.add_subplot(2, n_stages, n_stages + i + 1, projection='3d')
        draw_bloch_sphere(theta, phi, ax=ax3d)
        axes[1, i] = ax3d

    plt.tight_layout()
    return fig, axes


def draw_unitary_decomposition(U, title=None):
    """Decompose a 1- or 2-qubit unitary into gates and show the circuit.

    Parameters
    ----------
    U : 2x2 or 4x4 array — the unitary matrix
    title : str or None

    Returns
    -------
    fig, (ax_matrix, ax_circuit)
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator
    from qiskit.compiler import transpile

    U = np.array(U, dtype=complex)
    n_qubits = 1 if U.shape[0] == 2 else 2

    qc = QuantumCircuit(n_qubits)
    qc.unitary(Operator(U), list(range(n_qubits)))

    decomposed = transpile(qc, basis_gates=['rx', 'ry', 'rz', 'cx'], optimization_level=2)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.imshow(np.abs(U), cmap='Blues', vmin=0, vmax=1)
    for i in range(U.shape[0]):
        for j in range(U.shape[1]):
            val = U[i, j]
            ax1.text(j, i, f'{val:.2f}', ha='center', va='center', fontsize=8)
    ax1.set_title('Target unitary', fontsize=11)
    ax1.set_xticks(range(U.shape[1]))
    ax1.set_yticks(range(U.shape[0]))

    decomposed.draw('mpl', ax=ax2)
    ax2.set_title(f'Decomposition ({decomposed.count_ops()} gates)', fontsize=11)

    if title:
        fig.suptitle(title, fontsize=13, y=1.02)

    plt.tight_layout()
    return fig, (ax1, ax2)


def draw_dj_step_by_step(f_type, n_qubits):
    """Show amplitude evolution through a Deutsch-Jozsa circuit.

    Parameters
    ----------
    f_type : str — 'constant_0', 'constant_1', or 'balanced'
    n_qubits : int — number of input qubits (total circuit = n_qubits + 1 ancilla)

    Returns
    -------
    fig, axes, stages : list of (label, statevector)
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    n = n_qubits
    N = 2 ** n

    qc = QuantumCircuit(n + 1)

    qc.x(n)
    sv0 = Statevector.from_instruction(qc)

    qc.h(range(n + 1))
    sv1 = Statevector.from_instruction(qc)

    if f_type == 'constant_0':
        pass
    elif f_type == 'constant_1':
        qc.x(n)
    elif f_type == 'balanced':
        qc.cx(0, n)
    sv2 = Statevector.from_instruction(qc)

    qc.h(range(n))
    sv3 = Statevector.from_instruction(qc)

    stages = [
        ('Initial: |0⟩⊗|1⟩', sv0.data),
        ('After H⊗(n+1)', sv1.data),
        (f'After oracle ({f_type})', sv2.data),
        ('After final H⊗n', sv3.data),
    ]

    fig, axes = draw_circuit_amplitude_evolution(stages)

    return fig, axes, stages
