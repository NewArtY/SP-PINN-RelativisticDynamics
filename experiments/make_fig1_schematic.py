"""Generate Figure 1: schematic of the SP-PINN two-stage architecture."""
import _paths  # noqa
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from relsim.plotstyle import apply_style, COLORS

apply_style()
plt.rcParams["axes.grid"] = False

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.set_xlim(0, 12); ax.set_ylim(0, 6); ax.axis("off")


def box(x, y, w, h, text, fc, ec="k", fs=9.5):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04,rounding_size=0.12",
                       fc=fc, ec=ec, lw=1.2, alpha=0.95)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs)


def arrow(x1, y1, x2, y2, color="k"):
    a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                        lw=1.5, color=color)
    ax.add_patch(a)


# Stage 1 (left)
ax.text(2.9, 5.6, r"Stage 1 — Hamiltonian learning (offline)",
        ha="center", fontsize=11, weight="bold")
box(0.3, 3.6, 2.3, 1.1, "Collocation points\n" r"$(\mathbf{r},\mathbf{P},t)$" "\nSobol sampling", "#EAF3FB")
box(0.3, 1.7, 2.3, 1.1, "PINN\n" r"$\mathcal{H}_\theta(\mathbf{r},\mathbf{P},t)$" "\ntanh MLP", "#E9F7F1")
box(3.1, 1.4, 2.6, 3.3,
    "Loss\n" r"$\mathcal{L}_{\mathrm{eqs}}$" "\n"
    r"$+\lambda_1\mathcal{L}_{\mathrm{constraint}}$" "\n"
    r"$+\lambda_2\mathcal{L}_{\mathrm{bc}}$" "\n\n"
    r"$\mathcal{H}_\theta=mc^2\gamma$" "\n(mass shell)", "#FBF0E6")
arrow(1.45, 3.6, 1.45, 2.85)
arrow(2.6, 2.25, 3.1, 2.6)
arrow(2.6, 4.15, 3.1, 3.6)
arrow(4.4, 1.05, 1.45, 1.05); ax.text(2.9, 0.78, "backprop / Adam+L-BFGS", ha="center", fontsize=8)

# divider
ax.plot([6.2, 6.2], [0.6, 5.3], "k--", lw=1.2)

# Stage 2 (right)
ax.text(9.2, 5.6, r"Stage 2 — symplectic integration (online)",
        ha="center", fontsize=11, weight="bold")
box(6.6, 3.6, 2.3, 1.1, "Frozen\n" r"$\mathcal{H}_\theta$" "  + autodiff\n"
    r"$\partial\mathcal{H}_\theta/\partial\mathbf{r},\ \partial\mathcal{H}_\theta/\partial\mathbf{P}$", "#E9F7F1")
box(6.6, 1.7, 2.3, 1.1, "Tao extended\nphase space\n" r"$(\mathbf{r},\mathbf{P},\mathbf{x},\mathbf{y})$", "#FBF0E6")
box(9.4, 1.4, 2.3, 3.3, "Yoshida 4th-order\nsymplectic map\n" r"$\Phi^{(4)}_{\Delta t}$"
    "\n\nPoincaré–Cartan\ninvariant\npreserved", "#EFEAF6")
arrow(7.75, 3.6, 7.75, 2.85)
arrow(8.9, 2.25, 9.4, 2.6)
arrow(8.9, 4.15, 9.4, 3.6)
arrow(10.55, 1.4, 10.55, 0.95); ax.text(10.55, 0.7, r"trajectory $(\mathbf{r}^n,\mathbf{P}^n)$", ha="center", fontsize=8)

# cross-stage arrow (transfer of the trained Hamiltonian to Stage 2)
arrow(5.75, 4.15, 6.6, 4.15, color=COLORS["SP-PINN"])
ax.text(5.6, 4.75, r"trained $\mathcal{H}_\theta$", ha="center", fontsize=10,
        color=COLORS["SP-PINN"])

fig.savefig(os.path.join(_paths.FIG, "figure1.pdf"), bbox_inches="tight")
fig.savefig(os.path.join(_paths.FIG, "figure1.png"), bbox_inches="tight", dpi=200)
plt.close(fig)
print("Figure 1 schematic written.")
