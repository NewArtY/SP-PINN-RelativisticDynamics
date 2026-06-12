"""Regenerate Figure 2 from data/sim2_magnetic.csv adding distinct line styles
(in addition to colour) for colour-blind safety (addresses review point 8)."""
import _paths  # noqa: F401
import os
import numpy as np
import matplotlib.pyplot as plt
from relsim.plotstyle import apply_style, COLORS

LS = {"RK4": "-", "Boris": (0, (5, 2)), "SP-PINN": (0, (1, 1))}

data = np.genfromtxt(os.path.join(_paths.DATA, "sim2_magnetic.csv"),
                     delimiter=",", names=True)
per = data["period"]
drL = {"Boris": data["drL_Boris"], "RK4": data["drL_RK4"], "SP-PINN": data["drL_SPPINN"]}
dgam = {"Boris": data["dgamma_Boris"], "RK4": data["dgamma_RK4"], "SP-PINN": data["dgamma_SPPINN"]}

apply_style()
fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
for s in ["RK4", "Boris", "SP-PINN"]:
    axes[0].loglog(per, np.maximum(drL[s], 1e-16), color=COLORS[s], ls=LS[s], label=s)
    axes[1].loglog(per, np.maximum(dgam[s], 1e-16), color=COLORS[s], ls=LS[s], label=s)
axes[0].set_xlabel("Cyclotron periods $n$")
axes[0].set_ylabel(r"Relative Larmor-radius error $\Delta r_L/r_L^{(0)}$")
axes[0].set_title("(a)")
axes[1].set_xlabel("Cyclotron periods $n$")
axes[1].set_ylabel(r"Relative Lorentz-factor error $\Delta\gamma/\gamma_0$")
axes[1].set_title("(b)")
for ax in axes:
    ax.legend(loc="upper left"); ax.set_ylim(1e-16, 1e-1)
fig.savefig(os.path.join(_paths.FIG, "figure2.pdf"))
fig.savefig(os.path.join(_paths.FIG, "figure2.png"))
plt.close(fig)
print("Figure 2 regenerated with distinct line styles.")
