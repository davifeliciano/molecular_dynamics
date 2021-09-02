import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import molecules as mol

grid = mol.Grid(particles=1000)

time = []
potential = []
kinetic = []
energy = []
temperature = []

fig, ax = plt.subplots()


def animate(i):
    potential_now = grid.potential_energy()
    kinetic_now = grid.kinetic_energy()
    energy_now = grid.total_energy()
    temperature_now = grid.temperature()

    time.append(grid.time)
    potential.append(potential_now)
    kinetic.append(kinetic_now)
    energy.append(energy_now)
    temperature.append(temperature_now)

    print(
        f"Time = {grid.time:.3e}",
        f"U = {potential_now:.3e}",
        f"K = {kinetic_now:.3e}",
        f"U + K = {energy_now:.3e}",
        f"T = {temperature_now:.3e}",
        sep="\n",
        end="\n\n",
    )

    ax.clear()

    # ax.set(yscale="log")

    ax.plot(time, potential, label="U")
    ax.plot(time, kinetic, label="K")
    ax.plot(time, energy, label="U + K")
    ax.plot(time, temperature, label="T")

    ax.legend(
        loc="upper center",
        ncol=4,
        bbox_to_anchor=(0.5, 1.10),
        fancybox=True,
        shadow=True,
    )

    grid.update()


ani = FuncAnimation(fig, animate)

plt.show()
