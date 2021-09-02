import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import molecules as mol

parser = argparse.ArgumentParser(
    description=(
        "plot the evolution of a 2-dimensional system of"
        "particles using cell linked-lists method"
    ),
    epilog=(
        "the letters in the plot legend stand for: "
        "U, the potential energy of the system; "
        "K, the kinetic energy of the system; "
        "U + K, the total energy of the system; "
        "T, the temperature, measure as mean value of K"
    ),
)

parser.add_argument(
    "particles",
    type=int,
    nargs="?",
    help=(
        "the number of particles in the system. The larger this number, "
        "the longer will be the computation time of each step."
    ),
)

parser.add_argument(
    "-s",
    "--size",
    type=float,
    nargs=2,
    default=[100, 100],
    metavar=("WIDTH", "HEIGHT"),
    help="desired width and height of the box. Default is 100 and 100.",
)

parser.add_argument(
    "-c",
    "--cell_size",
    type=float,
    nargs="?",
    default=3.0,
    const=3.0,
    help="size of the cells in which the bow will be divided. Default is 3.0.",
)

parser.add_argument(
    "-t",
    "--temperature",
    type=float,
    nargs="?",
    default=100.0,
    const=100.0,
    help="initial temperature of the system in energy units. Default is 100.",
)

args = parser.parse_args()

size = args.size
cell_size = args.cell_size
particles = args.particles
init_temp = args.temperature

grid = mol.Grid(
    size=size, cell_size=cell_size, particles=particles, init_temp=init_temp
)

time = []
potential = []
kinetic = []
energy = []
temperature = []

fig, ax = plt.subplots()


def move_cursor(y, x):
    print("\033[%d;%dH" % (y, x))


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

    # move_cursor(0, 0)

    print(
        f"Time = {grid.time:.3e}",
        f"U = {potential_now:.3e}",
        f"K = {kinetic_now:.3e}",
        f"U + K = {energy_now:.3e}",
        f"T = {temperature_now:.3e}",
        sep="\n",
        end="",
    )

    ax.clear()
    ax.set(xlabel="Time, in seconds")

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

    ax.grid(axis="y")

    grid.update()


ani = FuncAnimation(fig, animate)

plt.show()
