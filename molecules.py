import numpy as np
from numpy.random import default_rng
from time import sleep

rng = default_rng()

BOLTZMANN = 1.38064852e-23


def lenard_jones_potential(rel_position: np.ndarray) -> float:
    """Given a vector, compute Lenard Jones potential value"""
    norm = np.linalg.norm(rel_position)
    return 1 / norm ** 12 - 2 / norm ** 6


def lenard_jones_force(rel_position: np.ndarray) -> np.ndarray:
    """Given a vector, compute Lenard Jones force vector"""
    norm = np.linalg.norm(rel_position)
    return 12 * (1 / norm ** 14 - 1 / norm ** 8) * rel_position


class Cell:
    def __init__(self, row, col):

        self.indexes = self.row, self.col = row, col
        self.neighbors = []
        self.particles = []

    def __repr__(self):
        return f"Cell(row={self.row}, col={self.col}, particles={len(self.particles)})"

    def append_particle(self, particle):
        self.particles.append(particle)

    def remove_particle(self, particle):
        self.particles.remove(particle)


class Grid:
    def __init__(self, size=(18, 18), cell_size=3, particles=400, init_temp=173.15):

        self.cell_size = cell_size

        width, height = size

        self.cols, self.rows = self.shape = (
            int(np.ceil(width / cell_size)),
            int(np.ceil(height / cell_size)),
        )

        self.width, self.height = self.size = (
            self.cols * cell_size,
            self.rows * cell_size,
        )

        self.init_temp = init_temp / BOLTZMANN
        self.time = 0
        self.dt = 0.01 / np.sqrt(self.init_temp)

        self.cells = np.array(
            [[Cell(i, j) for i in range(self.rows)] for j in range(self.cols)],
            dtype=Cell,
        )

        # Appending neighbors to cell.neighbors for all cells in the grid
        for i in range(self.rows - 1):
            for j in range(1, self.cols - 1):
                neighbors = self.cells[i][j].neighbors
                neighbors.append(self.cells[i][j + 1])
                neighbors.append(self.cells[i + 1][j - 1])
                neighbors.append(self.cells[i + 1][j])
                neighbors.append(self.cells[i + 1][j + 1])

        # Appending neighbors of the first and last columns of self.cells
        for i in range(self.rows - 1):
            neighbors_left = self.cells[i][0].neighbors
            neighbors_left.append(self.cells[i][j + 1])
            neighbors_left.append(self.cells[i + 1][j])
            neighbors_left.append(self.cells[i + 1][j + 1])

            neighbors_right = self.cells[i][self.cols - 1].neighbors
            neighbors_right.append(self.cells[i + 1][j - 1])
            neighbors_right.append(self.cells[i + 1][j])

        # Appending neighbors of the last row of self.cells
        for j in range(self.cols - 1):
            i = self.rows - 1
            neighbors = self.cells[i][j].neighbors
            neighbors.append(self.cells[i][j + 1])

        # Adding particles to the grid
        self.particles = []
        for _ in range(particles):
            pos = np.array([rng.random() * x for x in self.size])
            vel = rng.normal(loc=0.0, scale=np.sqrt(init_temp), size=2)
            self.particles.append(Particle(pos, vel, self))

        # Updating particle forces and energies
        self.update()

    def __repr__(self):
        return f"Grid(size={self.size}, cell_size={self.cell_size}, " \
                "particles={self.particles}, init_temp={self.init_temp})"

    def cell(self, x, y):
        """Returns a cell based on position on the plane"""
        row = int(y / self.cell_size)
        col = int(x / self.cell_size)
        return self.cells[row][col]

    def potential_energy(self):
        sum = 0
        for particle in self.particles:
            sum += particle.energy
        return sum

    def kinetic_energy(self):
        sum = 0
        for particle in self.particles:
            sum += particle.velocity @ particle.velocity
        return sum / 2

    def energy(self):
        return self.potential_energy() + self.kinetic_energy()

    def temperature(self):
        return self.kinetic_energy() / len(self.particles)

    def update(self):
        """Update the system"""
        for particle in self.particles:
            particle.old_force = particle.force
            particle.force = np.zeros(2)
            particle.energy = 0

        for i in range(self.rows):
            for j in range(self.cols):
                # Compute forces between particles in same cell
                for first_particle in self.cells[i][j].particles:
                    for second_particle in self.cells[i][j].particles:
                        if first_particle is not second_particle:
                            rel_position = second_particle.position - first_particle.position
                            force = lenard_jones_force(rel_position)
                            energy = lenard_jones_potential(rel_position)
                            second_particle.force += force
                            first_particle.force += -force
                            second_particle.energy += energy
                            first_particle.energy += energy

                    # Compute forces between particles in different cells
                    for cell in self.cells[i][j].neighbors:
                        for second_particle in cell.particles:
                            rel_position = second_particle.position - first_particle.position
                            force = lenard_jones_force(rel_position)
                            energy = lenard_jones_potential(rel_position)
                            second_particle.force += force
                            first_particle.force += -force
                            second_particle.energy += energy
                            first_particle.energy += energy

        # Update position, velocity and cells of all particles
        for particle in self.particles:
            particle.old_force = particle.force
            particle.update()

        # Update time
        self.time += self.dt


class Particle:
    def __init__(self, pos, vel, grid):

        self.position = pos
        self.velocity = vel
        self.grid = grid

        self.cell = self.grid.cell(*pos)
        self.cell.append_particle(self)

        self.old_force = np.zeros(2)
        self.force = np.zeros(2)
        self.energy = 0

    def __repr__(self):
        return f"Particle(position={self.position}, velocity={self.velocity}, " \
                "energy={self.energy}, force={self.force})"

    def current_cell(self):
        """Return the cell in which the particle currently is"""
        return self.grid.cell(self.position[0], self.position[1])

    def update(self):
        """Update position, velocity and cell of the particle"""
        dt = self.grid.dt
        self.position += self.velocity * dt + self.force * 0.5 * dt * dt
        self.velocity += (self.old_force + self.force) * 0.5 * dt

        # Adding periodic boundary conditions
        self.position %= np.array(self.grid.size)

        # Reallocating particle to new cell, depending on the new position
        new_cell = self.current_cell()
        if self.cell is not new_cell:
            self.cell.remove_particle(self)
            new_cell.append_particle(self)
            self.cell = new_cell


if __name__ == "__main__":

    grid = Grid()

    print("Particles created:")
    for particle in grid.particles:
        print(particle)

    print("\nAmount of particles in each Cell:")
    for i in range(grid.rows):
        for j in range(grid.cols):
            print(
                f"Cell {i}, {j}: {len(grid.cells[i][j].particles)} particles")

    print("\nNeighbors of each cell:")
    for i in range(grid.rows):
        for j in range(grid.cols):
            print(f"Cell {i}, {j} neighbors -> {grid.cells[i][j].neighbors}")

    print(f"\nEvolving the system with dt={grid.dt}")
    t = 0
    while True:
        print(
            f"\nTime = {grid.time}, Energy = {grid.energy()}, Temperature = {grid.temperature()}"
        )
        print(grid.particles[0])
        grid.update()
        sleep(0.50)
