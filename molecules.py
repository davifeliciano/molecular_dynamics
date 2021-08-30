import numpy as np
from numpy.random import default_rng

rng = default_rng()
float_formatter = "{:.3e}".format
np.set_printoptions(formatter={"float_kind": float_formatter})

# BOLTZMANN = 1.38064852e-23
BOLTZMANN = 1.0


def lenard_jones_potential(rel_position: np.ndarray) -> float:
    """Given a vector, compute Lenard Jones potential value"""
    norm = np.linalg.norm(rel_position)
    return 1 / norm ** 12 - 2 / norm ** 6


def lenard_jones_force(rel_position: np.ndarray) -> np.ndarray:
    """Given a vector, compute Lenard Jones force vector"""
    norm = np.linalg.norm(rel_position)
    return 12 * (1 / norm ** 14 - 1 / norm ** 8) * rel_position


class Cell:
    """
    Cell of a Grid

    Atribbutes:

    row, col : ints
        define the position of the cell on the grid

    neighbors : list of Cells
        a list with all the neighbors of the cell

    particles : list of Particles
        a list with all the particles contained by the cell
    """

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

    def is_physical_neighbor(self, cell):
        """
        Returns True if the given cell is a physical neighbor of the cell instance
        Returns False if the given cell is not a neighbor, or is periodic image neighbor
        """
        if cell not in self.neighbors:
            return False
        for index_self, index in zip(self.indexes, cell.indexes):
            if abs(index_self - index) > 1:
                return False
        return True


class Grid:
    """
    Grid composing a box of particles

    Attributes:

    rows, cols : ints
        number of rows and columns of cells composing the grid

    width, height : float
        width and height of the box

    init_temp : float
        the initial temperature of the system

    time : float
        the current time elapsed since the system was created.
        This value must only be changed by calling the update method

    cells : np.ndarray
        numpy array of cells that compose the grid

    particles : list
        list of particles in the system

    time : float
        time elapsed relative to the initial state of the system

    iteration : int
        number of iterations done with update method since the initial
        state of the system
    """

    def __init__(self, size=(36, 36), cell_size=3, particles=1000, init_temp=10.0):

        self.cell_size = cell_size

        width, height = size

        self.cols, self.rows = self.shape = (
            int(width / cell_size),
            int(height / cell_size),
        )

        self.width, self.height = self.size = (
            self.cols * cell_size,
            self.rows * cell_size,
        )

        self.init_temp = init_temp / BOLTZMANN
        self.time = 0
        self.dt = 0.001 / np.sqrt(self.init_temp)
        self.iteration = 0

        self.cells = np.array(
            [[Cell(i, j) for j in range(self.cols)] for i in range(self.rows)],
            dtype=Cell,
        )

        # Appending neighbors to cell.neighbors for all cells in the grid
        rows = self.rows
        cols = self.cols
        for i in range(rows):

            neighbors = self.cells[i][0].neighbors
            neighbors.append(self.cells[i][1])
            neighbors.append(self.cells[(i + 1) % rows][0])
            neighbors.append(self.cells[(i + 1) % rows][1])
            neighbors.append(self.cells[(i + 1) % rows][cols - 1])

            for j in range(1, self.cols):
                neighbors = self.cells[i][j].neighbors
                neighbors.append(self.cells[i][(j + 1) % cols])
                neighbors.append(self.cells[(i + 1) % rows][j - 1])
                neighbors.append(self.cells[(i + 1) % rows][j])
                neighbors.append(self.cells[(i + 1) % rows][(j + 1) % cols])

        # Adding particles uniformly to the grid
        self.particle_count = particles
        self.particles = []

        area = self.width * self.height
        site_size = np.sqrt(area / self.particle_count)
        x_sites = int(self.width / site_size)
        y_sites = int(self.height / site_size)
        site_size = np.floor(min((self.width / x_sites, self.height / y_sites)))

        for i in range(self.particle_count):
            z = i / x_sites
            x = (x_sites * (z - np.floor(z)) + 0.5) * site_size
            y = (np.ceil((i + 1) / x_sites) - 0.5) * site_size
            pos = np.array([x, y])
            vel = rng.normal(loc=0.0, scale=np.sqrt(self.init_temp), size=2)
            self.particles.append(Particle(pos, vel, self))

        # Updating particle forces and energies
        self.update_forces()
        for particle in self.particles:
            particle.old_force = particle.force

    def __repr__(self):
        return (
            f"Grid(size={self.size}, cell_size={self.cell_size}, "
            f"particles={self.particles}, init_temp={self.init_temp:.3e})"
        )

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
            sum += np.linalg.norm(particle.velocity) ** 2
        return sum / 2

    def energy(self):
        return self.potential_energy() + self.kinetic_energy()

    def temperature(self):
        return self.kinetic_energy() / self.particle_count

    def update_forces(self):
        """Update the forces on all particles"""
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
                            rel_position = (
                                second_particle.position - first_particle.position
                            )

                            force = lenard_jones_force(rel_position)
                            energy = lenard_jones_potential(rel_position)
                            second_particle.force += force
                            first_particle.force += -force
                            second_particle.energy += energy
                            first_particle.energy += energy

                    # Compute forces between particles in different cells
                    for cell in self.cells[i][j].neighbors:
                        for second_particle in cell.particles:

                            # Check whether the second cell is a physical neighbor of the first
                            if self.cells[i][j].is_physical_neighbor(cell):
                                rel_position = (
                                    second_particle.position - first_particle.position
                                )

                            # If it is a periodic image, apply the corrections to the second particle position
                            else:
                                second_particle_x = second_particle.position[0]
                                second_particle_y = second_particle.position[1]
                                if cell.row == 0:
                                    second_particle_y += self.height
                                if cell.col == 0:
                                    second_particle_x += self.width
                                rel_position = (
                                    np.array([second_particle_x, second_particle_y])
                                    - first_particle.position
                                )

                            force = lenard_jones_force(rel_position)
                            energy = lenard_jones_potential(rel_position)
                            second_particle.force += force
                            first_particle.force += -force
                            second_particle.energy += energy
                            first_particle.energy += energy

    def update(self):
        """Updates the system"""
        self.update_forces()

        for particle in self.particles:
            particle.update()

        self.time += self.dt
        self.iteration += 1


class Particle:
    """
    Point Particle

    Atributtes:

    position : np.ndarray
        position of the particle on the grid relative to its
        top left corner

    velocity : np.ndarray
        velocity of the particle

    force : np.ndarray
        net force of the system in the particle

    energy : float
        potential energy of the particle

    grid : Grid
        grid in which the particle is inside

    cell : Cell
        cell in which the particle is inside
    """

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
        return (
            f"Particle(position={self.position}, velocity={self.velocity}, "
            f"energy={self.energy:.3e}, force={self.force})"
        )

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

    grid = Grid(size=(43, 71), particles=1000, init_temp=100.0)

    """ print("Particles created:")
    for particle in grid.particles:
        print(particle)

    print("\nAmount of particles in each Cell:")
    for i in range(grid.rows):
        for j in range(grid.cols):
            print(f"Cell {i}, {j}: {len(grid.cells[i][j].particles)} particles")

    print("\nNeighbors of each cell:")
    for i in range(grid.rows):
        for j in range(grid.cols):
            print(f"Cell {i}, {j}: {grid.cells[i][j].neighbors}") """

    print(
        f"\nEvolving {grid.width} by {grid.height} box with {grid.particle_count} particles,",
        f"dt = {grid.dt:.3e}, Initial temperature = {grid.init_temp:.3e}",
    )

    kinetic = grid.kinetic_energy()
    potential = grid.potential_energy()
    init_energy = kinetic + potential
    energy = 0

    while True:
        kinetic = grid.kinetic_energy()
        potential = grid.potential_energy()
        energy = kinetic + potential

        print(
            f"\nIteration = {grid.iteration}, Elapsed Time = {grid.time:.3e}"
            f"\nPotential = {potential:.3e}, Kinetic = {kinetic:.3e},",
            f"Energy = {energy:.3e}, Temperature = {grid.temperature():.3e}",
        )

        if (difference := (energy - init_energy) / init_energy) > 0.05:
            print(f"\nEnergy not conserved! Difference of {difference}\%")
            break

        grid.update()
        init_energy = energy

    print(
    f"\n{grid.width} by {grid.height} box with {grid.particle_count} particles,",
    f"dt = {grid.dt:.3e}, Initial temperature = {grid.init_temp:.3e}",
    )
