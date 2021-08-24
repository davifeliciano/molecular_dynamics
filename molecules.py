import numpy as np
from numpy.random import default_rng

rng = default_rng()


def lenard_jones_potential(rel_position):
    norm = np.linalg.norm(rel_position)
    return 1 / norm ** 12 - 2 / norm ** 6


def lenard_jones_force(rel_position):
    norm = np.linalg.norm(rel_position)
    return 12 * (1 / norm ** 14 - 1 / norm ** 8) * rel_position


class Cell:

    def __init__(self, row, col):

        self.indexes = self.row, self.col = row, col
        self.neighbors = []
        self.particles = []

    def __repr__(self):
        return f'Cell(row={self.row}, col={self.col})'

    def append_particle(self, particle):
        self.particles.append(particle)

    def remove_particle(self, particle):
        self.particles.remove(particle)

class Grid:

    def __init__(self, size=(18, 18), cell_size=3, particles=400, init_temp=273.15):
        
        self.cell_size = cell_size

        width, height = size
        self.cols, self.rows = self.shape = (int(np.ceil(width / cell_size)), int(np.ceil(height / cell_size)))
        self.width, self.height = self.size = (self.cols * cell_size, self.rows * cell_size)
        self.init_temp = init_temp

        self.cells = np.array([[Cell(i, j) for i in range(self.rows)] for j in range(self.cols)], dtype=Cell)

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

        # Adding particles to the grid
        self.particles = []
        for _ in range(particles):
            pos = np.array([rng.random() * x for x in self.size])
            vel = rng.normal(loc=0.0, scale=np.sqrt(init_temp), size=2)
            self.particles.append(Particle(pos, vel, self))

    def __repr__(self):
        return f'Grid(size={self.size}, cell_size={self.cell_size}, particles={self.particles}, init_temp={self.init_temp})'

    def cell(self, x, y):
        ''' Returns a cell based on position on the plane '''
        row = int(y / self.cell_size)
        col = int(x / self.cell_size)
        return self.cells[row][col]

    def potential_energy(self):
        pass

    def kinetic_energy(self):
        sum = 0
        for particle in self.particles:
            sum += particle.velocity @ particle.velocity
        return sum / 2

    def energy(self):
        pass

    def temperature(self):
        return self.kinetic_energy() / self.particles

    def set_temperature(self, temp):
        ''' Apply the renormalization factor sqrt(temp / self.temperature())
            to the velocities of the particles '''
        pass

    def update(self):
        ''' Update the system '''
        pass


class Particle:

    def __init__(self, pos, vel, grid):

        self.position = pos
        self.velocity = vel
        self.grid = grid

        self.cell = self.grid.cell(*pos)
        self.cell.append_particle(self)

    def __repr__(self):
        return f'Particle(position={self.position}, velocity={self.velocity})'

    def update(self):
        ''' Update velocity and position of the particle '''
        pass


if __name__ == '__main__':

    grid = Grid()

    for particle in grid.particles:
        print(particle)

    for i in range(grid.rows):
        for j in range(grid.cols):
            print(f'Cell {i}, {j}: {len(grid.cells[i][j].particles)} particles')

    for i in range(grid.rows):
        for j in range(grid.cols):
            print(f'Cell {i}, {j} neighbors -> {grid.cells[i][j].neighbors}')