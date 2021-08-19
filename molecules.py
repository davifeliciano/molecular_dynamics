import numpy as np
from numpy.random import default_rng

rng = default_rng()

class Cell:

    def __init__(self, row, col):

        self.indexes = row, col
        self.particles = []

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

        self.particles = []
        for _ in range(particles):
            pos = np.array([rng.random() * x for x in self.size])
            vel = rng.normal(loc=0.0, scale=np.sqrt(init_temp), size=2)
            self.particles.append(Particle(pos, vel, self))

    def cell(self, x, y):
        ''' Returns a cell based on position on the plane '''
        row = int(y / self.cell_size)
        col = int(x / self.cell_size)
        return self.cells[row][col]

    def potential_energy(self):
        pass

    def kinetic_energy(self):
        pass

    def energy(self):
        pass

    def temperature(self):
        pass

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

    def update(self):
        pass


if __name__ == '__main__':

    grid = Grid()
    for particle in grid.particles:
        print(particle, '->', particle.velocity)

    for i in range(grid.rows):
        for j in range(grid.cols):
            print(f'Cell {i}, {j}: {len(grid.cells[i][j].particles)} particles')