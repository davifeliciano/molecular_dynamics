# molecular_dynamics
A basic simulation of particles under Lenard-Jones potential using the cell linked-lists method.

![100x100p2500t100](https://user-images.githubusercontent.com/26972046/131927852-19208fa1-7fac-4785-a865-bed4fc053aca.png)

The simulation consists of a 2-dimensional box with a number of particles with the potential of interation
<br><img src="https://latex.codecogs.com/gif.latex?V%28r%29%3D%5Cfrac%7B1%7D%7Br%5E%7B12%7D%7D-%5Cfrac%7B2%7D%7Br%5E6%7D"/><br>
and initial average velocity folowing the [Maxwell Distribution](https://en.wikipedia.org/wiki/Maxwell%E2%80%93Boltzmann_distribution). Integration of the equations of motion
is done using [Verlet Integration](https://en.wikipedia.org/wiki/Verlet_integration) method. The mass of the particles, 
as well as the Boltzmann Constant are set to 1, for simplicity. Running ```main.py``` will plot,
at the end of each step of integration, the relevant physical quantities of the system. 
Run ```main.py --help``` for usage info.

Make sure to intall Numpy and Matplotlib before running ```main.py```. 
