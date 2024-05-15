# General imports
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple

# Tudat imports
import tudatpy
from tudatpy.trajectory_design import transfer_trajectory
from tudatpy import constants
from tudatpy.numerical_simulation import environment_setup
from tudatpy.util import result2array
from tudatpy.astro.time_conversion import DateTime

# Pygmo imports
import pygmo as pg

## Helpers
"""
First of all, let us define a helper function which is used troughout this example.
"""


# The design variables in the current optimization problem are the departure time and the time of flight between transfer nodes. However, to evaluate an MGA trajectory in Tudat it is necessary to specify a different set of parameters: node times, node free parameters, leg free parameters. This function converts a vector of design variables to the parameters which are used as input to the MGA trajectory object.
#
# The node times are easily computed based on the departure time and the time of flight between nodes. Since an MGA transfer with unpowered legs is used, no node and leg free parameters are required; thus, these are defined as empty lists.

def convert_trajectory_parameters(
        transfer_trajectory_object: tudatpy.kernel.trajectory_design.transfer_trajectory.TransferTrajectory,
        trajectory_parameters: List[float]
        ) -> Tuple[List[float], List[List[float]], List[List[float]]]:
    # Declare lists of transfer parameters
    node_times = list()
    leg_free_parameters = list()
    node_free_parameters = list()

    # Extract from trajectory parameters the lists with each type of parameters
    departure_time = trajectory_parameters[0]
    times_of_flight_per_leg = trajectory_parameters[1:]

    # Get node times
    # Node time for the intial node: departure time
    node_times.append(departure_time)
    # None times for other nodes: node time of the previous node plus time of flight
    accumulated_time = departure_time
    for i in range(0, transfer_trajectory_object.number_of_nodes - 1):
        accumulated_time += times_of_flight_per_leg[i]
        node_times.append(accumulated_time)

    # Get leg free parameters and node free parameters: one empty list per leg
    for i in range(transfer_trajectory_object.number_of_legs):
        leg_free_parameters.append([])
    # One empty array for each node
    for i in range(transfer_trajectory_object.number_of_nodes):
        node_free_parameters.append([])

    return node_times, leg_free_parameters, node_free_parameters


## Optimisation problem
"""
The core of the optimization process is realized by PyGMO, which requires the definition of a problem class.
This definition has to be done in a class that is compatible with what the PyGMO library expects from a User Defined Problem (UDP). See [this page](https://esa.github.io/pygmo2/tutorials/coding_udp_simple.html) from the PyGMO's documentation as a reference. In this example, this class is called `TransferTrajectoryProblem`.

There are four mandatory methods that must be implemented in the class: 
* `__init__()`: This is the constructor for the PyGMO problem class. It is used to save all the variables required to setup the evaluation of the transfer trajectory.
* `get_number_of_parameters(self)`: Returns the number of optimized parameters. In this case, that is the same as the number of flyby bodies (i.e. 6).
* `get_bounds(self)`: Returns the bounds for each optimized parameter. These are provided as an input to `__init__()`. Their values are defined later in this example.
* `fitness(self, x)`: Returns the cost associated with a vector of design parameters. Here, the fitness is the $\Delta V$ required to execute the transfer.
"""


###########################################################################
# CREATE PROBLEM CLASS ####################################################
###########################################################################

class TransferTrajectoryProblem:
    def __init__(self, transfer_trajectory_object: tudatpy.kernel.trajectory_design.transfer_trajectory.TransferTrajectory,
                 departure_date_lb: float, departure_date_up: float,
                 legs_tof_lb: np.ndarray, legs_tof_ub: np.ndarray):
        self.departure_date_lb = departure_date_lb
        self.departure_date_ub = departure_date_ub
        self.legs_tof_lb = legs_tof_lb
        self.legs_tof_ub = legs_tof_ub
        self.transfer_trajectory_function = lambda: transfer_trajectory_object

    def get_bounds(self) -> tuple:
        lower_bound = [self.departure_date_lb] + list(self.legs_tof_lb)
        upper_bound = [self.departure_date_ub] + list(self.legs_tof_ub)
        return (lower_bound, upper_bound)

    def get_number_of_parameters(self):
        # Now reflects only the departure time and one time of flight
        return 2

    def fitness(self, trajectory_parameters: List[float]) -> list:
        transfer_trajectory = self.transfer_trajectory_function()
        node_times, leg_free_parameters, node_free_parameters = convert_trajectory_parameters(
            transfer_trajectory, trajectory_parameters)
        try:
            transfer_trajectory.evaluate(node_times, leg_free_parameters, node_free_parameters)
            delta_v = transfer_trajectory.delta_v
        except:
            delta_v = 1e10  # Large penalty for failed trajectory evaluation
        return [delta_v]



## Simulation Setup
"""
Before running the optimisation, it is first necessary to setup the simulation. In this case, this consists of creating an MGA object. This object is created according to the procedure described in the [MGA trajectory example](https://docs.tudat.space/en/stable/_src_getting_started/_src_examples/notebooks/propagation/mga_dsm_analysis.html). The object is created using the central body, transfer bodies order, departure orbit, and arrival orbit specified in the Cassini 1 problem statement (presented above).
"""

###########################################################################
# Define transfer trajectory properties
###########################################################################

# Define the central body
central_body = "Sun"

# Define order of bodies (nodes) for the revised trajectory
transfer_body_order = ['Earth', 'Jupiter']

# Define departure and arrival orbit parameters (assuming direct departure and arrival)
departure_semi_major_axis = np.inf
departure_eccentricity = 0
arrival_semi_major_axis = np.inf
arrival_eccentricity = 0

# Create a simplified system of bodies
bodies = environment_setup.create_simplified_system_of_bodies()

# Define the trajectory settings for the leg from Earth to Jupiter
transfer_leg_settings, transfer_node_settings = transfer_trajectory.mga_settings_unpowered_unperturbed_legs(
    transfer_body_order,
    departure_orbit=(departure_semi_major_axis, departure_eccentricity),
    arrival_orbit=(arrival_semi_major_axis, arrival_eccentricity))

# Create the transfer calculation object
transfer_trajectory_object = transfer_trajectory.create_transfer_trajectory(
    bodies,
    transfer_leg_settings,
    transfer_node_settings,
    transfer_body_order,
    central_body)

## Optimization
"""
"""

### Optimization Setup
"""
"""

# Before executing the optimization, it is necessary to select the bounds for the optimized parameters (departure date and time of flight per transfer leg). These are selected according to the values in the Cassini 1 problem statement [(Vinkó et al, 2007)](https://www.esa.int/gsp/ACT/doc/MAD/pub/ACT-RPR-MAD-2007-BenchmarkingDifferentGlobalOptimisationTechniques.pdf).

# Lower and upper bound on departure date
departure_date_lb = DateTime(2034, 1, 1).epoch()
departure_date_ub = DateTime(2034, 12, 31).epoch()

# Assuming a range of possible ToF based on typical mission profiles
legs_tof_lb = np.array([600 * constants.JULIAN_DAY])  # Minimum realistic time of flight
legs_tof_ub = np.array([800 * constants.JULIAN_DAY])  # Maximum realistic time of flight



# To setup the optimization, it is first necessary to initialize the optimization problem. This problem, defined through the class `TransferTrajectoryProblem`, is given to PyGMO trough the `pg.problem()` method.
#
# The optimiser is selected to be the Differential Evolution (DE) algorithm (its documentation can be found [here](https://esa.github.io/pygmo2/algorithms.html#pygmo.de)). When selecting the algorithm, here the coefficient F is selected to have the value 0.5, instead of the default 0.8. Additionaly, a fixed seed is selected; since PyGMO uses a random number generator, this ensures that PyGMO's results are reproducible.
#
# Finally, the initial population is created, with a size of 20 individuals.

###########################################################################
# Setup optimization
###########################################################################
# Initialize optimization class
optimizer = TransferTrajectoryProblem(transfer_trajectory_object,
                                      departure_date_lb,
                                      departure_date_ub,
                                      legs_tof_lb,
                                      legs_tof_ub)

# Creation of the pygmo problem object
prob = pg.problem(optimizer)

# To print the problem's information: uncomment the next line
# print(prob)

# Define number of generations per evolution
number_of_generations = 1

# Fix seed
optimization_seed = 4444

# Create pygmo algorithm object
algo = pg.algorithm(pg.de(gen=number_of_generations, seed=optimization_seed, F=0.5))

# To print the algorithm's information: uncomment the next line
# print(algo)

# Set population size
population_size = 20

# Create population
pop = pg.population(prob, size=population_size, seed=optimization_seed)

### Run Optimization
"""
Finally, the optimization can be executed by successively evolving the defined population.

A total number of evolutions of 800 is selected. Thus, the method `algo.evolve()` is called 800 times inside a loop. After each evolution, the best fitness and the list with the best design variables are saved.
"""

###########################################################################
# Run optimization
###########################################################################

# Set number of evolutions
number_of_evolutions = 100  # Adjusted for quicker convergence based on problem simplicity

# Initialize empty containers
individuals_list = []
fitness_list = []

for i in range(number_of_evolutions):
    pop = algo.evolve(pop)

    # individuals save
    individuals_list.append(pop.champion_x)
    fitness_list.append(pop.champion_f)

print('The optimization has finished')

## Results Analysis
"""
Having finished the optimisation, it is now possible to analyse the results.

According to [Vinkó et al (2007)](https://www.esa.int/gsp/ACT/doc/MAD/pub/ACT-RPR-MAD-2007-BenchmarkingDifferentGlobalOptimisationTechniques.pdf), the best known solution for the Cassini 1 problem has a final objective function value of 4.93 km/s.

The executed optimization process results in a final objective function value of 4933.17 m/s, with a slightly different decision vector from the one presented by Vinkó et al. (2017). This marginal difference can be explained by an inperfect convergence of the used optimizer, which is expected, considering that DE is a global optimizer. 

The evolution of the minimum $\Delta V$ throughout the optimization process can be plotted.
"""

###########################################################################
# Results post-processing
###########################################################################

# Extract the best individual
print('\n########### CHAMPION INDIVIDUAL ###########\n')
print('Total Delta V [m/s]: ', pop.champion_f[0])
best_decision_variables = pop.champion_x / constants.JULIAN_DAY
print('Departure time w.r.t J2000 [days]: ', best_decision_variables[0])
print('Earth-Jupiter time of flight [days]: ', best_decision_variables[1])



# Plot fitness over generations
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(np.arange(0, number_of_evolutions), np.float_(fitness_list) / 1000, label='Function value: Feval')
# Plot champion
champion_n = np.argmin(np.array(fitness_list))
ax.scatter(champion_n, np.min(fitness_list) / 1000, marker='x', color='r', label='All-time champion', zorder=10)

# Prettify
ax.set_xlim((0, number_of_evolutions))
ax.set_ylim([5, 12])  # Wider range to capture unexpected optimization results
ax.grid('major')
ax.set_title('Best individual over generations', fontweight='bold')
ax.set_xlabel('Number of generation')
ax.set_ylabel(r'$\Delta V [km/s]$')
ax.legend(loc='upper right')
plt.tight_layout()
plt.legend()

### Plot the transfer
"""
Finally, the position history throughout the transfer can be retrieved from the transfer trajectory object and plotted.
"""

# Reevaluate the transfer trajectory using the champion design variables
node_times, leg_free_parameters, node_free_parameters = convert_trajectory_parameters(transfer_trajectory_object,
                                                                                      pop.champion_x)
transfer_trajectory_object.evaluate(node_times, leg_free_parameters, node_free_parameters)

# Extract the state history
state_history = transfer_trajectory_object.states_along_trajectory(500)
fly_by_states = np.array([state_history[node_times[i]] for i in range(len(node_times))])
state_history = result2array(state_history)
au = 1.5e11

# Plot the transfer trajectory
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111)

# Assuming state_history[:, 1] and state_history[:, 2] are the x and y coordinates in AU
ax.plot(state_history[:, 1] / au, state_history[:, 2] / au, label='Trajectory')

# Only plot relevant flybys (Earth departure and Jupiter arrival)
ax.scatter(fly_by_states[0, 0] / au, fly_by_states[0, 1] / au, color='blue', label='Earth departure')
ax.scatter(fly_by_states[1, 0] / au, fly_by_states[1, 1] / au, color='red', label='Jupiter fly-by')

# Sun as the central point
ax.scatter([0], [0], color='orange', label='Sun')

# Set labels and legend
ax.set_xlabel('x [AU]')
ax.set_ylabel('y [AU]')
ax.set_title('Hohmann Transfer from Earth to Jupiter')
ax.legend(loc='upper right')

# Ensure equal aspect ratio to correctly represent distances
ax.set_aspect('equal')
plt.grid(True)
plt.tight_layout()
plt.show()

