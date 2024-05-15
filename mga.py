# Multiple Gravity Assist trajectories
"""

Copyright (c) 2010-2022, Delft University of Technology. All rights reserved. This file is part of the Tudat. Redistribution and use in source and
binary forms, with or without modification, are permitted exclusively
under the terms of the Modified BSD license. You should have received
a copy of the license with this file. If not, please or visit:
http://tudat.tudelft.nl/LICENSE.

"""

## Context
"""

This example demonstrates how Multiple Gravity Assist (MGA) transfer trajectories can be simulated. Three types of transfers are analyzed:
* High-thrust transfer with unpowered legs
* High-thrust transfer with deep space maneuvers (DSMs) and manually-created legs and nodes
* Low-thrust transfer with hodographic shaping


In addition, this example show how the results, such as partial $\Delta$V's, total $\Delta$V and time of flight
values can be retrieved from the transfer object.

A complete guide on transfer trajectory design is given on [this page](https://tudat-space.readthedocs.io/en/latest/_src_user_guide/astrodynamics/trajectory_design.html) of tudat user documentation.
"""

## MGA Transfer With Unpowered Legs
"""
"""

## Import statements
"""

The required import statements are made here, at the very beginning.

Some standard modules are first loaded: numpy and matplotlib.pyplot.

Then, the different modules of tudatpy that will be used are imported.
"""

# Load standard modules
import numpy as np
import matplotlib.pyplot as plt

# Load tudatpy modules
from tudatpy.trajectory_design import transfer_trajectory, shape_based_thrust
from tudatpy.numerical_simulation import environment_setup
from tudatpy.util import result2array
from tudatpy import constants

# First, let's explore an MGA transfer trajectory with no thrust applied during the transfer legs. In this case, the impulsive $\Delta$Vs are only applied during the gravity assists.

### Setup and inputs
"""

A simplified system of bodies suffices for this application, with the Sun as central body. The planets that are visited for a gravity assist are defined in the list `transfer_body_order`. The first body in the list is the departure body and the last one is the arrival body.

The departure and arrival orbit can be specified, but they are not mandatory. If not specified, the departure and arrival planets are selected to be swing-by nodes.
Departures and arrivals at the edge of the Sphere Of Influence (SOI) of a node can be done by specifying eccentricity $e=0$ and semi-major axis $a=\infty$.

In this example, the spacecraft departs from the edge of Earth’s SOI and is inserted into a highly elliptical orbit around Saturn.
"""

# Create a system of simplified bodies (create all main solar system bodies with their simplest models)
bodies = environment_setup.create_simplified_system_of_bodies()
central_body = 'Sun'

# Define the order of bodies (nodes) for gravity assists
transfer_body_order = ['Earth', 'Venus', 'Venus', 'Earth', 'Jupiter', 'Saturn']

# Define the departure and insertion orbits
departure_semi_major_axis = np.inf
departure_eccentricity = 0.

arrival_semi_major_axis = 1.0895e8 / 0.02
arrival_eccentricity = 0.98

### Create transfer settings and transfer object
"""
The specified inputs can not be used directly, but they have to be translated to distinct settings, relating to either the nodes (departure, gravity assist, and arrival planets) or legs (trajectories in between planets). The fact that unpowered legs are used is indicated by the creation of unpowered and unperturbed leg settings.
These settings are, in turn, used to create the transfer trajectory object.
"""

# Define the trajectory settings for both the legs and at the nodes
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

### Define transfer parameters
"""
Next, it is necessary to specify the parameters which define the transfer. The advantage of having a transfer trajectory object is that it allows analyzing many different sets of transfer parameters using the same transfer trajectory object.
The definition of the parameters that need to be specified for this transfer can be printed using the `transfer_trajectory.print_parameter_definitions()` function.
"""

# Print transfer parameter definitions
print("Transfer parameter definitions:")
transfer_trajectory.print_parameter_definitions(transfer_leg_settings, transfer_node_settings)

# For this transfer with unpowered legs, the transfer parameters only constitute the times at which the powered gravity assists are executed, i.e. at the nodes.
# This type of legs does not require any node free parameters or leg free parameters to be specified. Thus, they are defined as lists containing empty arrays.

# Define times at each node
julian_day = constants.JULIAN_DAY
node_times = list()
node_times.append((-789.8117 - 0.5) * julian_day)
node_times.append(node_times[0] + 158.302027105278 * julian_day)
node_times.append(node_times[1] + 449.385873819743 * julian_day)
node_times.append(node_times[2] + 54.7489684339665 * julian_day)
node_times.append(node_times[3] + 1024.36205846918 * julian_day)
node_times.append(node_times[4] + 4552.30796805542 * julian_day)

# Define free parameters per leg (for now: none)
leg_free_parameters = list()
for i in transfer_leg_settings:
    leg_free_parameters.append(np.zeros(0))

# Define free parameters per node (for now: none)
node_free_parameters = list()
for i in transfer_node_settings:
    node_free_parameters.append(np.zeros(0))

### Evaluate transfer
"""
The transfer parameters are now used to evaluate the transfer trajectory, which means that the semi-analytical methods used to determine the $\Delta$V of each leg are now applied.
"""

# Evaluate the transfer with given parameters
transfer_trajectory_object.evaluate(node_times, leg_free_parameters, node_free_parameters)

### Extract results and plot trajectory
"""
Last but not least, with the transfer trajectory computed, we can now analyse it.
"""

#### Print results
"""
Having evaluated the transfer trajectory, it is possible to extract various transfer characteristics, such as the $\Delta$V and time of flight.
"""

# Print the total DeltaV and time of Flight required for the MGA
print('Total Delta V of %.3f m/s and total Time of flight of %.3f days\n' % \
      (transfer_trajectory_object.delta_v, transfer_trajectory_object.time_of_flight / julian_day))

# Print the DeltaV required during each leg
print('Delta V per leg: ')
for i in range(len(transfer_body_order) - 1):
    print(" - between %s and %s: %.3f m/s" % \
          (transfer_body_order[i], transfer_body_order[i + 1], transfer_trajectory_object.delta_v_per_leg[i]))
print()

# Print the DeltaV required at each node
print('Delta V per node : ')
for i in range(len(transfer_body_order)):
    print(" - at %s: %.3f m/s" % \
          (transfer_body_order[i], transfer_trajectory_object.delta_v_per_node[i]))

#### Plot the transfer
"""
The state throughout the transfer can be retrieved from the transfer trajectory object, here at 500 instances per leg, to visualize the transfer.
"""

# Extract the state history
state_history = transfer_trajectory_object.states_along_trajectory(500)
fly_by_states = np.array([state_history[node_times[i]] for i in range(len(node_times))])
state_history = result2array(state_history)
au = 1.5e11

# Plot the transfer
fig = plt.figure(figsize=(8, 5))
ax = fig.add_subplot(111, projection='3d')
# Plot the trajectory from the state history
ax.plot(state_history[:, 1] / au, state_history[:, 2] / au, state_history[:, 3] / au)
# Plot the position of the nodes
ax.scatter(fly_by_states[0, 0] / au, fly_by_states[0, 1] / au, fly_by_states[0, 2] / au, color='blue',
           label='Earth departure')
ax.scatter(fly_by_states[1, 0] / au, fly_by_states[1, 1] / au, fly_by_states[1, 2] / au, color='brown',
           label='Venus fly-by')
ax.scatter(fly_by_states[2, 0] / au, fly_by_states[2, 1] / au, fly_by_states[2, 2] / au, color='brown',
           label='Venus fly-by')
ax.scatter(fly_by_states[3, 0] / au, fly_by_states[3, 1] / au, fly_by_states[3, 2] / au, color='green',
           label='Earth fly-by')
ax.scatter(fly_by_states[4, 0] / au, fly_by_states[4, 1] / au, fly_by_states[4, 2] / au, color='peru',
           label='Jupiter fly-by')
ax.scatter(fly_by_states[5, 0] / au, fly_by_states[5, 1] / au, fly_by_states[5, 2] / au, color='red',
           label='Saturn arrival')
# Plot the position of the Sun
ax.scatter([0], [0], [0], color='orange', label='Sun')
# Add axis labels and limits
ax.set_xlabel('x wrt Sun [AU]')
ax.set_ylabel('y wrt Sun [AU]')
ax.set_zlabel('z wrt Sun [AU]')
ax.set_xlim([-10.5, 2.5])
ax.set_ylim([-8.5, 4.5])
ax.set_zlim([-6.5, 6.5])
# Put legend on the right
ax.legend(bbox_to_anchor=[1.15, 1])
plt.tight_layout()
plt.show()