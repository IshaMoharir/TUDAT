##################### IMPORT STATEMENTS #####################
    # - Loads the standard modules
    # - Loads the tudatpy modules
############################################################

import numpy as np
import matplotlib as plt
from matplotlib import pyplot as plt

from tudatpy.interface import spice
from tudatpy import numerical_simulation
from tudatpy.numerical_simulation import environment
from tudatpy.numerical_simulation import environment_setup, propagation_setup
from tudatpy.astro import element_conversion
from tudatpy import constants
from tudatpy.kernel.interface import spice_interface
from tudatpy.util import result2array
from tudatpy.astro.time_conversion import DateTime



##################### CONFIGURATION #####################
    # - Load spice kernels
    # - Set simulation start and end epoch
    # - Set simulation time step
############################################################
# Load spice kernels
spice.load_standard_kernels()

date = "2034-01-01T00:00:00"
epoch = spice_interface.convert_date_string_to_ephemeris_time(date)

##################### ENVIRONMENT SETUP #####################
    # - Create bodies
    # - Create body settings
    # - Create system of bodies
############################################################

# Create bodies
bodies_to_create = ["Earth", "Sun"]
body_settings = environment_setup.get_default_body_settings(bodies_to_create)

# Create system of bodies
bodies = environment_setup.create_system_of_bodies(body_settings)

##################### VEHICLE SETUP #####################
    # - Create vehicle
    # - Create vehicle settings
    # - Create vehicle object
############################################################

# Create vehicle
bodies.create_empty_body("NIBIRU")
bodies.get("NIBIRU").mass = 100

# Add "Planet 9" to the system
bodies.create_empty_body("Planet9")

# Define Keplerian elements for "Planet 9"
a_planet9 = 400 * constants.ASTRONOMICAL_UNIT  # Semi-major axis
keplerian_elements = np.array([a_planet9, 0.0, 0.0, 0.0, 0.0, 0.0])

# Convert Keplerian elements to Cartesian coordinates using the Sun's gravitational parameter
state = element_conversion.keplerian_to_cartesian(
    keplerian_elements=keplerian_elements,
    gravitational_parameter=bodies.get_body("Sun").gravitational_parameter
)

# Set ephemeris directly to "Planet 9"
ephemeris_settings = environment_setup.ephemeris.constant(state, "Sun")
bodies.get_body("Planet9").ephemeris(ephemeris_settings)

# Define initial state of spacecraft in a low Earth orbit
earth_radius = spice_interface.get_average_radius("Earth")
altitude = 200e3  # 200 km
velocity_earth_orbit = np.sqrt(bodies.get_body("Earth").gravitational_parameter / (earth_radius + altitude))
initial_state = np.array([earth_radius + altitude, 0, 0, 0, velocity_earth_orbit, 0])

# Define propagator settings
propagator_settings = propagation_setup.propagator.translational(
    central_bodies=["Earth"],
    bodies_to_propagate=["NIBIRU"],
    acceleration_models=propagation_setup.create_acceleration_models(
        bodies,
        {"NIBIRU": {"Earth": [propagation_setup.acceleration.point_mass_gravity()]}},
        ["NIBIRU"],
        ["Earth"]
    ),
    initial_states=initial_state,
    start_epoch=epoch
)

# Define integrator settings
integrator_settings = propagation_setup.integrator.runge_kutta_4(
    initial_time=epoch,
    step_size=300.0
)

# Create the dynamics simulator
dynamics_simulator = numerical_simulation.SingleArcSimulator(
    bodies, integrator_settings, propagator_settings
)

# Retrieve the state history
state_history = dynamics_simulator.state_history

# Calculate delta-v needed for Hohmann transfer
# Gravitational parameter of the Sun (mu = GM)
mu_sun = bodies.get_body("Sun").gravitational_parameter

# Orbital radii
r1 = earth_radius + altitude  # Orbit radius of Earth plus altitude of spacecraft
r2 = 400 * constants.ASTRONOMICAL_UNIT  # Orbit radius of Planet 9 (400 AU)

# Semi-major axis of Hohmann transfer orbit
a_transfer = (r1 + r2) / 2

# Velocity at departure orbit and at entry into transfer orbit
v1 = np.sqrt(mu_sun / r1)
v_transfer1 = np.sqrt(mu_sun * (2/r1 - 1/a_transfer))

# Velocity at arrival orbit and at exit from transfer orbit
v2 = np.sqrt(mu_sun / r2)
v_transfer2 = np.sqrt(mu_sun * (2/r2 - 1/a_transfer))

# Calculate delta-v at Earth departure and Planet 9 arrival
delta_v1 = np.abs(v_transfer1 - v1)
delta_v2 = np.abs(v2 - v_transfer2)

# Calculate the time of flight for half the Hohmann transfer orbit (pi * sqrt(a^3 / mu))
tof = np.pi * np.sqrt(a_transfer**3 / mu_sun)

# Print out the results
print(f"Delta-v for Earth departure: {delta_v1:.2f} m/s")
print(f"Delta-v for arrival at Planet 9: {delta_v2:.2f} m/s")
print(f"Time of flight to reach Planet 9: {tof / constants.JULIAN_DAY:.2f} days")



# Extract positions for plotting
times = np.array(list(state_history.keys()))
positions = np.array([state[0:3] for state in state_history.values()])

plt.figure(figsize=(10, 7))
plt.plot(positions[:, 0], positions[:, 1])
plt.scatter([0], [0], color='yellow')  # Plot the Sun
plt.title('Trajectory of NIBIRU')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.grid(True)
plt.axis('equal')
plt.show()
