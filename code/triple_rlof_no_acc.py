import os
import os.path
import shutil
import math
import subprocess
import numpy
import pickle
import pandas as pd
import matplotlib.pyplot as plt


from amuse.units import units, constants, nbody_system
from amuse.units.generic_unit_converter import ConvertBetweenGenericAndSiUnits
from amuse.datamodel import Particles, Particle
from amuse.io import write_set_to_file, read_set_from_file

from amuse.datamodel.particle_attributes import total_angular_momentum, kinetic_energy
from amuse.ext.orbital_elements import orbital_elements_from_binary, orbital_period_to_semimajor_axis, get_orbital_elements_from_binaries
from amuse.ext.star_to_sph import convert_stellar_model_to_SPH, StellarModelInSPH
from amuse.couple.bridge import Bridge, CalculateFieldForParticles

from amuse.community.evtwin.interface import EVtwin
from amuse.community.mesa.interface import MESA # original
from amuse.community.gadget2.interface import Gadget2
from amuse.community.twobody.twobody import TwoBody
from amuse.community.huayno.interface import Huayno

import matplotlib
#matplotlib.use("Agg")
from matplotlib import pyplot
from amuse.plot import scatter, xlabel, ylabel, plot, sph_particles_plot
from amuse.plot import pynbody_column_density_plot, HAS_PYNBODY

###################################################################
##### Function to calculate important parameters of the triple ####
###################################################################

def Kelvin_Helmholtz_timescale(m, r, L):
    t_KH = constants.G*m**2/(r*L)
    return t_KH.in_(units.yr)

def dynamical_time_scale(m, r):
    t_dyn = numpy.sqrt(r**3/(2*constants.G*m))
    return t_dyn.in_(units.yr)

def kozai_cycle_timescale(P_out,P_in,m1,m2,m3,e_out,a=1):
    t_koz = a*(P_out**2/P_in)*((m1+m2+m3)/m3)*(1-e_out**2)**(3/2)
    return t_koz.in_(units.yr)

def nuclear_time_scale(m, L):
    return 7e-4 * m * constants.c**2/L

def relaxation_time_scale(N, M, R, G=constants.G):
    return 0.138*N/numpy.log(0.4*N) * dynamical_time_scale(M, R, G=constants.G)

def octupole_term(m1,m2,a_in,a_out,e_out):
    e_oct = ((m1-m2)/(m1+m2))*(a_in/a_out)*(e_out/(1-e_out**2))
    return abs(e_oct)


######################################################################
##### Setting up working directories and direcotries for the data ####
######################################################################

def new_working_directory():
    '''
    Checks if the 'run_n' directory exists and creates a new 'run_n+1' directory for the new run, copies the python script in the new 'run_n+1' directory (so that
    a run can continue in case of interruption), creates a directory for the giant models if it does not already exists and then moves to the newly created 'run_n+1' directory.
    '''
    i = 0
    current_directory = os.getcwd()
    while os.path.exists(os.path.join(current_directory, "run_{0:=03}".format(i))):
        i += 1
    new_directory = os.path.join(current_directory, "run_{0:=03}".format(i))
    os.mkdir(new_directory)
    print("Created new directory for output:", new_directory)
    os.mkdir(os.path.join(new_directory, "plots"))
    os.mkdir(os.path.join(new_directory, "snapshots"))
    shutil.copy(__file__, new_directory)
    if not os.path.exists(os.path.join(current_directory, "giant_models")):
        os.mkdir(os.path.join(current_directory, "giant_models"))
    os.chdir(new_directory)

##########################################
# Phase one: Set up the triple system and evolve the components until the Roche lobe overflow 
##########################################

def get_relative_velocity(total_mass, semimajor_axis, ecc):
    '''
    calculates and returns the relative velocity of a star as ( GM *((1+e)/(1-e))/alpha )^0.5
    '''
    return (constants.G * total_mass * ((1.0 + ecc)/(1.0 - ecc)) / semimajor_axis).sqrt()

def set_up_inner_binary(orbital_period_1, ecc_1):
    '''
    Sets up the inner binary based on the relative orbit's parameters, and defines
    the center of mass of the inner binary:

    particles.position -= particles.center_of_mass()
    particles.velocity -= particles.center_of_mass_velocity()

    and returns the binary

    !!! Important Note: In literature the semimajor axis of a binary orbit referes to the separation between the
    two stars. If though the orbit is eccentric the semimajor-axis of the relative orbit and the separation of the two stars
    are not the same. In that case, to find the semimajor axis you need to correct the given 'semimajor-axis'
    (separation of the stars) with a factor (1-e)
    '''
    
    eccentricity = ecc_1
    masses = [3.2, 3.1] | units.MSun #3.2, 3.1
    semimajor_axis = orbital_period_to_semimajor_axis(orbital_period_1, masses[0], masses[1])
    
    print("   Initializing inner binary")
    print("   Orbital period inner binary:", orbital_period_1)
    stars =  Particles(2)
    stars.mass = masses
    stars.position = [0.0, 0.0, 0.0] | units.AU
    stars.velocity = [0.0, 0.0, 0.0] | units.km / units.s
    # If the orbit is elliptical we need a correction
    # omega=90deg
    stars[0].y = semimajor_axis*(1-eccentricity)                           
    stars[0].vx = -get_relative_velocity(stars.total_mass(), semimajor_axis, eccentricity)
    stars.move_to_center()                     # more info in /src/amuse/datamodel/particle_attributes.py
    return stars

def set_up_outer_star(inner_binary_mass, orbital_period_2, ecc_2, mutual_inclin):
    '''
    Input: inner binary combined mass 
    
    Sets up the tertiary based on the relative orbit's parameters. Takes the inner binary mass to set 
    the relative velocity of the tertiary with respect to the inner binary center of mass (the tertiary
    gravitationaly 'sees' the inner binary as a star with mass = inner_binary_mass) 
    and returns the tertiary.

    !!! Important Note: In literature the semimajor axis of a binary orbit referes to the separation between the
    two stars. If though the orbit is eccentric the semimajor-axis of the relative orbit and the separation of the two stars
    are not the same. In that case, to find the semimajor axis you need to correct the given 'semimajor-axis'
    (separation of the stars) with a factor (1-e)
    '''

    eccentricity = ecc_2   # 0.3 
    inclination = math.radians(mutual_inclin)  #  9.0
    print("   Initializing outer star")
    print("   Orbital period inner binary:", orbital_period_2)
    giant = Particle()
    giant.mass = 5.5 | units.MSun # 16
    semimajor_axis = orbital_period_to_semimajor_axis(orbital_period_2, inner_binary_mass, giant.mass)
    giant.position = semimajor_axis* (1-eccentricity) * ([math.cos(inclination), 0, math.sin(inclination)] | units.none)
    giant.velocity = get_relative_velocity(giant.mass + inner_binary_mass, 
        semimajor_axis, eccentricity) * ([0, 1, 0] | units.none)
    return giant

def set_up_initial_conditions(orbital_period_1, orbital_period_2, ecc1, ecc2, mut_inclin):
    '''
    Sets up the initial conditions for the triple system by adding the tertiary to the system.
    'stars' variable contains now the three stars but 'view_on_giant' is just the tertiary.
    It also moves the system's coordinates in to the center of mass of the triple and returns the triple and
    the tertiary alone. The local 'stars' variable is the 'triple' variable in the main code.
    '''
    stars = set_up_inner_binary(orbital_period_1, ecc1)
    giant = set_up_outer_star(stars.total_mass(), orbital_period_2, ecc2, mut_inclin)
    view_on_giant = stars.add_particle(giant) # This is just the tertiary and not the whole system
    stars.move_to_center()
    return stars, view_on_giant

def triple_set_up_info(triple, view_on_giant, view_on_giant_se, orbital_period_1, orbital_period_2, ecc_2):
    
    lines=[]

    lines.append("\nStellar Evolution Model:")
    lines.append("\nInner orbit:")

    inner_seperation = (triple[0].position-triple[1].position).lengths().in_(units.au)
    lines.append("Initial Separation (or (1-e)*semi-major axis of the relative orbit) of inner binary stars {:.2f} AU or {:.2f} RSun".format( \
        inner_seperation.value_in(units.au), inner_seperation.value_in(units.RSun)))
    
    sm_1 = ((triple-view_on_giant).center_of_mass() - triple[0].position).lengths().in_(units.au)
    lines.append("Distance of the first star from the Center of Mass {:.3f} AU or {:.2f} RSun".format(sm_1.value_in(units.au), \
                    sm_1.value_in(units.RSun)))
    sm_2 = ((triple-view_on_giant).center_of_mass() - triple[1].position).lengths().in_(units.au)
    lines.append("Distance of the second star from the Center of Mass {:.3f} AU or {:.2f} RSun".format(sm_2.value_in(units.au), \
                    sm_2.value_in(units.RSun)))
    
    lines.append("\nOuter orbit:")

    a_bin = ((triple-view_on_giant).center_of_mass() - triple.center_of_mass()).length().in_(units.au)
    lines.append("\nDistance of the binary's CoM from the triple system CoM {:.2f} AU or {:.2f} RSun".format(a_bin.value_in(units.au), \
                                                                        a_bin.value_in(units.RSun))) 
    a_giant = triple[2].position.length().in_(units.AU)      
    lines.append("Distance of the tertiary from the triple system CoM  {:0.2f} AU or {:.2f} RSun".format(a_giant.value_in(units.au), \
                                                                        a_giant.value_in(units.RSun)))
    dist_giant = (view_on_giant.position - (triple-view_on_giant).center_of_mass()).length().in_(units.au)
    lines.append("Giant's initial separation (or (1-e)*semi-major axis of the relative orbit) from binary's COM is {:.2f} AU or {:.2f} RSun".format( \
    dist_giant.value_in(units.au), dist_giant.value_in(units.RSun)))

    stop_radius_local = radius_factor * estimate_roche_radius(triple, view_on_giant)  
    lines.append("Tertiary's radius is {:.2f} RSun or {:.2f} AU, which corresponds to {:2f} * Roche lobe radious".format(stop_radius_local.value_in(units.RSun), \
                                                                     stop_radius_local.value_in(units.au),radius_factor))

    t_koz = kozai_cycle_timescale(orbital_period_2, orbital_period_1 ,triple[0].mass.value_in(units.MSun),
                      triple[1].mass.value_in(units.MSun),triple[2].mass.value_in(units.MSun),ecc_2).value_in(units.yr)
    lines.append("\nThe Lidov-Kozai  timescale is {:2f} yr".format( \
                        t_koz))
    
    oct_term = octupole_term(triple[0].mass.value_in(units.MSun),triple[1].mass.value_in(units.MSun),a_bin,a_giant,ecc_2)
    lines.append("The importance of the octupole term compared to the quadropole term is {:.2f}".format( \
        oct_term))
    
    t_dyn = dynamical_time_scale(view_on_giant_se.mass,view_on_giant_se.radius).value_in(units.day)
    t_kh = Kelvin_Helmholtz_timescale(view_on_giant_se.mass,view_on_giant_se.radius,view_on_giant_se.luminosity).value_in(units.yr)
    
    lines.append("\nThe Giant's Dynamical and Thermal timescales are {:2f} days and {:2f} yrs, respectively".format( \
                    t_dyn, t_kh))
    
    with open('triple_set_up_info.txt', 'w') as f:
        f.write('\n'.join(lines))

def giant_info(giant_model_after_relax):

    more_lines = []
    more_lines.append('\n###############################################################################')
    more_lines.append("\n SPH Model:")
    more_lines.append("Giant info after the relaxation")

    more_lines.append("Tertiary's core mass is {:.2f} MSun, while envelope's mass is {:.2f} MSun" \
        .format(giant_model_after_relax.core_particle.mass.value_in(units.MSun), \
                giant_model_after_relax.gas_particles.total_mass().value_in(units.MSun)))
    
    with open('triple_set_up_info.txt', 'a') as f:
        f.write('\n'.join(more_lines))

def triple_before_combined_run_info(giant_system_after_relax, binary, orbital_period_1, orbital_period_2,ecc_2):

    tot_mass = giant_system_after_relax.particles.total_mass()
    more_lines = []
    
    more_lines.append("\n Inner orbit:")

    inner_seperation = (binary.particles[0].position-binary.particles[1].position).lengths().in_(units.au)
    more_lines.append("Separation (or (1-e)*semi-major axis of the relative orbit) of inner binary stars {:.2f} AU or {:.2f} RSun".format( \
        inner_seperation.value_in(units.au), inner_seperation.value_in(units.RSun)))
    
    sm_1 = (binary.particles.center_of_mass() - binary.particles[0].position).lengths().in_(units.au)
    more_lines.append("Semi-major axis1 of the absolute orbit {:.2f} AU or {:.2f} RSun".format(sm_1.value_in(units.au), \
                    sm_1.value_in(units.RSun)))
    sm_2 = (binary.particles.center_of_mass() - binary.particles[1].position).lengths().in_(units.au)
    more_lines.append("Semi-major axis2 of the absolute orbit {:.2f} AU or {:.2f} RSun".format(sm_2.value_in(units.au), \
                    sm_2.value_in(units.RSun)))
    
    bin_local = binary.particles.copy()
    triple_local = giant_system_after_relax.particles.copy()
    triple_local.add_particles(bin_local)

    
    more_lines.append("\n Outer orbit:")
    a_bin = (binary.particles.center_of_mass() - triple_local.center_of_mass()).length().in_(units.au)
    more_lines.append("\nBinary system CoM semi-major axis of the absolute orbit {:.2f} AU or {:.2f} RSun".format(a_bin.value_in(units.au), \
                                                                               a_bin.value_in(units.RSun))) 
    a_giant = (giant_system_after_relax.particles.center_of_mass() - triple_local.center_of_mass()).length()      
    more_lines.append("Giant's semi-major axis of the absolute orbit is {:0.2f} AU or {:.2f} RSun".format(a_giant.value_in(units.au), \
                                                                        a_giant.value_in(units.RSun)))
    
    dist_giant = (giant_system_after_relax.particles.center_of_mass() - binary.particles.center_of_mass()).length().in_(units.au)
    more_lines.append("Giant's separation (or (1-e)*semi-major axis of the relative orbit) from binary's COM is {:.2f} AU or {:.2f} RSun".format( \
        dist_giant.value_in(units.au), dist_giant.value_in(units.RSun)))
    
    q = giant_system_after_relax.particles.mass.sum() / bin_local.mass.sum()
    # Assume ~ circular orbit:
    q13 = q**(1./3.)
    q23 = q13**2
    roche_lobe = (dist_giant*(0.49*q23/(0.6*q23+math.log(1+q13)))).as_quantity_in(units.RSun)

    stop_radius_local = radius_factor * roche_lobe
    more_lines.append("Tertiary's radius is {:.2f} RSun or {:.2f} AU, which corresponds to {:2f} * Roche lobe radious".format(stop_radius_local.value_in(units.RSun), \
                                                                     stop_radius_local.value_in(units.au),radius_factor))

    # Need to alter the values in case of different orbital periods
    t_koz = kozai_cycle_timescale(orbital_period_2, orbital_period_1, binary.particles[0].mass.value_in(units.MSun),
                                  binary.particles[1].mass.value_in(units.MSun), tot_mass.value_in(units.MSun),ecc_2).value_in(units.yr)
    more_lines.append("\nThe Lidov-Kozai  timescale is {:2f} yr".format( \
                        t_koz))
    
    oct_term = octupole_term(binary.particles[0].mass.value_in(units.MSun),binary.particles[1].mass.value_in(units.MSun),a_bin,a_giant,ecc_2)
    more_lines.append("The importance of the octupole term compared to the quadropole term is {:.2f}".format( \
        oct_term))
    
    more_lines.append('\n###############################################################################')
    more_lines.append("Giant info before initiating the final simulation")

    more_lines.append("Tertiary's core mass is {:.2f} MSun, while envelope's mass is {:.2f} MSun" \
                      .format(giant_system_after_relax.dm_particles.total_mass().value_in(units.MSun), \
                      giant_system_after_relax.gas_particles.total_mass().value_in(units.MSun)))

    with open('triple_info_before_simulation.txt', 'w') as f:
        f.write('\n'.join(more_lines))


def estimate_roche_radius(triple, view_on_giant):
    '''
    Estimates and returns the Roche radius of the tertiary. In the calculation of the semi-major axis of the 
    tertiary the inclination of the tertiary's orbit is also taken into account by the position
    of the tertiary in the 3D space. The semi-major axis is calculated as the distance from the center of mass
    of the inner binary. Hence, the inner binay is 'seen' as a point by the tertiary with the total mass of the binary.
    Furthermore, it is assumed that the tertiary's orbit is circular in the calculation
    of the Roche radius. 
    
    --> At this point a correction can be made to include the eccenticity of the tertiary, in the calculation
    of the Roche lobe 
    '''
    # 'mass ratio' of giant to inner binary
    q = (view_on_giant.mass / (triple-view_on_giant).total_mass())
    # Assume ~ circular orbit:
    a = (view_on_giant.position - (triple-view_on_giant).center_of_mass()).length()
    q13 = q**(1./3.)
    q23 = q13**2
    return (a*(0.49*q23/(0.6*q23+math.log(1+q13)))).as_quantity_in(units.RSun)

def estimate_roche_radius_bin(star1, star2):
    '''
    Estimates and returns the Roche radius of the binary components. The binary orbit is circular in 
    the calculation. 
    '''
    # 'mass ratio' of giant to inner binary
    q = (star1.mass / star2.mass)
    # Assume ~ circular orbit:
    a = (star1.position - star2.position).length()
    q13 = q**(1./3.)
    q23 = q13**2
    return (a*(0.49*q23/(0.6*q23+math.log(1+q13)))).as_quantity_in(units.RSun)

def estimate_escaping_radius(giant_star, binary_sys):
    '''
    Estimates and returns the Roche radius of the giant to use it as the escaping radius.
    The orbit is assumed circular in this calculation. 
    '''
    # 'mass ratio' of giant to inner binary
    q = (giant_star.particles.total_mass() / binary_sys.particles.total_mass())
    # Assume ~ circular orbit:
    a = (giant_star.particles.center_of_mass() - binary_sys.particles.center_of_mass()).length()
    q13 = q**(1./3.)
    q23 = q13**2
    return (a*(0.49*q23/(0.6*q23+math.log(1+q13)))).as_quantity_in(units.RSun)

def evolve_stars(triple, view_on_giant, stellar_evolution, radius_factor):
    '''
    Stellar evolution of the stars: Evolves the three stars until the more massive star's (tertiary in this case) 
    radius = 'radius_factor' (in this case=1) * 'Roche radius'. It also returns the evolved stars at the end of the 
    evolution and the .log file of the choosen stellar evolution code.
    '''
    #switch of the winds, because they are not important for this system
    stellar_evolution.parameters.AGB_wind_scheme = 0
    stellar_evolution.parameters.RGB_wind_scheme = 0
    se_giant = stellar_evolution.particles.add_particle(view_on_giant) # gives the tertiary into the stel. evol. code
    stop_radius = radius_factor * estimate_roche_radius(triple, view_on_giant)                                             
    #stellar_evolution = stellar_evolution_code(redirection='file', redirect_file='stellar_evolution_code_out.log')
    
    while (se_giant.radius < stop_radius):
        stellar_evolution.evolve_model(keep_synchronous = False)
    
    se_binary = stellar_evolution.particles.add_particles(triple - view_on_giant)
    for particle in se_binary:
        particle.evolve_for(se_giant.age)
    return stellar_evolution.particles, stellar_evolution

##########################################
# Phase two: Convert the 1D stellar evolution model to a gas particle distribution 
##########################################

def convert_giant_to_sph(view_on_se_giant, number_of_sph_particles,tar_core): #pickle_file
    '''
    Converting the 1D stellar evolution model to a gas particle distribution (see /src/amuse/ext/star_to_sph.py):
    First uses the class 'EnclosedMassInterpolator' (see /src/amuse/ext/spherical_model.py) to calculate the enclosed mass profile of the star using the radius and density profiles
    given by the Stellar evolution code. Then uses the class 'StellarModel2SPH' (see /src/amuse/ext/star_to_sph.py). At first finds the index in the enclosed mass profile (it's a numpy array)
    where the 'target_core_mass' would be added and would agree with the enclosed mass profile. Then finds the respective radius for that index from the radius profile (effectivelly corresponding 
    to the minimum core radius). Furthermore, defines a maximum core radius very close to the surface of the star. At this point, there an iterations from the minimum towards the maximum radius and
    vice versa using internal energies. I do not understand the process yet, but I think the goal is to converge to a radius (minimum radius < radius < maximum radius) which will reprsent now the
    SPH core particle. We want the SPH particle to be bigger than the physical core, without violating the density profile, in order to achieve higher resolution at the outer layers of the star.
    For yet not well understood process this paper is cited (Integrals over r**2 times the cubic spline kernel W of Monaghan & Lattanzio (1985)).


    Converts the tertiary star to a collection of SPH particles based on the output of the stellar evolution code.
    Thus, the collection of particles is characterized by the temperature, density etc profiles that characterize the evolved
    tertiary at the moment when its radius is equal with its Roche radius. It returns the the collection of the particles.
    '''
    giant_in_sph = convert_stellar_model_to_SPH(
        view_on_se_giant,                                                              # Star particle to be converted to an SPH model
        number_of_sph_particles,                                                       
        with_core_particle = True,                                                     # Model the core as a heavy, non-sph particle
        target_core_mass  = tar_core | units.MSun,     # If (with_core_particle): target mass for the non-sph particle (in paper M = 2, ?)
        do_store_composition = False                                                   # If set, store the local chemical composition on each particle 
    )
    return giant_in_sph

def relax_in_isolation(giant_in_sph, sph_code, output_base_name, mult_factor,epsilon):
    '''
    Calculates the total mass of the star as the mass of the SPH particles representing the outer layers plus
    the mass of a non-SPH particle representing the core. Calculates the total radius of the star as the position
    of the most distant SPH particle. Calculates the dynamical evolution timescale which is used to calculate
    the relaxation time scale of the system. The latter represent the time scale on which the global characteristics
    (bulk system parameters and stellar orbital elements) of the system change.
    

     --> not sure why t_end = 10*t_dyn, if t_end is the t_relax then the constant 10 seems small based on the
     t_relax = 0.138*(N/lnγN)*t_dyn, where in this code N=50000

    '''
    total_mass = giant_in_sph.gas_particles.total_mass() + giant_in_sph.core_particle.mass  
    total_radius = max(giant_in_sph.gas_particles.position.lengths_squared()).sqrt()
    dynamical_timescale = (total_radius**3 / (2 * constants.G * total_mass)).sqrt().as_quantity_in(units.day) 

    t_end = (mult_factor * dynamical_timescale).as_quantity_in(units.day)
    n_steps = 100

    softening_length = epsilon*total_radius

    hydro_code_options = dict(number_of_workers=2, redirection='file', redirect_file='hydrodynamics_code_relax_out.log')

    #unit_converter = ConvertBetweenGenericAndSiUnits(total_radius, total_mass, t_end)           # more info in /src/amuse/units/generic_unit_converter.py
    unit_converter = ConvertBetweenGenericAndSiUnits(1 | units.MSun, 1 | units.RSun, 1 | units.day)
    hydrodynamics = sph_code(unit_converter, **hydro_code_options)
    hydrodynamics.parameters.epsilon_squared = softening_length**2  # Softhing parameter for gravity calculations, removes the singularity in the inverse-square force 
    hydrodynamics.parameters.max_size_timestep = t_end
    hydrodynamics.parameters.time_max = 1.1 * t_end
    hydrodynamics.parameters.time_limit_cpu = 100.0 | units.day
    hydrodynamics.gas_particles.add_particles(giant_in_sph.gas_particles)
    hydrodynamics.dm_particles.add_particle(giant_in_sph.core_particle)
    
    potential_energies = hydrodynamics.potential_energy.as_vector_with_length(1).as_quantity_in(units.erg)
    kinetic_energies = hydrodynamics.kinetic_energy.as_vector_with_length(1).as_quantity_in(units.erg)
    thermal_energies = hydrodynamics.thermal_energy.as_vector_with_length(1).as_quantity_in(units.erg)
    energy_comparisons = (2*hydrodynamics.kinetic_energy.as_vector_with_length(1) + \
        hydrodynamics.potential_energy.as_vector_with_length(1)).as_quantity_in(units.erg)
    
    print("Relaxing for {:.2f} days or ({:.1f} * dynamical timescale)".format(t_end.value_in(units.day),mult_factor))
    times = (t_end * list(range(1, n_steps+1)) / n_steps).as_quantity_in(units.day)
    for i_step, time in enumerate(times):
        hydrodynamics.evolve_model(time)
        print("   Relaxed for:", time)
        potential_energies.append(hydrodynamics.potential_energy)
        kinetic_energies.append(hydrodynamics.kinetic_energy)
        thermal_energies.append(hydrodynamics.thermal_energy)
        energy_comparisons.append(2*hydrodynamics.kinetic_energy + hydrodynamics.potential_energy)
    
    hydrodynamics.gas_particles.copy_values_of_attributes_to(
        ['mass', 'x','y','z', 'vx','vy','vz', 'u', 'h_smooth'], 
        giant_in_sph.gas_particles)
    giant_in_sph.core_particle.position = hydrodynamics.dm_particles[0].position
    giant_in_sph.core_particle.velocity = hydrodynamics.dm_particles[0].velocity
    hydrodynamics.stop()
    
    snapshotfile = output_base_name + "_gas.amuse"
    write_set_to_file(giant_in_sph.gas_particles, snapshotfile, format='amuse')
    shutil.copy(snapshotfile, os.path.join("..", "giant_models"))
    
    snapshotfile = output_base_name + "_core.amuse"
    # temporarily store the smoothing length as core_radius on the core particle
    giant_in_sph.core_particle.radius = softening_length
    write_set_to_file(giant_in_sph.core_particle.as_set(), snapshotfile, format='amuse')
    giant_in_sph.core_particle.radius = 0.0 | units.m
    shutil.copy(snapshotfile, os.path.join("..", "giant_models"))
    
    energy_evolution_plot(times, kinetic_energies, potential_energies, thermal_energies, 
        figname = output_base_name + "_energy_evolution.png")
    virial_eq_plot(times/dynamical_timescale, energy_comparisons, \
        figname = output_base_name + "_virial_equilibrium.png")

def load_giant_model(file_base_name):
    snapshotfile = os.path.join("..", "giant_models", file_base_name + "_gas.amuse")
    sph_particles = read_set_from_file(snapshotfile, format='amuse')
    snapshotfile = os.path.join("..", "giant_models", file_base_name + "_core.amuse")
    gd_particles = read_set_from_file(snapshotfile, format='amuse')
    core_radius = gd_particles[0].radius
    gd_particles[0].radius =  0.0 | units.m
    giant_model = StellarModelInSPH(
        gas_particles = sph_particles, 
        core_particle = gd_particles[0], 
        core_radius = core_radius)
    return giant_model

##########################################
# Phase three: Coupling hydrodynamics with gravity in the evolution model 
##########################################

def update_sinks_radii(sink_cand):
    '''
    Updates the accretion radius of the sinks based on the Roche Lobe radius
    '''
    sink_cand[0].sink_radius = estimate_roche_radius_bin(sink_cand[0],sink_cand[1])
    sink_cand[1].sink_radius = estimate_roche_radius_bin(sink_cand[1],sink_cand[0])
    return sink_cand

def prepare_binary_system(dynamics_code, binary_particles):
    '''
    Prepares the binary system for the N-body code by specifing the units
    for the code.
    '''
    unit_converter = nbody_system.nbody_to_si(
        binary_particles.total_mass(),
        (binary_particles[0].position - binary_particles[1].position).length())
    system = dynamics_code(unit_converter, redirection="none")
    system.particles.add_particles(binary_particles)
    return system

def prepare_giant_system(sph_code, giant_model, view_on_giant, time_unit, n_steps):
    '''
    Prepares the tertiary for the SPH code by specifing the units
    for the code.
    '''
    hydro_code_options = dict(number_of_workers=3, 
        redirection='file', redirect_file='hydrodynamics_code_out.log')
    #unit_converter = ConvertBetweenGenericAndSiUnits(
    #    1.0 | units.RSun, 
    #    giant_model.gas_particles.total_mass() + giant_model.core_particle.mass, 
    #    time_unit)
    unit_converter = ConvertBetweenGenericAndSiUnits(1 | units.MSun, 1 | units.RSun, 1 | units.day)
    system = sph_code(unit_converter, **hydro_code_options)
    # Softening parameter for gravity calculations, removes the singularity in the inverse-square force. It contains the smoothing length defined in relax_in_isolation 
    system.parameters.epsilon_squared = giant_model.core_radius**2 

    # This gives the maximum timestep a particle may take. This should be set to a sensible value in order to protect against too large timesteps for particles with
    # very small acceleration. For cosmological simulations, the parameter given here is the maximum allowed step in the logarithm of the expansion factor. 
    # Note that the definition of MaxSizeTimestep has changed compared to Gadget-1.1 for cosmological simulations.
    system.parameters.max_size_timestep = time_unit / n_steps            

    # This sets the final time for the simulation. The code normally tries to run until this time is reached. 
    # For cosmological integrations, the value given here is the final scale factor.
    system.parameters.time_max = 1.1 * time_unit       
    # CPU-time limit for the present submission of the code. If 85 percent of this time have been reached at the end of a timestep, the code terminates itself and produces restart files.                  
    system.parameters.time_limit_cpu = 100.0 | units.day  

    giant_model.gas_particles.position += view_on_giant.position
    giant_model.gas_particles.velocity += view_on_giant.velocity
    
    giant_model.core_particle.position += view_on_giant.position
    giant_model.core_particle.velocity += view_on_giant.velocity
    
    system.gas_particles.add_particles(giant_model.gas_particles)
    system.dm_particles.add_particle(giant_model.core_particle)
    return system

def calculate_orbital_elements(m1, m2, pos1, pos2, vel1, vel2, m3, pos3, vel3):
    print("   Calculating semimajor axis and eccentricity evolution of the giant's orbit")

    m12 = m1+m2
    rel_position = (m1 * pos1 + m2 * pos2)/m12 - pos3
    rel_velocity = (m1 * vel1 + m2 * vel2)/m12 - vel3
    mtot = m12 + m3
    separation = rel_position.lengths()
    speed_squared = rel_velocity.lengths_squared()
    
    # Now calculate the important quantities:
    semimajor_axis = (constants.G * mtot * separation / 
        (2 * constants.G * mtot - separation * speed_squared)).as_quantity_in(units.AU)
    eccentricity = numpy.sqrt(1.0 - (rel_position.cross(rel_velocity).lengths()**2) / 
        (constants.G * mtot * semimajor_axis))
    return semimajor_axis, eccentricity

def evolve_coupled_system(binary_system, giant_system, giant_model, inner_binary, computational_domain_removed_particles, \
                         hydro_channels, gravity_channels, t_end, n_steps, do_energy_evolution_plot, previous_data=None):
    '''
    
    
    '''    
    directsum = CalculateFieldForParticles(particles=giant_system.particles, gravity_constant=constants.G)  # more info ./src/amuse/couple/bridge.py
    directsum.smoothing_length_squared = giant_system.parameters.gas_epsilon**2
    # The timestep of the Bridge is a critical parameter and needs to be explored
    coupled_system = Bridge(timestep=(t_end / n_steps), verbose=False, use_threading=True)
    coupled_system.add_system(binary_system, (directsum,), False)
    coupled_system.add_system(giant_system, (binary_system,), False)
    
    times = (t_end * list(range(1, n_steps+1)) / n_steps).as_quantity_in(units.day)
    
    if previous_data:
        with open(previous_data, 'rb') as file:
            (all_times, potential_energies, kinetic_energies, thermal_energies, 
                giant_center_of_mass, ms1_position, ms2_position, 
                giant_center_of_mass_velocity, ms1_velocity, ms2_velocity, giant_mass, escaped_mass) = pickle.load(file)
        all_times.extend(times + all_times[-1])
    else:
        all_times = (t_end * list(range(0, n_steps+1)) / n_steps).as_quantity_in(units.day)
        if do_energy_evolution_plot:
            potential_energies = coupled_system.particles.potential_energy().as_vector_with_length(1).as_quantity_in(units.erg)
            kinetic_energies = coupled_system.particles.kinetic_energy().as_vector_with_length(1).as_quantity_in(units.erg)
            thermal_energies = coupled_system.gas_particles.thermal_energy().as_vector_with_length(1).as_quantity_in(units.erg)
        else:
            potential_energies = kinetic_energies = thermal_energies = None
        
        giant_center_of_mass = [] | units.RSun
        ms1_position = [] | units.RSun
        ms2_position = [] | units.RSun
        giant_center_of_mass_velocity = [] | units.km / units.s
        ms1_velocity = [] | units.km / units.s
        ms2_velocity = [] | units.km / units.s
        giant_mass =[] | units.MSun
        escaped_mass = [] | units.MSun

        RL = 1.0 * estimate_escaping_radius(giant_system, binary_system)
        giant_envelope = (giant_model.gas_particles[(giant_model.gas_particles.position - giant_model.core_particle.position).lengths() <= RL]).copy()
        escaping_gas = (giant_model.gas_particles[(giant_model.gas_particles.position - giant_model.core_particle.position).lengths() > RL]).copy()

        giant_total_mass = (giant_system.dm_particles.mass + giant_envelope.total_mass())
        ms1_mass = binary_system.particles[0].mass
        ms2_mass = binary_system.particles[1].mass

        giant = giant_envelope.copy()
        giant.add_particle(giant_model.core_particle)

        giant_center_of_mass.append(giant.center_of_mass())
        ms1_position.append(binary_system.particles[0].position)
        ms2_position.append(binary_system.particles[1].position)
        giant_center_of_mass_velocity.append(giant.center_of_mass_velocity())
        ms1_velocity.append(binary_system.particles[0].velocity)
        ms2_velocity.append(binary_system.particles[1].velocity)
        giant_mass.append(giant_total_mass)
        escaped_mass.append(0 | units.MSun)

        computational_domain_removed_particles = Particles(0)

        a_giant, e_giant = calculate_orbital_elements(ms1_mass, ms2_mass,
                                                    ms1_position, ms2_position,
                                                    ms1_velocity, ms2_velocity,
                                                    giant_total_mass,
                                                    giant_center_of_mass,
                                                    giant_center_of_mass_velocity)
        print("Outer Orbit:", 0.0 | units.day,  a_giant[-1].in_(units.AU), e_giant[-1], giant_total_mass.in_(units.MSun))

        snapshotfile = os.path.join("snapshots", "hydro_triple_{0:=07}_gas.amuse".format(0))
        write_set_to_file(giant_system.gas_particles, snapshotfile, format='amuse')
        snapshotfile = os.path.join("snapshots", "hydro_triple_{0:=07}_core.amuse".format(0))
        write_set_to_file(giant_system.dm_particles, snapshotfile, format='amuse')
        snapshotfile = os.path.join("snapshots", "hydro_triple_{0:=07}_binary.amuse".format(0))
        write_set_to_file(binary_system.particles, snapshotfile, format='amuse')
        snapshotfile = os.path.join("snapshots", "hydro_triple_{0:=07}_removed_gas.amuse".format(0))
        write_set_to_file(computational_domain_removed_particles, snapshotfile, format='amuse')


        datafile = os.path.join("snapshots", "hydro_triple_{0:=07}_info.amuse".format(0))
        with open(datafile, 'wb') as outfile:
            pickle.dump((all_times[:len(giant_center_of_mass)], potential_energies, 
                kinetic_energies, thermal_energies, giant_center_of_mass,
                ms1_position, ms2_position, giant_center_of_mass_velocity,
                ms1_velocity, ms2_velocity, giant_mass, escaped_mass), outfile)

        figname1 = os.path.join("plots", "hydro_triple_small{0:=07}.png".format(0))
        figname2 = os.path.join("plots", "hydro_triple_large{0:=07}.png".format(0))
        print("  -   Hydroplots are saved to: ", figname1, "and", figname2)
        for plot_range, plot_name in [(8|units.AU, figname1), (40|units.AU, figname2)]:
            if HAS_PYNBODY:
                pynbody_column_density_plot(coupled_system.gas_particles, width=plot_range, vmin=26, vmax=32)
                scatter(coupled_system.dm_particles.x, coupled_system.dm_particles.y, c="w")
            else:
                pyplot.figure(figsize = [16, 16])
                sph_particles_plot(coupled_system.gas_particles, gd_particles=coupled_system.dm_particles, 
                    view=plot_range*[-0.5, 0.5, -0.5, 0.5])
            pyplot.savefig(plot_name)
            pyplot.close()



    i_offset = len(giant_center_of_mass)

    computational_domain_removed_particles_temp = Particles(0)
    
    print("   Evolving for", t_end)
    for i_step, time in enumerate(times):
        coupled_system.evolve_model(time)
        print("   Evolved to:", time, end=' ')

        #update the gas particles FROM the code
        hydro_channels["code_to_local_gas"].copy()
        #update the core particle FROM the code
        hydro_channels["code_to_local_core"].copy()
        #update the inner binary particles [mass,radius,vx ,vy,vz,x ,y,z] FROM the code
        gravity_channels["code_to_local"].copy()

        # remove escaped gas particles if possible, particles that are at distance > 10au from the CoM
        computational_domain_removed_particles_temp = (giant_model.gas_particles[giant_model.gas_particles.position.lengths().value_in(units.au)>7]).copy()

        if len(computational_domain_removed_particles_temp) > 0:
             # add particles to the removed particles
            computational_domain_removed_particles.add_particles(computational_domain_removed_particles_temp)
            tot_removed_mass = computational_domain_removed_particles_temp.mass.sum()
            dM_removed = tot_removed_mass
            print("\n Mass removed dM=", dM_removed.in_(units.MSun))
            # remove the lost particles
            giant_model.gas_particles.remove_particles(computational_domain_removed_particles_temp)
            # synchronize the code gas particles with the local gas particles in the code
            giant_model.gas_particles.synchronize_to(giant_system.gas_particles) 

        RL = 1.0 * estimate_escaping_radius(giant_system, binary_system)
        giant_envelope = (giant_model.gas_particles[(giant_model.gas_particles.position - giant_model.core_particle.position).lengths() <= RL]).copy()
        escaping_gas = (giant_model.gas_particles[(giant_model.gas_particles.position - giant_model.core_particle.position).lengths() > RL]).copy()

        giant_total_mass = (giant_system.dm_particles.mass + giant_envelope.total_mass())
        ms1_mass = binary_system.particles[0].mass
        ms2_mass = binary_system.particles[1].mass
        
        if do_energy_evolution_plot:
            potential_energies.append(coupled_system.particles.potential_energy())
            kinetic_energies.append(coupled_system.particles.kinetic_energy())
            thermal_energies.append(coupled_system.gas_particles.thermal_energy())

        giant = giant_envelope.copy()
        giant.add_particle(giant_model.core_particle)

        giant_center_of_mass.append(giant.center_of_mass())
        ms1_position.append(binary_system.particles[0].position)
        ms2_position.append(binary_system.particles[1].position)
        giant_center_of_mass_velocity.append(giant.center_of_mass_velocity())
        ms1_velocity.append(binary_system.particles[0].velocity)
        ms2_velocity.append(binary_system.particles[1].velocity)
        giant_mass.append(giant_total_mass)

        if len(escaping_gas) > 0:
            if len(computational_domain_removed_particles) > 0:
                escaped_mass.append(escaping_gas.total_mass() + computational_domain_removed_particles.total_mass())
            else:
                escaped_mass.append(escaping_gas.total_mass())
        else:
            if len(computational_domain_removed_particles) > 0:
                escaped_mass.append(computational_domain_removed_particles.total_mass())
            else:
                escaped_mass.append(0 | units.MSun)

        a_giant, e_giant = calculate_orbital_elements(ms1_mass, ms2_mass,
                                                      ms1_position, ms2_position,
                                                      ms1_velocity, ms2_velocity,
                                                      giant_total_mass,
                                                      giant_center_of_mass,
                                                      giant_center_of_mass_velocity)
        print("Outer Orbit:", time.in_(units.day),  a_giant[-1].in_(units.AU), e_giant[-1], giant_total_mass.in_(units.MSun))
        if (i_step + i_offset) % 10 == 0:
            snapshotfile = os.path.join("snapshots", "hydro_triple_{0:=07}_gas.amuse".format(i_step + i_offset))
            write_set_to_file(giant_system.gas_particles, snapshotfile, format='amuse')
            snapshotfile = os.path.join("snapshots", "hydro_triple_{0:=07}_core.amuse".format(i_step + i_offset))
            write_set_to_file(giant_system.dm_particles, snapshotfile, format='amuse')
            snapshotfile = os.path.join("snapshots", "hydro_triple_{0:=07}_binary.amuse".format(i_step + i_offset))
            write_set_to_file(binary_system.particles, snapshotfile, format='amuse')
            snapshotfile = os.path.join("snapshots", "hydro_triple_{0:=07}_removed_gas.amuse".format(i_step + i_offset))
            write_set_to_file(computational_domain_removed_particles, snapshotfile, format='amuse')


            datafile = os.path.join("snapshots", "hydro_triple_{0:=07}_info.amuse".format(i_step + i_offset))
            with open(datafile, 'wb') as outfile:
                pickle.dump((all_times[:len(giant_center_of_mass)], potential_energies, 
                    kinetic_energies, thermal_energies, giant_center_of_mass, 
                    ms1_position, ms2_position, giant_center_of_mass_velocity,
                    ms1_velocity, ms2_velocity, giant_mass, escaped_mass), outfile)

        figname1 = os.path.join("plots", "hydro_triple_small{0:=07}.png".format(i_step + i_offset))
        figname2 = os.path.join("plots", "hydro_triple_large{0:=07}.png".format(i_step + i_offset))
        print("  -   Hydroplots are saved to: ", figname1, "and", figname2)
        for plot_range, plot_name in [(8|units.AU, figname1), (40|units.AU, figname2)]:
            if HAS_PYNBODY:
                pynbody_column_density_plot(coupled_system.gas_particles, width=plot_range, vmin=26, vmax=32)
                scatter(coupled_system.dm_particles.x, coupled_system.dm_particles.y, c="w")
            else:
                pyplot.figure(figsize = [16, 16])
                sph_particles_plot(coupled_system.gas_particles, gd_particles=coupled_system.dm_particles, 
                    view=plot_range*[-0.5, 0.5, -0.5, 0.5])
            pyplot.savefig(plot_name)
            pyplot.close()
        
    
    coupled_system.stop()
    
    #make_movie()
    
    if do_energy_evolution_plot:
        energy_evolution_plot(all_times[:len(kinetic_energies)-1], kinetic_energies, 
            potential_energies, thermal_energies)
    
    print("   Calculating semimajor axis and eccentricity evolution for inner binary")
    # Some temporary variables to calculate semimajor_axis and eccentricity evolution
    total_mass = ms1_mass + ms2_mass
    rel_position = ms1_position - ms2_position
    rel_velocity = ms1_velocity - ms2_velocity
    separation_in = rel_position.lengths()
    speed_squared_in = rel_velocity.lengths_squared()
    
    # Now calculate the important quantities:
    semimajor_axis_binary = (constants.G * total_mass * separation_in / 
        (2 * constants.G * total_mass - separation_in * speed_squared_in)).as_quantity_in(units.AU)
    eccentricity_binary = numpy.sqrt(1.0 - (rel_position.cross(rel_velocity).lengths()**2) / 
        (constants.G * total_mass * semimajor_axis_binary))

    print("   Calculating semimajor axis and eccentricity evolution of the giant's orbit")
    # Some temporary variables to calculate semimajor_axis and eccentricity evolution
    rel_position = ((ms1_mass * ms1_position + ms2_mass * ms2_position)/total_mass - 
        giant_center_of_mass)
    rel_velocity = ((ms1_mass * ms1_velocity + ms2_mass * ms2_velocity)/total_mass - 
        giant_center_of_mass_velocity)
    total_mass += giant_total_mass
    separation = rel_position.lengths()
    speed_squared = rel_velocity.lengths_squared()

    # orbital angular momentum
    h_vec = rel_position.cross(rel_velocity).value_in(units.m**2 * units.s**-1)
    h_dire = numpy.linalg.norm(h_vec, axis=1)
    
    # Now calculate the important quantities:
    semimajor_axis_giant = (constants.G * total_mass * separation / 
        (2 * constants.G * total_mass - separation * speed_squared)).as_quantity_in(units.AU)
    eccentricity_giant = numpy.sqrt(1.0 - (rel_position.cross(rel_velocity).lengths()**2) / 
        (constants.G * total_mass * semimajor_axis_giant))
    
    mutual_inclination = numpy.degrees(numpy.arccos(h_vec[:,2] / h_dire))
    
    orbit_incl_plot(mutual_inclination,all_times[:len(eccentricity_binary)])
    
    orbit_parameters_plot(semimajor_axis_binary, semimajor_axis_giant, all_times[:len(semimajor_axis_binary)])
    orbit_ecc_plot(eccentricity_binary, eccentricity_giant, all_times[:len(eccentricity_binary)])
    
    orbit_parameters_plot(separation_in.as_quantity_in(units.AU), 
        separation.as_quantity_in(units.AU), 
        all_times[:len(eccentricity_binary)], 
        par_symbol="r", par_name="separation")
    orbit_parameters_plot(speed_squared_in.as_quantity_in(units.km**2 / units.s**2), 
        speed_squared.as_quantity_in(units.km**2 / units.s**2), 
        all_times[:len(eccentricity_binary)], 
        par_symbol="v^2", par_name="speed_squared")

    mass_plot(giant_mass.as_quantity_in(units.MSun),all_times[:len(eccentricity_binary)],figname="giant_mass")
    mass_plot(escaped_mass.as_quantity_in(units.MSun),all_times[:len(eccentricity_binary)],figname="lost_mass")

def make_movie():
    print("   Creating movie from snapshots")
    try:
        subprocess.call(['mencoder', "mf://hydro_triple_small*.png", '-ovc', 'lavc', 
            '-o', '../hydro_triple_small.avi', '-msglevel', 'all=1'], cwd="./plots")
        subprocess.call(['mencoder', "mf://hydro_triple_large*.png", '-ovc', 'lavc', 
            '-o', '../hydro_triple_large.avi', '-msglevel', 'all=1'], cwd="./plots")
    except Exception as exc:
        print("   Failed to create movie, error was:", str(exc))

def continue_evolution(sph_code, dynamics_code, t_end, n_steps, 
        relaxed_giant_output_base_name, do_energy_evolution_plot):
    print("Loading snapshots...", end=' ')
    files = os.listdir("snapshots")
    files.sort()
    files = files[-5:]
    print(files)
    binary = read_set_from_file(os.path.join("snapshots", files[0]), format='amuse')
    gd_particles = read_set_from_file(os.path.join("snapshots", files[1]), format='amuse')
    sph_particles = read_set_from_file(os.path.join("snapshots", files[2]), format='amuse')

    if os.path.exists(os.path.join("snapshots", files[4])):
        computational_domain_removed_particles = read_set_from_file(os.path.join("snapshots", files[4]), format='amuse')

    snapshotfile = os.path.join("..", "giant_models", relaxed_giant_output_base_name + "_core.amuse")
    core_particle = read_set_from_file(snapshotfile, format='amuse')

    giant_model = StellarModelInSPH(
        gas_particles = sph_particles, 
        core_particle = gd_particles[0], 
        core_radius = core_particle.radius)
    
    view_on_giant = Particle()
    view_on_giant.position = [0]*3 | units.m
    view_on_giant.velocity = [0]*3 | units.m / units.s
    
    print("\nSetting up {0} to simulate inner binary system".format(dynamics_code.__name__))
    binary_system = prepare_binary_system(dynamics_code, binary)
    
    gravity_channels = {}
    gravity_channels["local_to_code"] = binary.new_channel_to(binary_system.particles)
    gravity_channels["code_to_local"] = binary_system.particles.new_channel_to(binary)
    
    print("\nSetting up {0} to simulate giant in SPH".format(sph_code.__name__))
    giant_system = prepare_giant_system(sph_code, giant_model, view_on_giant, t_end, n_steps)

    hydro_channels = {}
    hydro_channels["local_to_code"] = giant_model.gas_particles.new_channel_to(giant_system.gas_particles)
    hydro_channels["code_to_local_gas"] = giant_system.gas_particles.new_channel_to(giant_model.gas_particles)
    hydro_channels["code_to_local_core"] = giant_system.dm_particles.new_channel_to(giant_model.core_particle.as_set())

    
    print('\nLoading binary components',binary_system.particles)
    print('\nLoading Giant core particle',giant_system.dm_particles)
    
    print("\nEvolving with bridge between", sph_code.__name__, "and", dynamics_code.__name__)
    evolve_coupled_system(binary_system, giant_system, giant_model, binary, computational_domain_removed_particles, \
                          hydro_channels,gravity_channels, \
                          t_end, n_steps, do_energy_evolution_plot,\
                          previous_data = os.path.join("snapshots", files[3]))
    print("Done")

def energy_evolution_plot(time, kinetic, potential, thermal, figname = "energy_evolution.png"):
    time.prepend(0.0 | units.day)
    pyplot.figure(figsize = (5, 5))
    plot(time, kinetic, label='K')
    plot(time, potential, label='U')
    plot(time, thermal, label='Q')
    plot(time, kinetic + potential + thermal, label='E')
    xlabel('Time')
    ylabel('Energy')
    pyplot.legend(prop={'size':"x-small"}, loc=4)
    pyplot.savefig(figname)
    pyplot.close()

def virial_eq_plot(time, energy_comp, figname = "virial_equilibrium.png"):
    pyplot.figure(figsize = (7, 7), dpi=150)
    plot(time, energy_comp, label='2K + U')
    xlabel(r'Time [$t_{dyn}$]', fontsize=15)
    ylabel('Energy', fontsize=15)
    #pyplot.legend(prop={'size':"x-small"}, loc=4)
    pyplot.savefig(figname, bbox_inches='tight')
    pyplot.close()

def orbit_ecc_plot(eccentricity_in,eccentricity_out,time):
    figure = pyplot.figure(figsize = (10, 6), dpi = 100)
    subplot = figure.add_subplot(2, 1, 1)
    plot(time,eccentricity_in)
    xlabel('t')
    ylabel('e$_\mathrm{binary}$')
    
    subplot = figure.add_subplot(2, 1, 2)
    plot(time,eccentricity_out )  
    xlabel('t')
    ylabel('e$_\mathrm{giant}$')
    pyplot.minorticks_on()
    pyplot.savefig("eccentricity_evolution.png")
    pyplot.close()

def orbit_incl_plot(mutual_incl,time):
    figure = pyplot.figure(figsize = (10, 6), dpi = 100)
    plot(time,mutual_incl)
    xlabel('t')
    ylabel('i$_\mathrm{mutual}$ (deg)')
    pyplot.savefig("mutual_inclination_evolution.png")
    pyplot.close()


def orbit_parameters_plot(semi_major_in,semi_major_out, time, par_symbol="a", par_name="semimajor_axis"):
    figure = pyplot.figure(figsize = (10, 6), dpi = 100)
    subplot = figure.add_subplot(2, 1, 1)
    plot(time,semi_major_in )
    xlabel('t')
    ylabel('$'+par_symbol+'_\mathrm{binary}$')
    
    subplot = figure.add_subplot(2, 1, 2)
    plot(time,semi_major_out )
    xlabel('t')
    ylabel('$'+par_symbol+'_\mathrm{giant}$')
    pyplot.minorticks_on()
    pyplot.savefig(par_name+"_evolution.png")
    pyplot.close()


def mass_plot(mass,time,figname):
    figure = pyplot.figure(figsize = (10, 6), dpi = 100)
    plot(time,mass)
    xlabel('t')
    ylabel('M')
    pyplot.savefig(figname+"_evolution.png")
    pyplot.close()

if __name__ == "__main__":
    # community codes to be used
    stellar_evolution_code = MESA(version='2208')
    sph_code = Gadget2
    dynamics_code = Huayno

    # The number of SPH particles determines the resolution of the simulations, this needs to be explored
    number_of_sph_particles = 50000
    # Stop stellar evolution when giant's radius is (radius_factor * Roche lobe radius)
    radius_factor = 1.1
    epsilon = 0.16
    relaxed_giant_output_base_name = "relaxed_giant_" + str(number_of_sph_particles) + "_R" + str(radius_factor) +  \
        "_e" + str(epsilon) 
    t_end = 360 | units.day #14600
    n_steps = 3600  # 146000
    
    do_energy_evolution_plot = True
    
    if os.path.exists("snapshots"):
        '''
        If there is a 'snapshots' directory continue the evolution from that point
        '''
        print("Found snapshots folder, continuing evolution of previous run")
        continue_evolution(sph_code, dynamics_code, t_end, n_steps, 
            relaxed_giant_output_base_name, do_energy_evolution_plot)
        exit(0)
    
    new_working_directory()
    # Some parameters of the two orbits
    period_orbit_1 = 7.15 | units.day
    period_orbit_2 = 145.5 | units.day
    eccentricity_orbit_1 = 0.0
    eccentricity_orbit_2 = 0.15
    mutual_incliantion = 0.0
    
    print("Initializing triple")
    triple, view_on_giant = set_up_initial_conditions(period_orbit_1, period_orbit_2, \
                            eccentricity_orbit_1, eccentricity_orbit_2, mutual_incliantion)
    print("\nInitialization done:\n", triple)
    
    #print("\nEvolving with", stellar_evolution_code.__name__) 
    se_stars, se_code_instance = evolve_stars(triple, view_on_giant, stellar_evolution_code, radius_factor)
    # Return the new radii for the stars
    triple[2].radius = se_stars[0].radius
    triple[0].radius = se_stars[1].radius
    triple[1].radius = se_stars[2].radius

    triple[2].mass = se_stars[0].mass
    triple[0].mass = se_stars[1].mass
    triple[1].mass = se_stars[2].mass
    print("\nAfter stellar evolution done star:\n", triple)
    
    # relaxation duration = mult_factor*dynamical_timescale of the giant
    mult_factor = 10.0
    core_mult_f = 0.255

    inner_binary = triple - view_on_giant
    #inner_binary = se_stars - se_stars[0] #takes in to account the lost mass due to winds for the inner binary

    if os.path.exists(os.path.join("..", "giant_models", relaxed_giant_output_base_name + "_gas.amuse")):
        '''
        Check if there is a model (in a folder called 'giant_models') representing the relaxed giant already
        and load it instead of creating a new one
        '''
        print("\nLoading SPH model for giant from:", end=' ') 
        print(os.path.join("..", "giant_models", relaxed_giant_output_base_name + "_gas.amuse"))
        giant_model = load_giant_model(relaxed_giant_output_base_name)
        se_code_instance.stop()
    else:
        '''
        If there is no relaxed giant model representing the giant already, create a new one and relax it
        '''
        print("\nConverting giant to", number_of_sph_particles, "SPH particles")
        view_on_se_giant = view_on_giant.as_set().get_intersecting_subset_in(se_stars)[0]
        triple_set_up_info(triple, view_on_giant, view_on_se_giant, period_orbit_1, period_orbit_2, eccentricity_orbit_2)
        target_core = core_mult_f * view_on_giant.mass.value_in(units.MSun)
        giant_model = convert_giant_to_sph(view_on_se_giant, number_of_sph_particles,target_core)
        se_code_instance.stop()
        print("Relaxing giant with", sph_code.__name__)
        relax_in_isolation(giant_model, sph_code, relaxed_giant_output_base_name, mult_factor, epsilon)
        giant_info(giant_model)

    
    print("\nSetting up {0} to simulate inner binary system".format(dynamics_code.__name__))
    binary_system = prepare_binary_system(dynamics_code, inner_binary)

    # Create a channel between the inner_binary particles (python) and the binary_system.particles (Huayno)

    gravity_channels = {}
    gravity_channels["local_to_code"] = inner_binary.new_channel_to(binary_system.particles)
    gravity_channels["code_to_local"] = binary_system.particles.new_channel_to(inner_binary)
    
    print("\nSetting up {0} to simulate giant in SPH".format(sph_code.__name__))
    giant_system = prepare_giant_system(sph_code, giant_model, view_on_giant, t_end, n_steps)

    hydro_channels = {}
    hydro_channels["local_to_code"] = giant_model.gas_particles.new_channel_to(giant_system.gas_particles)
    hydro_channels["code_to_local_gas"] = giant_system.gas_particles.new_channel_to(giant_model.gas_particles)
    hydro_channels["code_to_local_core"] = giant_system.dm_particles.new_channel_to(giant_model.core_particle.as_set())
    triple_before_combined_run_info(giant_system, binary_system, period_orbit_1, period_orbit_2, eccentricity_orbit_2)
    computational_domain_removed_particles = Particles(0)

    #t_end = (7500 * dynamical_time_scale(triple[2].mass,triple[2].radius).value_in(units.day)) | units.day
    # t_end*7e-6
    
    print("\nEvolving with bridge between", sph_code.__name__, "and", dynamics_code.__name__)
    evolve_coupled_system(binary_system, giant_system, giant_model, inner_binary, computational_domain_removed_particles, \
                          hydro_channels, gravity_channels, t_end, n_steps, do_energy_evolution_plot, previous_data=None)
    print("Done")
