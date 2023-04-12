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
from amuse.units.core import enumeration_unit
from amuse.units.generic_unit_converter import ConvertBetweenGenericAndSiUnits
from amuse.datamodel import Particles, Particle, ParticlesSuperset
from amuse.io import write_set_to_file, read_set_from_file
from amuse.support.exceptions import AmuseException

from amuse.datamodel.particle_attributes import total_angular_momentum, kinetic_energy
from amuse.ext.orbital_elements import orbital_elements_from_binary, orbital_period_to_semimajor_axis, get_orbital_elements_from_binaries
from amuse.ext.star_to_sph import convert_stellar_model_to_SPH, StellarModelInSPH
from amuse.ext.sink import new_sink_particles
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
from amuse.plot import scatter, xlabel, ylabel, plot,loglog,semilogx,semilogy, sph_particles_plot
from amuse.plot import pynbody_column_density_plot, HAS_PYNBODY


def dynamical_time_scale(m, r, G=constants.G):
    return numpy.sqrt(r**3/(2*constants.G*m))

def relaxation_time_scale(N, M, R, G=constants.G):
    return 0.138*N/numpy.log(0.4*N) * dynamical_time_scale(M, R, G=constants.G)

def get_relative_velocity(total_mass, semimajor_axis, ecc):
    '''
    calculates and returns the relative velocity of a star as ( GM *((1+e)/(1-e))/alpha )^0.5
    '''
    return (constants.G * total_mass * ((1.0 + ecc)/(1.0 - ecc)) / semimajor_axis).sqrt()

def set_up_inner_binary():
    '''
    Sets up the inner binary based on the relative orbit's parameters, and defines
    the center of mass of the inner binary:

    particles.position -= particles.center_of_mass()
    particles.velocity -= particles.center_of_mass_velocity()

    and returns the binary
    '''
    #semimajor_axis = 0.048273275517640366 | units.AU  # 0.133256133158 
    orbital_period = 1.10 | units.day
    eccentricity = 0
    masses = [6.5, 5.9] | units.MSun #3.2,3.1
    semimajor_axis = orbital_period_to_semimajor_axis(orbital_period, masses[0], masses[1])
    
    #orbital_period = (4 * numpy.pi**2 * semimajor_axis**3 / 
    #    (constants.G * masses.sum())).sqrt().as_quantity_in(units.day)
    
    print("   Initializing inner binary")
    print("   Orbital period inner binary:", orbital_period)
    stars =  Particles(2)
    stars.mass = masses
    stars.position = [0.0, 0.0, 0.0] | units.AU
    stars.velocity = [0.0, 0.0, 0.0] | units.km / units.s
    stars[0].x = semimajor_axis
    stars[0].vy = get_relative_velocity(stars.total_mass(), semimajor_axis, eccentricity)
    stars.move_to_center()                     # more info in /src/amuse/datamodel/particle_attributes.py
    return stars

def set_up_outer_star(inner_binary_mass):
    '''
    Input: inner binary combined mass 
    
    Sets up the tertiary based on the relative orbit's parameters. Takes the inner binary mass to set 
    the relative velocity of the tertiary with respect to the inner binary center of mass (the tertiary
    gravitationaly 'sees' the inner binary as a star with mass = inner_binary_mass) 
    and returns the tertiary.
    '''
    #outer_orbital_sep = 0.8274288819130811 | units.AU  # 1.22726535008
    orbital_period = 52.04 | units.day
    eccentricity = 0.3     #0.15
    inclination = math.radians(16.8)  # 9.0
    
    print("   Initializing outer star")
    giant = Particle()
    giant.mass = 16 | units.MSun # 5.5
    outer_orbital_sep = orbital_period_to_semimajor_axis(orbital_period, giant.mass, inner_binary_mass)
    giant.position = outer_orbital_sep * ([math.cos(inclination), 0, math.sin(inclination)] | units.none)
    giant.velocity = get_relative_velocity(giant.mass + inner_binary_mass, 
        outer_orbital_sep, eccentricity) * ([0, 1, 0] | units.none)
    print("   Initializing outer star")
    print("   Orbital period outer binary:", orbital_period.value_in(units.day))
    return giant

def set_up_initial_conditions():
    '''
    Sets up the initial conditions for the triple system by adding the tertiary to the system.
    'stars' variable contains now the three stars but 'view_on_giant' is just the tertiary.
    It also moves the system's coordinates in to the center of mass of the triple and returns the triple and
    the tertiary alone. The local 'stars' variable is the 'triple' variable in the main code.
    '''
    stars = set_up_inner_binary()
    giant = set_up_outer_star(stars.total_mass())
    view_on_giant = stars.add_particle(giant) # This is just the tertiary and not the whole system
    stars.move_to_center()
    return stars, view_on_giant
def triple_set_up_info(triple, view_on_giant):
    
    print("\n Inner orbit:")
    inner_seperation = (triple[0].position-triple[1].position).lengths().in_(units.au)
    print("Orbital Separation of inner binary stars {:.2f} AU or {:.2f} RSun".format( \
        inner_seperation.value_in(units.au), inner_seperation.value_in(units.RSun)))
    
    sm_1 = ((triple-view_on_giant).center_of_mass() - triple[0].position).lengths().in_(units.au)
    print("Semi-major axis1 {:.2f} AU or {:.2f} RSun".format(sm_1.value_in(units.au), \
                    sm_1.value_in(units.RSun)))
    sm_2 = ((triple-view_on_giant).center_of_mass() - triple[1].position).lengths().in_(units.au)
    print("Semi-major axis1 {:.2f} AU or {:.2f} RSun".format(sm_2.value_in(units.au), \
                    sm_2.value_in(units.RSun)))
    
    print("\n Outer orbit:")
    a_bin = ((triple-view_on_giant).center_of_mass() - triple.center_of_mass()).length().in_(units.au)
    print("\nBinary system semi-major axis is {:.2f} AU or {:.2f} RSun".format(a_bin.value_in(units.au), \
                                                                               a_bin.value_in(units.RSun))) 
    a_giant = triple[2].position.length().in_(units.AU)      
    print("Giant's semi-major is axis {:0.2f} AU or {:.2f} RSun".format(a_giant.value_in(units.au), \
                                                                        a_giant.value_in(units.RSun)))
    dist_giant = (view_on_giant.position - (triple-view_on_giant).center_of_mass()).length().in_(units.au)
    print("Giant's distance from binary's COM is {:.2f} AU or {:.2f} RSun".format(dist_giant.value_in(units.au), \
                                                                                  dist_giant.value_in(units.RSun)))


def estimate_roche_radius(triple, view_on_giant):
    '''
    Estimates and returns the Roche radius of the tertiary. In the calculation of the semi-major axis of the 
    tertiary the inclination of the tertiary's orbit is also taken into account by the position
    of the tertiary in the 3D space, but the the inner binay is 'seen' as a mass point by the
    tertiary. Furthermore, it is assumed that the tertiary's orbit is circular in the calculation
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


def evolve_stars(triple, view_on_giant, stellar_code,radius_factor):
    se_giant = stellar_code.particles.add_particle(view_on_giant)
    
    stop_radius = radius_factor * estimate_roche_radius(triple, view_on_giant)
    print('Roche Lobe radius of the tertiary ={:.2f} au'.format(stop_radius.value_in(units.au)))
    
    while se_giant.radius < stop_radius.in_(units.RSun):
        stellar_code.evolve_model(keep_synchronous = False)#, **stellar_code_options)
        print(se_giant.mass.in_(units.MSun), se_giant.wind.in_(units.MSun / units.yr), se_giant.age.in_(units.Myr),
             se_giant.radius.in_(units.RSun), se_giant.stellar_type)

    return stellar_code.particles, stellar_code

def energy_evolution_plot(time, kinetic, potential, thermal, figname = "energy_evolution.png"):
    time.prepend(0.0 | units.day)
    pyplot.figure(figsize = (6, 6))
    plot(time, kinetic, label='Kinetic En.')
    plot(time, potential, label='Potential En.')
    plot(time, thermal, label='Thermal En')
    plot(time, kinetic + potential + thermal, label='Total En.')
    xlabel('Time')
    ylabel('Energy')
    pyplot.legend(prop={'size':"x-small"}, loc=4)
    pyplot.savefig(figname)
    pyplot.close()
    
def virial_eq_plot(time, energy_comp, figname = "virial_equilibrium.png"):
    pyplot.figure(figsize = (6, 6))
    plot(time, energy_comp, label='2K + U')
    xlabel(r'Time [$t_{dyn}$]')
    ylabel('Energy')
    pyplot.legend(prop={'size':"x-small"}, loc=4)
    pyplot.savefig(figname)
    pyplot.close()
      
    

def convert_giant_to_sph(view_on_se_giant, number_of_sph_particles,tar_core): #pickle_file
    '''
    Converting the 1D stellar evolution model to a gas particle distribution       # more info /src/amuse/ext/star_to_sph.py

    Converts the tertiary star to a collection of SPH particles based on the output of the stellar evolution code.
    Thus, the collection of particles is characterized by the temperature, density etc profiles that characterize the evolved
    tertiary at the moment when its radius is equal with its Roche radius. It returns the the collection of the particles.
    '''
    giant_in_sph = convert_stellar_model_to_SPH(
        view_on_se_giant,                                                              # Star particle to be converted to an SPH model
        number_of_sph_particles,                                                       
        with_core_particle = True,                                                     # Model the core as a heavy, non-sph particle
        target_core_mass  = tar_core | units.MSun,                  # 1.4 | units.MSun      # If (with_core_particle): target mass for the non-sph particle (in paper M = 2, ?)
        do_store_composition = False                                                   # If set, store the local chemical composition on each particle  
    )
    return giant_in_sph


def relax_in_isolation(giant_in_sph, sph_code, enc_mass_prof, mult_factor, epsilon, where_to_save):
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
    #print(total_mass.in_(units.MSun),total_radius.in_(units.RSun))
    dynamical_timescale = (total_radius**3 / (2 * constants.G * total_mass)).sqrt().as_quantity_in(units.day)
    t_end = (mult_factor * dynamical_timescale).as_quantity_in(units.day) 
    #t_end = 0.138*(N/numpy.log(0.4*N)) * dynamical_time_scale
    n_steps = 100
    hydro_code_options = dict(number_of_workers=3, redirection='file', redirect_file='hydrodynamics_code_relax_out.log')

    unit_converter = ConvertBetweenGenericAndSiUnits(total_radius, total_mass, t_end)           # more info in /src/amuse/units/generic_unit_converter.py
    hydrodynamics = sph_code(unit_converter, **hydro_code_options)
    hydrodynamics.parameters.epsilon_squared = (epsilon*total_radius)**2  # Softening removes the singularity in the inverse-square force
    hydrodynamics.parameters.max_size_timestep = t_end
    hydrodynamics.parameters.time_max = 1.1 * t_end
    hydrodynamics.parameters.time_limit_cpu = 7.0 | units.day
    hydrodynamics.gas_particles.add_particles(giant_in_sph.gas_particles)
    hydrodynamics.dm_particles.add_particle(giant_in_sph.core_particle)
    
    print('Smoothing Length = {:.4f} RSun'.format((hydrodynamics.parameters.epsilon_squared**(1/2)).value_in(units.RSun)))
    
    potential_energies = hydrodynamics.potential_energy.as_vector_with_length(1).as_quantity_in(units.erg)
    kinetic_energies = hydrodynamics.kinetic_energy.as_vector_with_length(1).as_quantity_in(units.erg)
    thermal_energies = hydrodynamics.thermal_energy.as_vector_with_length(1).as_quantity_in(units.erg)
    energy_comparisons = (2*hydrodynamics.kinetic_energy.as_vector_with_length(1) + \
        hydrodynamics.potential_energy.as_vector_with_length(1)).as_quantity_in(units.erg)
    
    
    print("Relaxing for {:.2f} ({:.0f} * dynamical timescale)".format(t_end.value_in(units.day),mult_factor))
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
    
    plot_giant(giant_in_sph.gas_particles,giant_in_sph.core_particle, mult_factor, where_to_save, epsilon)
    
    radii, densities = tertiary_profiles(giant_in_sph.gas_particles,giant_in_sph.core_particle, True)
    
    plt_dens_profiles(giant_rad_profile, giant_dens_profile, radii, densities, enc_mass_prof, mult_factor,
                      giant_in_sph.core_particle.mass.value_in(units.MSun), where_to_save, epsilon)
    
    #print((hydrodynamics.parameters.epsilon_squared**(1/2)).in_(units.RSun))
    hydrodynamics.stop()
    
    energy_evolution_plot(times, kinetic_energies, potential_energies, thermal_energies,  
                    figname = where_to_save + "/energy_evolution_eps{:.2f}.png".format(epsilon))
    virial_eq_plot(times/dynamical_timescale, energy_comparisons, \
                    figname = where_to_save + "/virial_equilibrium_eps{:.4f}.png".format(epsilon))
    
    
def amuse_mesa_profiles(tertiary):
    """
    Return all the profiles of the tertiary calculated by the stellar evolution code
    """
    giant_num_zones = numpy.arange(1,tertiary.get_number_of_zones()+1)
    giant_dens_profile = tertiary.get_density_profile(number_of_zones = tertiary.get_number_of_zones())
    giant_temp_profile = tertiary.get_temperature_profile(number_of_zones = tertiary.get_number_of_zones())
    giant_rad_profile = tertiary.get_radius_profile(number_of_zones = tertiary.get_number_of_zones())
    giant_mass_profile = tertiary.get_mass_profile(number_of_zones = tertiary.get_number_of_zones())
    return giant_num_zones, giant_dens_profile, giant_temp_profile, giant_rad_profile, giant_mass_profile

class EnclosedMassInterpolator(object):
    """
    Interpolator used in 'get_enclosed_mass_from_tabulated' and 'get_radius_for_enclosed_mass'.
    These two functions are required for 'new_spherical_particle_distribution'.
    """
    def __init__(self, radii = None, densities = None, core_radius = None):
        self.initialized = False
        self.four_thirds_pi = numpy.pi * 4.0/3.0
        if (radii and densities):
            self.initialize(radii, densities, core_radius = core_radius)
    
    def initialize(self, radii, densities, core_radius = None):
        self.sort_density_and_radius(densities*1.0, radii*1.0, core_radius = core_radius)
        self.calculate_enclosed_mass_table()
        self.initialized = True
        
    def sort_density_and_radius(self, densities, radii, core_radius = None):
        self.radii, self.densities = radii.sorted_with(densities)
        self.radii.prepend(core_radius or 0 | units.m)
    
    def calculate_enclosed_mass_table(self):
        self.radii_cubed = self.radii**3
        self.enclosed_mass = [0.0] | units.kg
        for rho_shell, r3_in, r3_out in zip(self.densities, self.radii_cubed, self.radii_cubed[1:]):
            self.enclosed_mass.append(self.enclosed_mass[-1] + rho_shell * (r3_out - r3_in))
        self.enclosed_mass = self.four_thirds_pi * self.enclosed_mass
        
def intersting_subset(part_collect,num):
    return print(part_collect.as_set().get_intersecting_subset_in(part_collect)[num])

def den_profile(gas,core,mesa_num_of_zones,shell_edges,shell_cents):
    
    # gonna store here the mass of each shell
    shells_mass = numpy.zeros(mesa_num_of_zones) #one less because the core will be added as a compact shell later
    shells_volume = shell_edges**3
    shells_dr = shells_volume[1:]-shells_volume[:-1]
    # finds in which shell each particle belongs to. Need to make this faster
    for dist,mass in zip(gas.dist_from_cm.value_in(units.RSun),gas.mass.value_in(units.kg)):
        index = numpy.where((shell_edges[:-1] < dist) & (shell_edges[1:] >= dist))
        if index[0].size >0:
            shells_mass[index[0][0]] += mass
    
    core.density = (3*core.mass).in_(units.kg) /(4*numpy.pi*core.radius**3).in_(units.RSun**3)               
    density_profile = (3*shells_mass)/(4*numpy.pi*shells_dr)
    density_profile = numpy.insert(density_profile,0,(core.density.value_in(units.kg / units.RSun**3)))

    # position of the core particle
    shell_cents = numpy.insert(shell_cents,0,0.0)
    return (shell_cents | units.RSun), (density_profile | units.kg / units.RSun**3).in_(units.g / units.cm**3) #.value_in(units.RSun)
    

def tertiary_profiles(gas,core,not_array=False):
    '''
    combines the gas particles with the core particle and moves the system to the center of mass, i.e. center
    of the star. Then seperates the star in `mesa_num_of_zones` spherical shells to calculate the different
    profiles
    '''
    third_star = gas.copy()
    third_star.add_particle(core)
    third_star.move_to_center()
    third_star.dist_from_cm = third_star.position.lengths().in_(units.RSun)
    
    #max_rad = (71.8 | units.RSun).value_in(units.cm)
    max_rad = third_star.position.lengths().max().value_in(units.RSun)
    # shells of 0.65 RSun width
    mesa_num_of_zones = int(max_rad//0.65)
    
    if not_array == True:
        shell_edges = numpy.linspace(core.radius.value_in(units.RSun), max_rad, mesa_num_of_zones+1)
    else:
        shell_edges = numpy.linspace(core.radius.value_in(units.RSun)[0], max_rad, mesa_num_of_zones+1)
    shell_cents = numpy.convolve(shell_edges,numpy.array([0.5, 0.5]), mode= 'valid')
    
    # return the gas and the core
    return den_profile(third_star[:-1],third_star[-1],mesa_num_of_zones,shell_edges,shell_cents)

def plot_giant(gas, core, relax_time,where_to_save,e):
    plt.figure(figsize=(5,5))
    plt.scatter(gas.x.value_in(units.RSun) ,gas.y.value_in(units.RSun))
    #plt.scatter(binary_particles.x.value_in(units.RSun) ,binary_particles.y.value_in(units.RSun), c='black')
    plt.scatter(core.x.value_in(units.RSun) ,core.y.value_in(units.RSun),c='red')
    plt.xlabel(r'R $(R_{\odot})$')
    plt.ylabel(r'R $(R_{\odot})$')
    #plt.xlim(-1,1)
    #plt.ylim(-1,1)
    if relax_time == 0.0:
        plt.savefig(where_to_save+'/Giant_before_relaxation_{:.2f}MSun.png'.format(core.mass.value_in(units.MSun)))
    else:
        plt.savefig(where_to_save+'/giant_relaxed{:.0f}tdyn_{:.2f}MSun_{:.4f}eps.png'.format(relax_time, \
                    core.mass.value_in(units.MSun),e))
    plt.close()

def enc_mass(enc_mass_prof, rad_prof,perce):
    ind = numpy.where(enc_mass_prof>=perce)[0][0]
    return rad_prof[ind].value_in(units.RSun)
    
def plt_dens_profiles(se_rad, se_dens, hyd_rad, hyd_dens ,star_enc_mass, relax_time,m_core,where_to_save,e):
    fig, ax1 = plt.subplots(dpi=200)
    ax1.plot(se_rad.value_in(units.RSun), se_dens.value_in(units.g / units.cm**3), label='MESA') 
    ax1.scatter(hyd_rad.value_in(units.RSun), hyd_dens.value_in(units.g / units.cm**3), marker='.', label='Gadget2', c='blue')
    X = enc_mass(star_enc_mass, se_rad, 0.999)
    ax1.axvspan(0, X, alpha=0.2, color='blue', label = r'$M_{encl}=99.9\% M_{\star}$')
    ax1.scatter(hyd_rad[0].value_in(units.RSun),hyd_dens[0].value_in(units.g / units.cm**3), color='red', label='Mc = {:.2f} M_sun'.format(m_core))
    ax1.set_xlabel(r'$R \; (R_{\odot})$')
    ax1.set_ylabel(r'$\rho \; (g/cm^3)$')
    plt.title("Relaxation Time {:.0f} dynamical timescales".format(relax_time))
    ax1.legend()
    plt.yscale('log')
    if relax_time == 0.0:
        plt.savefig(where_to_save+'/Giant_dens_prof_before_relaxation_{:.2f}MSun.png'.format(m_core))
    else:
        plt.savefig(where_to_save+'/giant_dens_prof_relaxed{:.0f}tdyn_{:.2f}MSun_{:.4f}eps.png'.format(relax_time, \
                         m_core,e))
    plt.close()

    


##############
#### Main ####
##############

'''
The parameter that needs to be optimized for a good relaxation of the stars is the smoothing lenght of the target core. The optimum smoothing length
is directly related to the target core mass and the stepness of the density profile that we try to resolve (see the produced graphs for better understanding).
The stepness of the density porfile is again related with the initial mass of the star, but most importantly with the internal structure of the star the moment
we jump from the stellar evolution code to the sph code. More specifically, convective envelopes are much more homogenius than radiative envelopes and thus easier
to resolve with a smaller number of particles. Furthermore, the number of particles effects slightly the optimum smoothing length, thus it is preferable to start 
testing with a small number of particles just enough to resolve the majority of the mass. 

It seems that for convective envelopes or in other words when we try to resolve <=6 orders of magnitude in densisty (between the density of the core and the
minimum density at the edge of the envelope) a rule of thumb for good relaxation results is smoothing length = [0.1-0.25] * radius of the star.

'''


number_of_sph_particles = 5000

triple, view_on_giant = set_up_initial_conditions()

# Stop stellar evolution when giant's radius is (radius_factor * Roche lobe radius)
radius_factor= 8.0
stop_radius = radius_factor * estimate_roche_radius(triple, view_on_giant)

print("Tertiary's Roche lobe radius is {:.2f} RSun or {:.2f} AU".format(stop_radius.value_in(units.RSun), \
                                                                     stop_radius.value_in(units.au)))

triple_set_up_info(triple, view_on_giant)

# Create directories

path = os.getcwd()

if os.path.isdir("core_masses") == False:
    #create directory to store the output
    os.mkdir(path+'/core_masses')

mult_factor = 10.0

# target core : m_factor * M_{star}
#target_core_m_factor = numpy.arange(0.5,0.8,0.1)
target_core_m_factor = numpy.array(list([0.5,0.6]))

epsilon_m_factor = numpy.arange(0.12, 0.2, 0.01)    
#epsilon_m_factor = numpy.array(list([0.35,0.37,0.4,0.45,0.5]))


for dummy_var, core_mult_f in enumerate(target_core_m_factor):
    
    se_code = MESA(version='2208')
    
    # evolve stars with mesa until R_star = radius_factor R_RLOF
    se_stars, se_code_instance = evolve_stars(triple, view_on_giant, se_code, radius_factor)
    view_on_se_giant = view_on_giant.as_set().get_intersecting_subset_in(se_stars)[0]
    
    # create a giant model with target core mass and N sph particles representing the envelope based
    # on the MESA profiles
    target_core = core_mult_f * view_on_giant.mass.value_in(units.MSun) 
    giant_model = convert_giant_to_sph(view_on_se_giant, number_of_sph_particles,target_core)
    
    dir_name = 'M_{:.1f}Mcore_{:.1e}SPH_part'.format(target_core,number_of_sph_particles)

    if os.path.isdir(path+'/core_masses/'+dir_name) == False:
        #create directory to store the output
        os.mkdir(path+'/core_masses/'+dir_name)

    path_to_dir_for_data = path+'/core_masses/'+dir_name

    ## Giant's profile at ROLF
    giant_num_zones, giant_dens_profile, giant_temp_profile, giant_rad_profile, \
    giant_mass_profile\
    =amuse_mesa_profiles(se_stars[0])

    interpolator = EnclosedMassInterpolator()
    interpolator.initialize(giant_rad_profile, giant_dens_profile)
    enc_mass_calc =(interpolator.enclosed_mass.value_in(units.MSun)/se_stars[0].mass.value_in(units.MSun))[1:]
        
    #save the profile before relaxation
    gas_before_relax = giant_model.gas_particles.copy()
    core_before_relax = giant_model.core_particle.copy()
    radii0, densities0 = tertiary_profiles(giant_model.gas_particles,giant_model.core_particle, True)
        
    plot_giant(gas_before_relax, core_before_relax, 0.0, path_to_dir_for_data,None)
    plt_dens_profiles(giant_rad_profile, giant_dens_profile, radii0, densities0, enc_mass_calc, 0.0, target_core, 
                      path_to_dir_for_data,None)
      
    
    for eps in epsilon_m_factor:
        sph_code = Gadget2
        
        print("Relaxing giant with {:}".format(sph_code.__name__) + ' epsilon = {:.2f}'.format(eps))
        relax_in_isolation(giant_model, sph_code, enc_mass_calc, mult_factor,eps,path_to_dir_for_data)
        giant_model = convert_giant_to_sph(view_on_se_giant, number_of_sph_particles,target_core)
        
    se_code_instance.stop()
    
    
