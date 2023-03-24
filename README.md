# Title to be

[Monitoring Project](https://docs.google.com/spreadsheets/d/1cWGcrytDQaClFpIWc6oRl2DwaFXRPTdyPRCGte0bnr8/edit?usp=sharing)

## Introduction

This repository will contain documents, code, and analysis from my master thesis.


## Abstract

Abstract to be

# Modeling a star as a collection of SPH particles representing the envelope and a Dark Matter particle representing the interior

## Relaxation

Smoothed Particle Hydrodynamics (SPH) simulations are commonly used to model fluids as a collection of particles. However, the discrete nature of the particles can lead to numerical noise and instability in the simulation, particularly in regions of high gradients.

To reduce this noise and ensure the conservation of physical quantities, a technique known as relaxation is employed. Relaxation is a process that involves allowing a system to evolve over time towards its equilibrium state through the dissipation of energy. Relaxation involves iteratively adjusting the positions and velocities of the particles based on their interactions with neighboring particles until the simulation reaches a state of equilibrium.

The goal of relaxation is to reduce numerical noise to an acceptable level, allowing the physical behavior of the system to be accurately simulated. This is achieved by conserving physical quantities such as mass, momentum, and energy.

Overall, relaxation is an essential technique in SPH simulations to ensure that the numerical noise does not overwhelm the physical behavior of the system being simulated.

## Giant DM core particle

In AMUSE, the smoothed particle hydrodynamics (SPH) particles model the gas or fluid in a simulation. These particles interact through a kernel function, which calculates the smoothed density and other smoothed quantities around each particle. The interactions between SPH particles include hydrodynamic forces such as pressure, viscosity, and gravity. 

On the other hand, dark matter (DM) particles are used to model the dark matter component of a simulation. These particles are not subject to hydrodynamic interactions since they are assumed to be collisionless. Instead, they only interact gravitationally with other particles, including both DM particles and SPH particles. This means that the DM particles can influence the dynamics of the SPH particles and the stars, but not vice versa. 

When replacing the core of the Roche lobe filling star with a single DM particle, adjustments are made to the density and internal energy profiles to account for the softening of the core particle. The softened region's density and internal energy are adapted in a manner that preserves pressure equilibrium while conserving the original entropy profile. The softening length of the core particle is then solved for, and its value depends on the mass of the core particle, Mcore.
