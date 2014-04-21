Introduction
============
DeerQMC is an implementation of the Determinantal Quantum Monte Carlo
simulation to study the one- and two-dimensional Hubbard models. Its main
feature is that it implements an anisotropic transformation of the
electron-electron interaction on every lattice site, which can be chosen
freely (cf. [1] of the revelant papers).

This is work in progress
------------------------
DeerQMC is currently under heavy development and therefore by no means stable.
At the moment, it is mainly concerned with generating a Markov-Chain of lattice
configurations.

The `TODO` contains some information on the outstanding fixes and possible
extensions.

Documentation & citation
------------------------
A full documentation on how to use this software is in preparation and will
be made available once it has (again) reached a sufficiently stable state. In the
meantime, an introductory review of the DQMC method (as well as an extended list
of the relevant literature) can be found in my Master thesis available at:
http://kth.diva-portal.org/smash/record.jsf?searchId=1&pid=diva2:708672


If you obtained your numerical results using this software, I would kindly ask
you to send me an email with a reference to your work, and to cite this
software as:
```
R. J. Beckert, DeerQMC (2014), GitHub repository, https://github.com/SuperFluffy/DeerQMC
```


Dependencies
============
+ `Python >= 3.3`
+ `Numpy >= 1.8`
+ `Scipy >= 0.13`
+ `PyYAML >= 3.10`
+ `h5py >= 2.2.1`


Relevant papers
===============
1. E. Langmann, 2013, Unpublished Notes
2. http://dx.doi.org/10.1103/PhysRevD.24.2278
3. http://dx.doi.org/10.1103/PhysRevB.28.4059 
4. http://dx.doi.org/10.1103/PhysRevB.31.4403
5. “Stabilization of Simulations of Many-Fermion Systems” (pp. 156--167) in Proceedings of the Los Alamos Conference on Quantum Simulation (1990)

Description
-----------
1. The proposal for a generalized discrete Hubbard-Stratonovich transformation and
motivation for this implementation.
2. The initial proposal by Blanckenbecler, Scalapino, and Sugar for carrying out
Monte Carlo calculations of field theories with Fermionic degrees of freedom by
integrating these out.
3. Hirsch's discrete Hubbard-Stratonovich transformation to replace the on-site
electron-electron interaction by a coupling to Bosonic (Ising) fields.
4. The original paper by Hirsch introducing the algorithm to simulate the
two-dimensional Hubbard model.
5. Necessary stabilization methods for calculating the Green's functions
occuring in the simulation.
