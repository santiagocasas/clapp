Background functions
--------------------

Cosmic distances are defined as usual, i.e.

.. math::
   \begin{aligned}
   D_c(z) &= \int_0^z \frac{\mathrm{d}x}{H(x)} \\
   D_M(z) &= \left\{\begin{array}{lr} D_c(z), & \text{for } K=0\\
        \sin(\sqrt{K}D_c(z))/\sqrt{K}, & \text{for } K>0\\
        \sinh(\sqrt{-K}D_c(z))/\sqrt{-K}, & \text{for } K<0
        \end{array}\right. \\
   D_A(z) &= D_M(z) / (1+z) \\
   D_L(z) &= D_M(z) \cdot (1+z)
   \end{aligned}

We also compute the angular diameter distance between two redshifts. In that case we use the common replacement :math:`D_c(z) \to D_c(z_2)-D_c(z_1)` for the computation of :math:`D_M(z)` and let :math:`z_2` be the overall rescaling redshift of :math:`D_A(z_1, z_2)`.

 .. function:: get_background()

  Get entire background dictionary available in CLASS, containing a dictionary of all background quantities that CLASS saved



 .. function:: Hubble(z)
  
  Get Hubble parameter H(z) for the redshift z

  :param z: Redshift (value or list)

 .. function:: comoving_distance(z)
  
  Get conformal/comoving distance D_M(z) for the redshift z

  :param z: Redshift (value or list)

 .. function:: luminosity_distance(z)
  
  Get luminosity distance D_L(z) for the redshift z

  :param z: Redshift (value or list)

 .. function:: angular_distance(z)
  
  Get luminosity distance D_A(z) for the redshift z

  :param z: Redshift (value or list)

 .. function:: angular_distance_from_to(z1,t2)
  
  Get luminosity distance D_A(z1,z2) for the redshift pair (z1,z2)

  :param z1: Redshift (value only)
  :param z2: Redshift (value only)

Less used functions
^^^^^^^^^^^^^^^^^^^

 .. function:: Om_m(z)
  
  Get matter density fraction :math:`\Omega_m(z) = \rho_m(z)/\rho_\mathrm{crit}(z)` for the redshift z

  :param z: Redshift (value or list)

 .. function:: Om_b(z)
  
  Get baryon density fraction :math:`\Omega_b(z) = \rho_b(z)/\rho_\mathrm{crit}(z)` for the redshift z

  :param z: Redshift (value or list)

 .. function:: Om_cdm(z)
  
  Get cold dark matter density fraction :math:`\Omega_\mathrm{cdm}(z) = \rho_\mathrm{cdm}(z)/\rho_\mathrm{crit}(z)` for the redshift z

  :param z: Redshift (value or list)

 .. function:: Om_ncdm(z)
  
  Get non-cold dark matter density fraction (typically massive neutrinos) :math:`\Omega_\mathrm{ncdm}(z) = \rho_\mathrm{ncdm}(z)/\rho_\mathrm{crit}(z)` for the redshift z

  :param z: Redshift (value or list)

 .. function:: z_of_r(z)
  
  Get conformal/comoving distance D_M(z) and Hubble parameter H(z) for the redshift z

  :param z: Redshift (value or list)

 .. function:: z_of_tau(tau)
  
  Get redshift corresponding to a given conformal time

  :param tau: Conformal time
