Wrapper functions
==================

Main usage
----------

.. code:: python

  import classy
  cosmo = classy.Class()
  # Now work with the cosmo object!
  cosmo.set({'omega_b':0.022,'H0':71, 'output':'tCl'})
  cosmo.raw_cl()

.. function:: set(input)

  Tell classy to use certain input parameters, described in the python dictionary ``input``.
  Can also be passed as explicit keywords

  For more info, look at :ref:`wrapper-set`.

  :param input: Input parameters
  :type input: dict

.. note::

  ``cosmo.set({'omega_b':0.02})``
  is equivalent to
  ``cosmo.set(omega_b=0.02)``

.. caution::

  It is impossible to do the same for
  ``cosmo.set({'100*theta_s':1.04})``
  In this case, we would use instead the equivalent parameter name
  ``cosmo.set({'theta_s_100':1.04})``
  and convert it into
  ``cosmo.set(theta_s_100=1.04)``

.. function:: set_baseline(baseline_name):

  Set input parameters to one of the available baseline cosmologies.

  * | ``planck2018_lensing_bao`` or ``p18lb``:
    | Best-fitting cosmology to Planck 2018 + lensing + BAO (SDSS)

  * | ``planck2018_lensing`` or ``p18l``:
    | Best-fitting cosmology to Planck 2018 + lensing

  * | ``planck2018`` or ``p18``:
    | Best-fitting cosmology to Planck 2018

Conversion factors
------------------

Class works internally in units of Mpc^n, therefore we want to use conversion factors to SI units. These are available as properties of each cosmo object

* ``density_factor``
  Converting from class density units to kg/m^3

* ``kg_to_eV``
  Converting from kg to eV (useful also to convrt kg/m^3 to eV/m^3, see example)

* ``Mpc_to_m``
  Converting from Mpc to m

* ``kg_to_Msol``
  Converting from kg to solar mass ($2 \times 10^{30}$kg)

* ``kgm3_to_MsolMpc3``
  Converting from kg/m^3 to Msol/Mpc^3

.. admonition:: Example

  If we want the CDM density in SI units or in astrophysical units, we can do it like that:

  .. code:: python

    bg = cosmo.get_background()
    rho_cdm = bg['(.)rho_cdm']
    rho_cdm_SI = rho_cdm * cosmo.density_factor
    rho_cdm_astro = rho_cdm_SI * cosmo.kgm3_to_MsolMpc3
    print(f"The current CDM density is {rho_cdm_SI[-1]*1e27:.3f} * 10^(-27) kg/m^3")
    print(f"The current CDM density is {rho_cdm_SI[-1]*(cosmo.Mpc_to_m/1e6)**3*1e-22:.3f} * 10^22 kg/pc^3")
    print(f"The current CDM density is {rho_cdm_astro[-1]*1e-10:.3f} * 10^10 Msol/Mpc^3")
  :rtype: array of numpy arrays

.. Test
  comment:: .. attention:: -- for attention blocks
  comment:: caution, hint, tip, advice, warning, seealso, note
  comment:: .. admonition:: Example -- for examples
  comment:: .. code:: for code block
