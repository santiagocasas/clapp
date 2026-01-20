.. _wrapper-set:

Wrapper set function
====================

.. function:: set(input)

  Tell classy to use certain input parameters, described in the python dictionary ``input``.
  Can also be passed as explicit keywords

  :param input: Input parameters
  :type input: dict


General parameters
^^^^^^^^^^^^^^^^^^

* ``output``

  Here you can control the output of class that you'll be able to access after it computes. In the future the code might be smart enough to recompute always based on what you are actually using, but for now you explicitly have to tell it ahead of time. That means, if you want to use the :func:`get_pk_all` function, for example, you have to tell it ahead of time that you'll want the matter power spectrum (mPk) output. This is only required for the output spectra (angular or matter power spectra, spectral distortions), not for the output of intermediate modules (e.g. background/thermodynamics/perturbations/...).
  
  Examples of commonly needed output options:
    * ``'mPk'`` for :func:`get_pk_all` or :func:`sigma8` or anything else related to :math:`P(k,z)`
    * ``'tCl, pCl, lCl'`` for the :func:`lensed_cl` (coupled with ``{'lensing':'yes'}``), which computes the :math:`C_\ell` angular power spectra
    * ``'mTk, vTk'`` for the transfer functions, :func:`get_transfer`, e.g. for N-body initial conditions. These are the (gauge-transformed) perturbations **at a fixed time**

  If you want the perturbations **at a fixed wavenumber** (for multiple times), see instead ``'k_output_values'``.

  For all the possible options, see :ref:`explanatory`.

* ``lensing``

  There is the CMB lensing power spectrum (the 'lCl' above, corresponding to :math:`C_\ell^{\phi\phi}`) and there is the effect of gravitational lensing on the primary CMB (e.g. the lensed TT, :math:`C_\ell^{TT,\mathrm{lensed}}`. Into most CMB likelihoods you plug the lensed power spectra, not the unlensed ones. If you want to compute the effect of CMB lensing on the primary power spectra (such as TT, TE, EE, etc.), you can activate it by setting ``{'lensing':'yes'}`` (default: 'no')

* ``non_linear``

  Do we want to compute non-linear corrections to the late-time clustering? This is typically done with the halofit or HMcode approaches, or with perturbation theory. In any case, this will allow you to ask for the non-linear power spectrum (e.g. with ``.get_pk_all(k,z,nonlinear=True)``, see :ref:`get_pk_all`) but it will **also affect the primary lensed CMB** by changing the strength of the late-time lensing due to non-linear enhancement of the gravitational potentials (which lens the primary CMB). This effect is important for Planck data already, and will become increasingly important for future CMB analyses, so don't forget about this inconspicuous parameter if you're looking at the CMB.

* ``modes``

  Here you can control if you are interested only in scalar perturbations (default, 's') or in scalar and tensor perturbations ('s,t'). The latter is especially useful if you are doing studies of primordial B modes, following complex multi-field inflation, or want to check the effects of a large tensor-to-scalar ratio 'r'

* ``ic``

  Usually pople are most interested in the usual adiabatic initial conditions (default, 'ad'), but you might also want to investigate isocurvature modes,, choosing the corresponding 'bi' or 'cdi' for baryon and cold-dark-matter isocurvature, respectively. There are many related options for the primoridal power spectra for the isocurvature modes, so check out :ref:`explanatory`.

* ``l_max_scalars``
  The maximum multipole :math:`\ell` for which you want to have the CMB power spectra computed

* ``P_k_max_1/Mpc``
  The maximum wavenumber :math:`k` in [1/Mpc] for which you want to get the matter power spectrum from :ref:`get_pk_all`.

* ``z_max_pk``
  The maximum redshift :math:`z` to which you want to be able to get the matter power spectrum from :ref:`get_pk_all`. Usually it's better to overestimate rather than to be exact.

* ``XXX_verbose``
  If you feel like CLASS is doing something wrong, try to let it tell you what it's doing! You can increase the verbosity of any of CLASS's modules by setting the ``XXX_verbose`` flag for them to a number between 0 (nothing), 1 (basic info), 2 (advanced info) all the way to 5 (debug info). Simply replace the ``XXX`` with the name of the corresponding module, i.e. `ìnput``, ``background``, ``thermodynamics``, ``perturbations``, ``transfer``, ``primordial``, ``harmonic``, ``fourier``, ``lensing``, ``distortions ``and/or ``output``.

Model parameters
^^^^^^^^^^^^^^^^


* ``h``, ``H0``
  The Hubble parameter in [km/s/Mpc] ('H0') or in units of [100km/s/Mpc] ('h').

* ``Omega_b``, ``omega_b``
  The baryon mean density. Note that the capitalization is a common point of confusion: Use ``Omega_b`` for the fractional abundance, i.e. :math:`\Omega_b = \rho_b/\rho_\mathrm{crit}`, and use ``omega_b`` for the **reduced** fractional abundance (sometimes also called physical abundance), i.e. :math:`\omega_b = \Omega_b h^2 = C_\rho \cdot \rho_b`, where the constant :math:`C_\rho = h^2/\rho_\mathrm{crit} = 8\pi G h^2/(3 H_0^2) \approx 1.622 \cdot 10^{22} (\mathrm{kg}/\mathrm{m}^3)^{-1}`. Luckily you don't need to implement this constant yourself, but you can simply use the :ref:`density_factor` property -- The SI density of baryons is simply :math:`\rho_b` [SI] = ``omega_b * density_factor / (100*c)**2`` where ``c = 299792.458`` is the speed of light in [km/s]. So while ``Omega_b`` with capital ``O`` is the fractional abundance (a percentage), the ``omega_b`` with small ``o`` is the reduced fractional abundance, directly proportional to :math:`\rho_b` up to a numerical factor.

* ``Omega_cdm``, ``omega_cdm``, ``Omega_m``, ``omega_m``
  The cold dark matter mean density (CDM) and the total non-relativistic mean density (m). These are defined as ``Omega_b`` above, just for CDM or m instead of baryons. Note that this also means that e.g. ``Omega_m`` is the fractional abundance of non-relativistic matter, and ``omega_m`` is the reduced fractional abundance. You can either specify the amount of cold dark matter (CDM), or specify the total amount of non-relativistic matter (m). In the latter case we compute the amount of cold dark matter by substracting from the total requested non-relativistic matter the other contributions like baryons (b) and massive neutrinos (ncdm).

* ``Neff``, ``N_ur``
  'Neff' is simply the typical effective numbers of light degrees of freedom, which would be 3 with instantaneous neutrino decoupling, and 3.044 with non-instantaneous neutrino decoupling. Various dark radiation models can increase (or even decrease) 'Neff', though. It is defined as :math:`N_\mathrm{eff} = \frac{\rho_\mathrm{rad}-\rho_\gamma}{\rho_{1\nu}} = \frac{7}{8} \left(\frac{4}{11}\right)^{4/3} \frac{\rho_x}{\rho_\gamma}` where :math:`\rho_\gamma` the photon energy density, :math:`\rho_\mathrm{rad}` the total radiation energy density, :math:`\rho_x = \rho_\mathrm{rad}-\rho_\gamma` is the total density of all non-photon contributions to radiation in the early universe, and :math:`\rho_\mathrm{1\nu}` is the energy density of a single neutrino species in the instantaneous decoupling approximation. You can also work with the even more cumbersome 'N_ur', which will require you to input the precise amount of free-streaming radiation that is not contained in massive neutrinos, but for that you'll have to compute the contributions from massive neutrinos, see also :ref:`explanatory`. We recommend saving yourself the trouble and simply using 'Neff'. 

 .. note::
    Don't confuse ``Neff`` (good, new parameter) with ``N_eff`` (bad old parameter with an underscore). The latter is deprecated and is a synonym for ``N_ur`` -- you should never use it.
