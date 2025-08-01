Spectra functions
-----------------

.. function:: lensed_cl()

  | Lensed CMB power spectra (to be used for cosmological inference)
  | Can return temperature, polarization, lensing, depending on the ``'output'`` settings
  | For ``output`` including ``tCl`` has the temperature autocorrelation (TT)
  | For ``output`` including ``pCl`` has the polarization auto/cross-correlations (EE, BB, EB)
  | For ``output`` including ``lCl`` has the lensing auto-correlations (PP)
  | If multiple options are present, also their cross-correlations are included, e.g. with ``tCl, pCl`` we also have TE
  | The option ``lensing`` needs to be set to ``yes`` for this function to work correctly

  :param lmax: Define the maximum l for which the C_l will be returned
               (inclusively) -- by default this will just be the maximal l that is computed (which is given by the input parameter ``l_max_scalars``).
               This number will be checked against the maximum l
               at which they were actually computed by CLASS, and an error will
               be raised if the desired lmax is bigger than what CLASS can give.
  :type lmax: int, optional

  :param nofail: Check and enforce the computation of the C_l's to the given lmax.
  :type nofail: bool, optional

.. function:: raw_cl()
  
  | Raw un-lensed CMB power spectra
  | See documentation for :func:`lensed_cl` for more info.
  | The only differences are that ``lensing=yes`` is not required for this case

.. function:: density_cl()
  
  | Number count/Shear angular power spectra
  | See documentation for :func:`lensed_cl` for more info.
  | The only differences are that ``lensing=yes`` is not required for this case

  :return: Array that contains the list (in this order) of self correlation of
           1st bin, then successive correlations (set by non_diagonal) to the
           following bins, then self correlation of 2nd bin, etc. The array
           starts at index_ct_dd.


Power spectrum functions
^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: get_pk_all(k, z, nonlinear = True, cdmbar = False, z_axis_in_k_arr = 0, interpolation_kind='cubic')

  | Optimized function to get the power spectrum for wavenumber(s) k in 1/Mpc and redshift(s) z
  | Supports various options, such as the total nonlinear power spectrum or just the linear power spectrum, using the ``nonlinear`` flag
  | Can also return just CDM+baryons with the ``cdmbar`` flag
  | Supports inputs of single wavenumbers and redshifts, or lists thereof
  | The general rule is that as long as the k and z arrays are broadcastable (in the numpy sense) the function should work

  :param k: List or value for the wavenumber in 1/Mpc
  :param z: List or value of the redshift(s)

  :param nonlinear: Whether to return the non-linear power spectrum (if computed)
  :type nonlinear: bool, optional
  :param cdmbar: Whether to return only the CDM+baryon power spectrum or the total matter power spectrum (only different when neutrinos are present)
  :type cdmbar: bool, optional
  :param z_axis_in_k_arr: If ``k`` is a multi-dimensional array, this integer tells which of the axes in the ``k`` array correspond to redshift variations. Typically the first axis (0).
  :type nonlinear: int, optional
  :param interpolation_kind: How to interpolate from CLASS-computed k-values to user-provided k-values? Typically done via cubic splines ('cubic')
  :type interpolation_kind: str, optional

.. hint::
  If ``k`` is a multi-dimensional array of equal shapes, we assume it is a function of k=k_obs(z, ...), like e.g. k=k_obs(z, k, mu). This order can be changed using ``z_axis_in_k_arr``. 
  If ``k`` is a multi-dimensional list of arrays, each with different lengths, we assume/hope that each index of this list corresponds to a single redshift
  If ``k`` is a single-dimensional array, assume that we should evaluate P(k,z) with the same ``k`` array for each z
  Finally, if ``k`` is a single value, assume that we should evalute P(k,z) with that value for each z.

.. function:: get_pk_and_k_and_z(nonlinear=True, only_clustering_species = False, h_units=False)

  | Returns the power spectrum AND the k and z arrays used internally in CLASS, so that you can build your own interpolator on top of it
  | Supports various options, such as the total nonlinear power spectrum or just the linear power spectrum, using the ``nonlinear`` flag
  | Can also return just CDM+baryons with the ``only_clustering_species`` flag
  | The output wavenumbers can be converted into h/Mpc with the ``h_units`` flag 

  :return:
      pk : grid of power spectrum values, pk[index_k,index_z]
      k : vector of k values, k[index_k] (in units of 1/Mpc by default, or h/Mpc when setting h_units to True)
      z : vector of z values, z[index_z]

.. function:: get_transfer_and_k_and_z(output_format='class', h_units=False)
  
  Returns the transfer functions AND the k and z arrays used internally in CLASS, so that you can build your own interpolator on top of it

  :param output_format: What format? 'class' or 'camb'
  :type output_format: str, optional

  :param h_units: Should the returned wavenumbers be converted to h/Mpc instead of 1/Mpc?
  :type h_units: bool, optional

  :return:
      tk : dictionary containing all transfer functions.
           For instance, the grid of values of 'd_c' (= delta_cdm) is available in tk['d_c']
           All these grids have indices [index_k,index,z], for instance tk['d_c'][index_k,index,z]
      k : vector of k values (in units of 1/Mpc by default, or h/Mpc when setting h_units to True)
      z : vector of z values

.. hint::

   The class format is the good old `Ma & Bertschinger definitions`_, and includes all possible perturbations

   The CAMB format includes a factor :math:`-1/k^2` and only the :math:`\delta_\mathrm{cdm}`, :math:`\delta_b`, :math:`\delta_g`, :math:`\delta_\mathrm{ur}`, :math:`\delta_\mathrm{ncdm}` (massive neutrinos), and :math:`\delta_\mathrm{tot}` (total), as often used in N-body initial condition codes

.. _Ma & Bertschinger definitions: https://arxiv.org/abs/astro-ph/9506072

.. function:: get_Weyl_pk_and_k_and_z(nonlinear=True, h_units=False)

  | Returns the Weyl power spectrum AND the k and z arrays used internally in CLASS, so that you can build your own interpolator on top of it
  | See :func:`get_pk_and_k_and_z` for more info on the input/output parameters
  | Note that this function just calls get_pk_and_k_and_z and corrects the output by the ratio of transfer functions [(phi+psi)/d_m]^2.

.. function:: get_primordial()

  Get full dictionary of the primordial power spectrum (k-vector in h/Mpc and primordial power spectrum). The input parameter 'output' must be set to something, e.g. 'tCl'
  Will contain the scalar and/or tensor spectrum depending on the 'modes' input

.. function:: get_perturbations(return_copy=True)

  Get full dictionary of the perturbations for the given k-values set by the 'k_output_values' input parameter. Set 'output' to something, e.g. 'tCl'
  'return_copy=False' will give you access directly to the underlying C pointers in CLASS, so be careful that these will become invalid once CLASS is deallocated.

  :return: get_perturbations()['scalar'] is an array of length 'k_output_values' of dictionares containing scalar perturbations. Similar for get_perturbations()['vector'] and get_perturbations()['tensor'].

.. function:: get_transfer(z=0, output_format='class')

  Get full dictionary of the transfer functions for given redshift 'z'. Set 'output' to at least 'mTk' and/or 'vTk'. To input z>0, you need to set z_pk > z in the input. See :func:`get_transfer_and_k_and_z` for input parameter descriptions.

.. function:: get_sources()

   Returns a dictionary of all source functions that CLASS stores, as well as the k and tau arrays

.. function:: sigma(R,z, h_units = False)

  Returns :math:`\sigma(R,z)` for the total matter power spectrum

  :param R: Array or value of the smoothing radius R in Mpc (or Mpc/h if h_units=True)
  :param z: Array or value of the redshift z

.. function:: sigma_cb(R, z, h_units = False)

  Same as :func:`sigma(R,z)` but for the CDM+baryons power spectrum only





Additional useful functions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

 .. function:: sigma8()
  
  Get sigma(8,z,hunits=True), otherwise known as :math:`\sigma_8`

 .. function:: S8()
  
  Get sigma8()*sqrt(Omega_m()/0.3), otherwise known as :math:`S_8`

 .. function:: sigma8_cb()
  
  Get sigma_cb(8,z,hunits=True), the equivalent of :math:`\sigma_8` for CDM+baryons only

.. function:: pk_tilt(k,z)

  (Numerical) derivative of the power spectrum at a given wavenumber and redshift

  :param k: Wavenumber in 1/Mpc
  :param z: Redshift z

 .. function:: scale_independent_growth_factor(z)
  
  Return the scale invariant growth factor :math:`D(a)` for CDM perturbations, which is defined as the solution to the scale-independent growth differential equation :math:`D''(a) + aH D'(a) = \frac{3}{2} a^2 \rho_m(a) D(a)`

  :param z: Redshift z

 .. function:: scale_independent_growth_factor_f(z)
  
  Return the scale invariant growth factor derivative :math:`f(a) = \mathrm{d} \ln D/\mathrm{d} \ln a` for CDM perturbations.

  :param z: Redshift z

 .. function:: scale_independent_growth_factor_f_sigma8(z)
  
  Return the scale invariant growth factor derivative :math:`f\sigma_8(a) = f(a) \sigma_8(z)` with :math:`\sigma_8(z)` determined as :func:`sigma`(8,z,h_units=True)

  :param z: Redshift z

 .. function:: scale_dependent_growth_factor_f(k, z, h_units=False, nonlinear=False, Nz=20)
  
  Return the true derivative of the power spectrum :math:`f(a) = \frac{1}{2} \mathrm{d} \ln P(k,a)/\mathrm{d} \ln a` for CDM perturbations. With the settings nonlinear you can select the linear or non-linear power spectrum for the computation. Nz is the number of redshifts used to determine the numerical derivative, using a UnivariateSpline.

  :param k: Wavenumber in 1/Mpc (except if h_units=True, then h/Mpc)
  :param z: Redshift z

 .. function:: scale_dependent_growth_factor_f(k, z, h_units=False, nonlinear=False, Nz=20)
  
  Same as :math:`scale_dependent_growth_factor_f` for CDM+baryons.

 .. function:: scale_dependent_growth_factor_D(k, z, h_units=False, nonlinear=False, Nz=20)
  
  Returns simply :math:`\sqrt{P(k,z)/P(k,0)}`, see :math:`scale_dependent_growth_factor_f` for the input parameters

 .. function:: scale_dependent_growth_factor_D_cb(k, z, h_units=False, nonlinear=False, Nz=20)
  
  Same as :math:`scale_dependent_growth_factor_D` for CDM+baryons.

 .. function:: effective_f_sigma8(z, z_step=0.1)
  
  Return the scale invariant growth factor derivative :math:`f\sigma^\mathrm{eff}_8(a) = \mathrm{d}\sigma_8(z)/\mathrm{d}\ln a` with :math:`\sigma_8(z)` determined as :func:`sigma`(8,z,h_units=True).
  Uses a double-sided forward derivative. For z < z_step the step is reduced progressively down to z_step/10 while sticking to a double-sided derivative. For z< z_step/10 a single-sided derivative is used instead.

  :param z: Redshift z
  :param z_step: Redshift step size \Delta z

 .. function:: effective_f_sigma8_spline(z, Nz=20)
  
  Same as :func:`effective_f_sigma8`, but using a spline of Nz evalutations in the range [z-0.1,z+0.1] instead, or [0,0.2] if z<0.1. Uses a CubicSpline for taking the derivative.

 .. function:: nonlinear_scale(z, z_size)
  
  Scale of non-linearity as defined in CLASS (:math:`R_\mathrm{nl}(z)`), i.e. the implicit solution of :math:`\sigma(R_\mathrm{nl}(z),z)=1`. z_size = number of items in z

 .. function:: nonlinear_scale_cb(z, z_size)
  
  Same as :func:`nonlinear_scale`, but for CDM+baryons only








Less used functions
^^^^^^^^^^^^^^^^^^^

.. function:: pk_numerical_nw(k,z)

  Function to get the numerically de-wiggled power spectrum of total matter

  :param k: Single wavenumber in 1/Mpc
  :type k: float

  :param z: Single redshift
  :type z: float

.. function:: pk_analytic_nw(k,z)

  Function to get the analytically de-wiggled power spectrum of total matter

  :param k: Single wavenumber in 1/Mpc
  :type k: float

  :param z: Single redshift
  :type z: float

.. function:: pk(k,z)

  Old/Depracated function to get the (non-linear) power spectrum of total matter, see :func:`get_pk_all` for the recommended functionality

  :param k: Single wavenumber in 1/Mpc
  :type k: float

  :param z: Single redshift
  :type z: float

.. function:: pk_cb(k,z)

  Old/Depracated function to get the (non-linear) power spectrum of CDM+baryons, see :func:`get_pk_all` for the recommended functionality

  :param k: Single wavenumber in 1/Mpc
  :type k: float

  :param z: Single redshift
  :type z: float

.. function:: pk_lin(k,z)

  Old/Depracated function to get the linear power spectrum of total matter, see :func:`get_pk_all` for the recommended functionality

  :param k: Single wavenumber in 1/Mpc
  :type k: float

  :param z: Single redshift
  :type z: float

.. function:: pk_cb_lin(k,z)

  Old/Depracated function to get the linear power spectrum of CDM+baryons, see :func:`get_pk_all` for the recommended functionality

  :param k: Single wavenumber in 1/Mpc
  :type k: float

  :param z: Single redshift
  :type z: float

.. function:: get_pk(k,z, k_size, z_size, mu_size)

   Speciality function to get the (non-linear) power spectrum of total matter for a grid of ``k`` and ``z`` values, where the ``k`` grid is expected to be acessed as ``k[index_k,index_z,index_mu]``, and the ``z`` grid as ``z[index_z]``

  :param k: Input wavenumbers k_obs(k, z, mu)
  :type k: Three-dimensional numpy array

  :param z: Input redshifts
  :type z: One-dimensional numpy array

.. function:: get_pk_cb(k,z, k_size, z_size, mu_size)

  See :func:`get_pk`, but specifically for CDM+baryons only

.. function:: get_pk_lin(k,z, k_size, z_size, mu_size)

  See :func:`get_pk`, but specifically linearly

.. function:: get_pk_cb_lin(k,z, k_size, z_size, mu_size)

  See :func:`get_pk`, but specifically linearly
  

.. function:: get_pk_array(k,z, k_size, z_size, nonlinear)

   Same as :func:`get_pk`, but instad for 1-dimensional k arrays, with 'nonlinear' set to 1 or 0

.. function:: get_pk_cb_array(k,z, k_size, z_size, nonlinear)

   Same as :func:`get_pk_array`, but instead for CDM+baryons
