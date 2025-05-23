## classy-current-docstrings.txt

```
Function: _check_task_dependency(self, level)
Docstring:

        Fill the level list with all the needed modules

        .. warning::

            the ordering of modules is obviously dependent on CLASS module order
            in the main.c file. This has to be updated in case of a change to
            this file.

        Parameters
        ----------

        level : list
            list of strings, containing initially only the last module required.
            For instance, to recover all the modules, the input should be
            ['lensing']

        
---------------------------------
Function: compute(self, level=["distortions"])
Docstring:

        compute(level=["distortions"])

        Main function, execute all the _init methods for all desired modules.
        This is called in MontePython, and this ensures that the Class instance
        of this class contains all the relevant quantities. Then, one can deduce
        Pk, Cl, etc...

        Parameters
        ----------
        level : list
                list of the last module desired. The internal function
                _check_task_dependency will then add to this list all the
                necessary modules to compute in order to initialize this last
                one. The default last module is "lensing".

        .. warning::

            level default value should be left as an array (it was creating
            problem when casting as a set later on, in _check_task_dependency)

        
---------------------------------
Function: density_factor(self)
Docstring:

        The density factor required to convert from the class-units of density to kg/m^3 (SI units)
        
---------------------------------
Function: kgm3_to_eVMpc3(self)
Docstring:

        Convert from kg/m^3 to eV/Mpc^3
        
---------------------------------
Function: kgm3_to_MsolMpc3(self)
Docstring:

        Convert from kg/m^3 to Msol/Mpc^3
        
---------------------------------
Function: raw_cl(self, lmax=-1, nofail=False)
Docstring:

        raw_cl(lmax=-1, nofail=False)

        Return a dictionary of the primary C_l

        Parameters
        ----------
        lmax : int, optional
                Define the maximum l for which the C_l will be returned
                (inclusively). This number will be checked against the maximum l
                at which they were actually computed by CLASS, and an error will
                be raised if the desired lmax is bigger than what CLASS can
                give.
        nofail: bool, optional
                Check and enforce the computation of the harmonic module
                beforehand, with the desired lmax.

        Returns
        -------
        cl : dict
                Dictionary that contains the power spectrum for 'tt', 'te', etc... The
                index associated with each is defined wrt. Class convention, and are non
                important from the python point of view. It also returns now the
                ell array.
        
---------------------------------
Function: lensed_cl(self, lmax=-1,nofail=False)
Docstring:

        lensed_cl(lmax=-1, nofail=False)

        Return a dictionary of the lensed C_l, computed by CLASS, without the
        density C_ls. They must be asked separately with the function aptly
        named density_cl

        Parameters
        ----------
        lmax : int, optional
                Define the maximum l for which the C_l will be returned (inclusively)
        nofail: bool, optional
                Check and enforce the computation of the lensing module beforehand

        Returns
        -------
        cl : dict
                Dictionary that contains the power spectrum for 'tt', 'te', etc... The
                index associated with each is defined wrt. Class convention, and are non
                important from the python point of view.
        
---------------------------------
Function: density_cl(self, lmax=-1, nofail=False)
Docstring:

        density_cl(lmax=-1, nofail=False)

        Return a dictionary of the primary C_l for the matter

        Parameters
        ----------
        lmax : int, optional
            Define the maximum l for which the C_l will be returned (inclusively)
        nofail: bool, optional
            Check and enforce the computation of the lensing module beforehand

        Returns
        -------
        cl : numpy array of numpy.ndarrays
            Array that contains the list (in this order) of self correlation of
            1st bin, then successive correlations (set by non_diagonal) to the
            following bins, then self correlation of 2nd bin, etc. The array
            starts at index_ct_dd.
        
---------------------------------
Function: luminosity_distance(self, z)
Docstring:

        luminosity_distance(z)
        
---------------------------------
Function: pk(self,double k,double z)
Docstring:

        Gives the total matter pk (in Mpc**3) for a given k (in 1/Mpc) and z (will be non linear if requested to Class, linear otherwise)

        .. note::

            there is an additional check that output contains `mPk`,
            because otherwise a segfault will occur

        
---------------------------------
Function: pk_cb(self,double k,double z)
Docstring:

        Gives the cdm+b pk (in Mpc**3) for a given k (in 1/Mpc) and z (will be non linear if requested to Class, linear otherwise)

        .. note::

            there is an additional check that output contains `mPk`,
            because otherwise a segfault will occur

        
---------------------------------
Function: pk_lin(self,double k,double z)
Docstring:

        Gives the linear total matter pk (in Mpc**3) for a given k (in 1/Mpc) and z

        .. note::

            there is an additional check that output contains `mPk`,
            because otherwise a segfault will occur

        
---------------------------------
Function: pk_cb_lin(self,double k,double z)
Docstring:

        Gives the linear cdm+b pk (in Mpc**3) for a given k (in 1/Mpc) and z

        .. note::

            there is an additional check that output contains `mPk`,
            because otherwise a segfault will occur

        
---------------------------------
Function: pk_numerical_nw(self,double k,double z)
Docstring:

        Gives the nowiggle (smoothed) linear total matter pk (in Mpc**3) for a given k (in 1/Mpc) and z

        .. note::

            there is an additional check that `numerical_nowiggle` was set to `yes`,
            because otherwise a segfault will occur

        
---------------------------------
Function: pk_analytic_nw(self,double k)
Docstring:

        Gives the linear total matter pk (in Mpc**3) for a given k (in 1/Mpc) and z

        .. note::

            there is an additional check that `analytic_nowiggle` was set to `yes`,
            because otherwise a segfault will occur

        
---------------------------------
Function: get_pk(self, np.ndarray[DTYPE_t,ndim=3] k, np.ndarray[DTYPE_t,ndim=1] z, int k_size, int z_size, int mu_size)
Docstring:
 Fast function to get the power spectrum on a k and z array 
---------------------------------
Function: get_pk_cb(self, np.ndarray[DTYPE_t,ndim=3] k, np.ndarray[DTYPE_t,ndim=1] z, int k_size, int z_size, int mu_size)
Docstring:
 Fast function to get the power spectrum on a k and z array 
---------------------------------
Function: get_pk_lin(self, np.ndarray[DTYPE_t,ndim=3] k, np.ndarray[DTYPE_t,ndim=1] z, int k_size, int z_size, int mu_size)
Docstring:
 Fast function to get the linear power spectrum on a k and z array 
---------------------------------
Function: get_pk_cb_lin(self, np.ndarray[DTYPE_t,ndim=3] k, np.ndarray[DTYPE_t,ndim=1] z, int k_size, int z_size, int mu_size)
Docstring:
 Fast function to get the linear power spectrum on a k and z array 
---------------------------------
Function: get_pk_all(self, k, z, nonlinear = True, cdmbar = False, z_axis_in_k_arr = 0, interpolation_kind='cubic')
Docstring:
 General function to get the P(k,z) for ARBITRARY shapes of k,z
            Additionally, it includes the functionality of selecting wether to use the non-linear parts or not,
            and wether to use the cdm baryon power spectrum only
            For Multi-Dimensional k-arrays, it assumes that one of the dimensions is the z-axis
            This is handled by the z_axis_in_k_arr integer, as described in the source code 
---------------------------------
Function: get_pk_and_k_and_z(self, nonlinear=True, only_clustering_species = False, h_units=False)
Docstring:

        Returns a grid of matter power spectrum values and the z and k
        at which it has been fully computed. Useful for creating interpolators.

        Parameters
        ----------
        nonlinear : bool
                Whether the returned power spectrum values are linear or non-linear (default)
        only_clustering_species : bool
                Whether the returned power spectrum is for galaxy clustering and excludes massive neutrinos, or always includes everything (default)
        h_units : bool
                Whether the units of k in output are h/Mpc or 1/Mpc (default)

        Returns
        -------
        pk : grid of power spectrum values, pk[index_k,index_z]
        k : vector of k values, k[index_k] (in units of 1/Mpc by default, or h/Mpc when setting h_units to True)
        z : vector of z values, z[index_z]
        
---------------------------------
Function: get_transfer_and_k_and_z(self, output_format='class', h_units=False)
Docstring:

        Returns a dictionary of grids of density and/or velocity transfer function values and the z and k at which it has been fully computed.
        Useful for creating interpolators.
        When setting CLASS input parameters, include at least one of 'dTk' (for density transfer functions) or 'vTk' (for velocity transfer functions).
        Following the default output_format='class', all transfer functions will be normalised to 'curvature R=1' at initial time
        (and not 'curvature R = -1/k^2' like in CAMB).
        You may switch to output_format='camb' for the CAMB definition and normalisation of transfer functions.
        (Then, 'dTk' must be in the input: the CAMB format only outputs density transfer functions).
        When sticking to output_format='class', you also get the newtonian metric fluctuations phi and psi.
        If you set the CLASS input parameter 'extra_metric_transfer_functions' to 'yes',
        you get additional metric fluctuations in the synchronous and N-body gauges.

        Parameters
        ----------
        output_format  : ('class' or 'camb')
                Format transfer functions according to CLASS (default) or CAMB
        h_units : bool
                Whether the units of k in output are h/Mpc or 1/Mpc (default)

        Returns
        -------
        tk : dictionary containing all transfer functions.
                For instance, the grid of values of 'd_c' (= delta_cdm) is available in tk['d_c']
                All these grids have indices [index_k,index,z], for instance tk['d_c'][index_k,index,z]
        k : vector of k values (in units of 1/Mpc by default, or h/Mpc when setting h_units to True)
        z : vector of z values
        
---------------------------------
Function: get_Weyl_pk_and_k_and_z(self, nonlinear=False, h_units=False)
Docstring:

        Returns a grid of Weyl potential (phi+psi) power spectrum values and the z and k
        at which it has been fully computed. Useful for creating interpolators.
        Note that this function just calls get_pk_and_k_and_z and corrects the output
        by the ratio of transfer functions [(phi+psi)/d_m]^2.

        Parameters
        ----------
        nonlinear : bool
                Whether the returned power spectrum values are linear or non-linear (default)
        h_units : bool
                Whether the units of k in output are h/Mpc or 1/Mpc (default)

        Returns
        -------
        Weyl_pk : grid of Weyl potential (phi+psi) spectrum values, Weyl_pk[index_k,index_z]
        k : vector of k values, k[index_k] (in units of 1/Mpc by default, or h/Mpc when setting h_units to True)
        z : vector of z values, z[index_z]
        
---------------------------------
Function: sigma(self,R,z, h_units = False)
Docstring:

        Gives sigma (total matter) for a given R and z
        (R is the radius in units of Mpc, so if R=8/h this will be the usual sigma8(z).
         This is unless h_units is set to true, in which case R is the radius in units of Mpc/h,
         and R=8 corresponds to sigma8(z))

        .. note::

            there is an additional check to verify whether output contains `mPk`,
            and whether k_max > ...
            because otherwise a segfault will occur

        
---------------------------------
Function: sigma_cb(self,double R,double z, h_units = False)
Docstring:

        Gives sigma (cdm+b) for a given R and z
        (R is the radius in units of Mpc, so if R=8/h this will be the usual sigma8(z)
         This is unless h_units is set to true, in which case R is the radius in units of Mpc/h,
         and R=8 corresponds to sigma8(z))

        .. note::

            there is an additional check to verify whether output contains `mPk`,
            and whether k_max > ...
            because otherwise a segfault will occur

        
---------------------------------
Function: pk_tilt(self,double k,double z)
Docstring:

        Gives effective logarithmic slope of P_L(k,z) (total matter) for a given k and z
        (k is the wavenumber in units of 1/Mpc, z is the redshift, the output is dimensionless)

        .. note::

            there is an additional check to verify whether output contains `mPk` and whether k is in the right range

        
---------------------------------
Function: angular_distance(self, z)
Docstring:

        angular_distance(z)

        Return the angular diameter distance (exactly, the quantity defined by Class
        as index_bg_ang_distance in the background module)

        Parameters
        ----------
        z : float
                Desired redshift
        
---------------------------------
Function: angular_distance_from_to(self, z1, z2)
Docstring:

        angular_distance_from_to(z)

        Return the angular diameter distance of object at z2 as seen by observer at z1,
        that is, sin_K((chi2-chi1)*np.sqrt(|k|))/np.sqrt(|k|)/(1+z2).
        If z1>z2 returns zero.

        Parameters
        ----------
        z1 : float
                Observer redshift
        z2 : float
                Source redshift

        Returns
        -------
        d_A(z1,z2) in Mpc
        
---------------------------------
Function: comoving_distance(self, z)
Docstring:

        comoving_distance(z)

        Return the comoving distance

        Parameters
        ----------
        z : float
                Desired redshift
        
---------------------------------
Function: scale_independent_growth_factor(self, z)
Docstring:

        scale_independent_growth_factor(z)

        Return the scale invariant growth factor D(a) for CDM perturbations
        (exactly, the quantity defined by Class as index_bg_D in the background module)

        Parameters
        ----------
        z : float
                Desired redshift
        
---------------------------------
Function: scale_independent_growth_factor_f(self, z)
Docstring:

        scale_independent_growth_factor_f(z)

        Return the scale independent growth factor f(z)=d ln D / d ln a for CDM perturbations
        (exactly, the quantity defined by Class as index_bg_f in the background module)

        Parameters
        ----------
        z : float
                Desired redshift
        
---------------------------------
Function: scale_dependent_growth_factor_f(self, k, z, h_units=False, nonlinear=False, Nz=20)
Docstring:

        scale_dependent_growth_factor_f(k,z)

        Return the scale dependent growth factor
        f(z)= 1/2 * [d ln P(k,a) / d ln a]
            = - 0.5 * (1+z) * [d ln P(k,z) / d z]
        where P(k,z) is the total matter power spectrum

        Parameters
        ----------
        z : float
                Desired redshift
        k : float
                Desired wavenumber in 1/Mpc (if h_units=False) or h/Mpc (if h_units=True)
        
---------------------------------
Function: scale_dependent_growth_factor_f_cb(self, k, z, h_units=False, nonlinear=False, Nz=20)
Docstring:

        scale_dependent_growth_factor_f_cb(k,z)

        Return the scale dependent growth factor calculated from CDM+baryon power spectrum P_cb(k,z)
        f(z)= 1/2 * [d ln P_cb(k,a) / d ln a]
            = - 0.5 * (1+z) * [d ln P_cb(k,z) / d z]


        Parameters
        ----------
        z : float
                Desired redshift
        k : float
                Desired wavenumber in 1/Mpc (if h_units=False) or h/Mpc (if h_units=True)
        
---------------------------------
Function: scale_independent_f_sigma8(self, z)
Docstring:

        scale_independent_f_sigma8(z)

        Return the scale independent growth factor f(z) multiplied by sigma8(z)

        Parameters
        ----------
        z : float
                Desired redshift

        Returns
        -------
        f(z)*sigma8(z) (dimensionless)
        
---------------------------------
Function: effective_f_sigma8(self, z, z_step=0.1)
Docstring:

        effective_f_sigma8(z)

        Returns the time derivative of sigma8(z) computed as (d sigma8/d ln a)

        Parameters
        ----------
        z : float
                Desired redshift
        z_step : float
                Default step used for the numerical two-sided derivative. For z < z_step the step is reduced progressively down to z_step/10 while sticking to a double-sided derivative. For z< z_step/10 a single-sided derivative is used instead.

        Returns
        -------
        (d ln sigma8/d ln a)(z) (dimensionless)
        
---------------------------------
Function: effective_f_sigma8_spline(self, z, Nz=20)
Docstring:

        effective_f_sigma8_spline(z)

        Returns the time derivative of sigma8(z) computed as (d sigma8/d ln a)

        Parameters
        ----------
        z : float
                Desired redshift
        Nz : integer
                Number of values used to spline sigma8(z) in the range [z-0.1,z+0.1]

        Returns
        -------
        (d ln sigma8/d ln a)(z) (dimensionless)
        
---------------------------------
Function: z_of_tau(self, tau)
Docstring:

        Redshift corresponding to a given conformal time.

        Parameters
        ----------
        tau : float
                Conformal time
        
---------------------------------
Function: Hubble(self, z)
Docstring:

        Hubble(z)

        Return the Hubble rate (exactly, the quantity defined by Class as index_bg_H
        in the background module)

        Parameters
        ----------
        z : float
                Desired redshift
        
---------------------------------
Function: Om_m(self, z)
Docstring:

        Omega_m(z)

        Return the matter density fraction (exactly, the quantity defined by Class as index_bg_Omega_m
        in the background module)

        Parameters
        ----------
        z : float
                Desired redshift
        
---------------------------------
Function: Om_b(self, z)
Docstring:

        Omega_b(z)

        Return the baryon density fraction (exactly, the ratio of quantities defined by Class as
        index_bg_rho_b and index_bg_rho_crit in the background module)

        Parameters
        ----------
        z : float
                Desired redshift
        
---------------------------------
Function: Om_cdm(self, z)
Docstring:

        Omega_cdm(z)

        Return the cdm density fraction (exactly, the ratio of quantities defined by Class as
        index_bg_rho_cdm and index_bg_rho_crit in the background module)

        Parameters
        ----------
        z : float
                Desired redshift
        
---------------------------------
Function: Om_ncdm(self, z)
Docstring:

        Omega_ncdm(z)

        Return the ncdm density fraction (exactly, the ratio of quantities defined by Class as
        Sum_m [ index_bg_rho_ncdm1 + n ], with n=0...N_ncdm-1, and index_bg_rho_crit in the background module)

        Parameters
        ----------
        z : float
                Desired redshift
        
---------------------------------
Function: ionization_fraction(self, z)
Docstring:

        ionization_fraction(z)

        Return the ionization fraction for a given redshift z

        Parameters
        ----------
        z : float
                Desired redshift
        
---------------------------------
Function: baryon_temperature(self, z)
Docstring:

        baryon_temperature(z)

        Give the baryon temperature for a given redshift z

        Parameters
        ----------
        z : float
                Desired redshift
        
---------------------------------
Function: T_cmb(self)
Docstring:

        Return the CMB temperature
        
---------------------------------
Function: Omega0_m(self)
Docstring:

        Return the sum of Omega0 for all non-relativistic components
        
---------------------------------
Function: get_background(self)
Docstring:

        Return an array of the background quantities at all times.

        Parameters
        ----------

        Returns
        -------
        background : dictionary containing background.
        
---------------------------------
Function: get_thermodynamics(self)
Docstring:

        Return the thermodynamics quantities.

        Returns
        -------
        thermodynamics : dictionary containing thermodynamics.
        
---------------------------------
Function: get_primordial(self)
Docstring:

        Return the primordial scalar and/or tensor spectrum depending on 'modes'.
        'output' must be set to something, e.g. 'tCl'.

        Returns
        -------
        primordial : dictionary containing k-vector and primordial scalar and tensor P(k).
        
---------------------------------
Function: get_perturbations(self, return_copy=True)
Docstring:

        Return scalar, vector and/or tensor perturbations as arrays for requested
        k-values.

        .. note::

            you need to specify both 'k_output_values', and have some
            perturbations computed, for instance by setting 'output' to 'tCl'.

            Do not enable 'return_copy=False' unless you know exactly what you are doing.
            This will mean that you get access to the direct C pointers inside CLASS.
            That also means that if class is deallocated,
            your perturbations array will become invalid. Beware!

        Returns
        -------
        perturbations : dict of array of dicts
                perturbations['scalar'] is an array of length 'k_output_values' of
                dictionary containing scalar perturbations.
                Similar for perturbations['vector'] and perturbations['tensor'].
        
---------------------------------
Function: get_transfer(self, z=0., output_format='class')
Docstring:

        Return the density and/or velocity transfer functions for all initial
        conditions today. You must include 'mTk' and/or 'vCTk' in the list of
        'output'. The transfer functions can also be computed at higher redshift z
        provided that 'z_pk' has been set and that 0<z<z_pk.

        Parameters
        ----------
        z  : redshift (default = 0)
        output_format  : ('class' or 'camb') Format transfer functions according to
                         CLASS convention (default) or CAMB convention.

        Returns
        -------
        tk : dictionary containing transfer functions.
        
---------------------------------
Function: get_current_derived_parameters(self, names)
Docstring:

        get_current_derived_parameters(names)

        Return a dictionary containing an entry for all the names defined in the
        input list.

        Parameters
        ----------
        names : list
                Derived parameters that can be asked from Monte Python, or
                elsewhere.

        Returns
        -------
        derived : dict

        .. warning::

            This method used to take as an argument directly the data class from
            Monte Python. To maintain compatibility with this old feature, a
            check is performed to verify that names is indeed a list. If not, it
            returns a TypeError. The old version of this function, when asked
            with the new argument, will raise an AttributeError.

        
---------------------------------
Function: nonlinear_scale(self, np.ndarray[DTYPE_t,ndim=1] z, int z_size)
Docstring:

        nonlinear_scale(z, z_size)

        Return the nonlinear scale for all the redshift specified in z, of size
        z_size

        Parameters
        ----------
        z : numpy array
                Array of requested redshifts
        z_size : int
                Size of the redshift array
        
---------------------------------
Function: nonlinear_scale_cb(self, np.ndarray[DTYPE_t,ndim=1] z, int z_size)
Docstring:


make        nonlinear_scale_cb(z, z_size)

        Return the nonlinear scale for all the redshift specified in z, of size

        z_size

        Parameters
        ----------
        z : numpy array
                Array of requested redshifts
        z_size : int
                Size of the redshift array
        
---------------------------------
Function: __call__(self, ctx)
Docstring:

        Function to interface with CosmoHammer

        Parameters
        ----------
        ctx : context
                Contains several dictionaries storing data and cosmological
                information

        
---------------------------------
Function: get_pk_array(self, np.ndarray[DTYPE_t,ndim=1] k, np.ndarray[DTYPE_t,ndim=1] z, int k_size, int z_size, nonlinear)
Docstring:
 Fast function to get the power spectrum on a k and z array 
---------------------------------
Function: get_pk_cb_array(self, np.ndarray[DTYPE_t,ndim=1] k, np.ndarray[DTYPE_t,ndim=1] z, int k_size, int z_size, nonlinear)
Docstring:
 Fast function to get the power spectrum on a k and z array 
---------------------------------
Function: Omega0_k(self)
Docstring:
 Curvature contribution 
---------------------------------
Function: get_sources(self)
Docstring:

        Return the source functions for all k, tau in the grid.

        Returns
        -------
        sources : dictionary containing source functions.
        k_array : numpy array containing k values.
        tau_array: numpy array containing tau values.
        
---------------------------------

```

## classy-generated-docstrings.txt

```
```python                           
"""
The following docstrings were extracted from classy-py.py
"""
def viewdictitems(d):
    """Return items from a dictionary for Python 2 and 3 compatibility.

    Args:
        d (dict): The dictionary to retrieve items from.

    Returns:
        A view of the dictionary's items.
    """
def _check_task_dependency(self, level):
    """Fill the level list with all the needed modules.

    Warning:
        The ordering of modules is obviously dependent on CLASS module order
        in the main.c file. This has to be updated in case of a change to
        this file.

    Args:
        level (list): List of strings, containing initially only the last module required.
            For instance, to recover all the modules, the input should be
            ['lensing'].
    """
def _pars_check(self, key, value, contains=False, add=""):
    """Check parameters (implementation detail, no docstring provided)."""
def compute(self, level=["distortions"]):
    """Compute the CLASS cosmology.

    Main function, execute all the _init methods for all desired modules.
    This is called in MontePython, and this ensures that the Class instance
    of this class contains all the relevant quantities. Then, one can deduce
    Pk, Cl, etc...

    Args:
        level (list, optional): List of the last module desired. The internal function
            _check_task_dependency will then add to this list all the
            necessary modules to compute in order to initialize this last
            one. The default last module is "lensing". Defaults to ["distortions"].

    Warning:
        level default value should be left as an array (it was creating
        problem when casting as a set later on, in _check_task_dependency)
    """
def raw_cl(self, lmax=-1, nofail=False):
    """Return a dictionary of the primary C_l.

    Args:
        lmax (int, optional): Define the maximum l for which the C_l will be returned
            (inclusively). This number will be checked against the maximum l
            at which they were actually computed by CLASS, and an error will
            be raised if the desired lmax is bigger than what CLASS can
            give. Defaults to -1.
        nofail (bool, optional): Check and enforce the computation of the harmonic module
            beforehand, with the desired lmax. Defaults to False.

    Returns:
        dict: Dictionary that contains the power spectrum for 'tt', 'te', etc... The
            index associated with each is defined wrt. Class convention, and are non
            important from the python point of view. It also returns now the
            ell array.
    """
def lensed_cl(self, lmax=-1,nofail=False):
    """Return a dictionary of the lensed C_l.

    Return a dictionary of the lensed C_l, computed by CLASS, without the
    density C_ls. They must be asked separately with the function aptly
    named density_cl

    Args:
        lmax (int, optional): Define the maximum l for which the C_l will be returned (inclusively). Defaults to -1.
        nofail (bool, optional): Check and enforce the computation of the lensing module beforehand. Defaults to False.

    Returns:
        dict: Dictionary that contains the power spectrum for 'tt', 'te', etc... The
            index associated with each is defined wrt. Class convention, and are non
            important from the python point of view.
    """
def density_cl(self, lmax=-1, nofail=False):
    """Return a dictionary of the primary C_l for the matter.

    Args:
        lmax (int, optional): Define the maximum l for which the C_l will be returned (inclusively). Defaults to -1.
        nofail (bool, optional): Check and enforce the computation of the lensing module beforehand. Defaults to False.

    Returns:
        numpy array of numpy.ndarrays: Array that contains the list (in this order) of self correlation of
            1st bin, then successive correlations (set by non_diagonal) to the
            following bins, then self correlation of 2nd bin, etc. The array
            starts at index_ct_dd.
    """
def sigma(self,R,z, h_units = False):
    """Give sigma (total matter) for a given R and z.

    (R is the radius in units of Mpc, so if R=8/h this will be the usual sigma8(z).
    This is unless h_units is set to true, in which case R is the radius in units of Mpc/h,
    and R=8 corresponds to sigma8(z))

    Note:
        there is an additional check to verify whether output contains `mPk`,
        and whether k_max > ...
        because otherwise a segfault will occur

    Args:
        R:
        z:
        h_units:
    """
def sigma_cb(self,double R,double z, h_units = False):
    """Give sigma (cdm+b) for a given R and z.

    (R is the radius in units of Mpc, so if R=8/h this will be the usual sigma8(z)
    This is unless h_units is set to true, in which case R is the radius in units of Mpc/h,
    and R=8 corresponds to sigma8(z))

    Note:
        there is an additional check to verify whether output contains `mPk`,
        because otherwise a segfault will occur

    Args:
        R:
        z:
        h_units:
    """
def pk_tilt(self,double k,double z):
    """Give effective logarithmic slope of P_L(k,z) (total matter) for a given k and z.

    (k is the wavenumber in units of 1/Mpc, z is the redshift, the output is dimensionless)

    Note:
        there is an additional check that output contains `mPk` and whether k is in the right range

    Args:
        k:
        z:
    """
def age(self):
    """Return the age of the Universe (implementation detail, no docstring provided)."""
def h(self):
    """Return the Hubble parameter (implementation detail, no docstring provided)."""
def n_s(self):
    """Return the scalar spectral index (implementation detail, no docstring provided)."""
def tau_reio(self):
    """Return the reionization optical depth (implementation detail, no docstring provided)."""
def Omega_m(self):
    """Return the matter density parameter (implementation detail, no docstring provided)."""
def Omega_r(self):
    """Return the radiation density parameter (implementation detail, no docstring provided)."""
def theta_s_100(self):
    """Return the sound horizon angle (implementation detail, no docstring provided)."""
def theta_star_100(self):
    """Return the sound horizon angle at decoupling (implementation detail, no docstring provided)."""
def Omega_Lambda(self):
    """Return the cosmological constant density parameter (implementation detail, no docstring provided)."""
def Omega_g(self):
    """Return the photon density parameter (implementation detail, no docstring provided)."""
def r(self):
    """Return the tensor-to-scalar ratio (implementation detail, no docstring provided)."""
def A_s(self):
    """Return the primordial power spectrum amplitude (implementation detail, no docstring provided)."""
def ln_A_s_1e10(self):
    """Return the natural logarithm of 10^10 times the primordial power spectrum amplitude (implementation detail, no docstring provided)."""
def lnAs_1e10(self):
    """Return the natural logarithm of 10^10 times the primordial power spectrum amplitude (implementation detail, no docstring provided)."""
def Neff(self):
    """Return the effective number of neutrino species (implementation detail, no docstring provided)."""
def get_transfer(self, z=0., output_format='class'):
    """Return the density and/or velocity transfer functions.

    Return the density and/or velocity transfer functions for all initial
    conditions, at a given value of z.
    By default, all transfer functions will be normalised to 'curvature R=1'
    at initial time (and not 'curvature R = -1/k^2' like in CAMB).
    You may switch to output_format='camb' for the CAMB definition and normalisation
    of transfer functions.
    When setting CLASS input parameters, include at least one of 'dTk' (for density transfer functions)
    or 'vTk' (for velocity transfer functions).
    For more details, see section II of the CLASS notebook.

    Args:
        z (float, optional): Redshift. Defaults to 0..
        output_format (str, optional): Format transfer functions according to CLASS (default) or CAMB. Defaults to 'class'.

    Returns:
        dict: Dictionary containing an entry for each transfer function. For a
        given transfer function, say, delta_tot, transfers['d_tot'] will be
        an array containing delta_tot(k) at the k values defined in the
        'k_output_values' list. When there are non-adiabatic conditions,
        the transfer dictionary will have keys like transfers['d_tot[name]'], where
        name is the name of the isocurvature mode.
    """
def get_current_derived_parameters(self, names):
    """Return a dictionary containing an entry for all the names defined in the input list.

    Args:
        names (list): Derived parameters that can be asked from Monte Python, or
            elsewhere.

    Returns:
        dict: A dictionary of derived parameters.

    Raises:
        TypeError: If `names` is not a list.
    """
def get_perturbations(self, return_copy=True):
    """Return scalar, vector and/or tensor perturbations as arrays for requested k-values.

    Note:
        you need to specify both 'k_output_values', and have some
        perturbations computed, for instance by setting 'output' to 'tCl'.
        Do not enable 'return_copy=False' unless you know exactly what you are doing.
        This will mean that you get access to the direct C pointers inside CLASS.
        That also means that if class is deallocated,
        your perturbations array will become invalid. Beware!

    Args:
        return_copy (bool, optional): Whether to return a copy of the data. Defaults to True.

    Returns:
        dict of array of dicts: perturbations['scalar'] is an array of length 'k_output_values' of
        dictionary containing scalar perturbations.
        Similar for perturbations['vector'] and perturbations['tensor'].
    """
def scale_dependent_growth_factor_f(self, k, z, Nz=50, h_units = False, evolution=False):
    """Return the scale-dependent growth factor, f(k,z) = d ln delta(k,z) / d ln a, at a given k and z, for total matter fluctuations.

    Args:
        k (float or array): wavenumber in units of 1/Mpc
        z (float or array): redshift
        Nz (int, optional): number of points for computing sigma(R,z) splines, default to 50. Defaults to 50.
        h_units (bool, optional): If true, returns k in h/Mpc. Defaults to False.
        evolution (bool, optional): . Defaults to False.
    """
def pk(self, k, z, lAccuracy=10):
    """Return the total matter power spectrum for a given k and z.

    Return the total matter power spectrum for a given k and z (will be
    non linear if requested to Class, linear otherwise)

    Args:
        k (float): wavenumber in units of 1/Mpc
        z (float): redshift
        lAccuracy (int, optional): Level of accuracy of the integration. Defaults to 10.
    """
def pk_cb(self,double k,double z):
    """Give the cdm+b pk (in Mpc**3) for a given k (in 1/Mpc) and z (will be non linear if requested to Class, linear otherwise).

    Note:
        there is an additional check that output contains `mPk`,
        because otherwise a segfault will occur

    Args:
        k:
        z:
    """
def pk_lin(self, k, z, lAccuracy=10):
    """Return the LINEAR total matter power spectrum for a given k and z.

    Args:
        k (float): wavenumber in units of 1/Mpc
        z (float): redshift
        lAccuracy (int, optional): Level of accuracy of the integration. Defaults to 10.
    """
def pk_cb_lin(self,double k,double z):
    """Give the LINEAR cdm+b pk (in Mpc**3) for a given k (in 1/Mpc) and z.

    Note:
        there is an additional check that output contains `mPk`,
        because otherwise a segfault will occur

    Args:
        k:
        z:
    """
def log_pk(self, k, z, lAccuracy=10):
    """Return the log of the total matter power spectrum for a given k and z.

    Args:
        k (float): wavenumber in units of 1/Mpc
        z (float): redshift
        lAccuracy (int, optional): Level of accuracy of the integration. Defaults to 10.
    """
def transfer(self, k, z, idx_T=1, lAccuracy=10):
    """Return a transfer function for a given k and z.

    Args:
        k (float): wavenumber in units of 1/Mpc
        z (float): redshift
        idx_T (int, optional): index of transfer function to return, with 0=delta_g, 1=delta_b,
            2=delta_cdm, 3=delta_ncdm[0], etc.... Defaults to 1.
        lAccuracy (int, optional): Level of accuracy of the integration. Defaults to 10.
    """
def rho_crit(self, z, lAccuracy=10):
    """Return the critical density at redshift z.

    Args:
        z (float): redshift
        lAccuracy (int, optional): Level of accuracy of the integration. Defaults to 10.
    """
def scale_independent_f_sigma8(self, z, Nz=50):
    """Return an interpolator for f \\sigma_8 (scale-INdependent), as a function of z.

    This will compute f(z) = d ln delta / d ln a,
    approximating this quantity with the scale-INdependent growth rate.
    For the scale-dependent one, use the proper function.

    Args:
        z (array): Redshift
    """
def scale_independent_growth_factor(self, z, Nz=50):
    """Return the linear growth factor by taking the ratio of Delta(z)/Delta(z=0).

    Args:
        z (array): Redshift
    """
def has_idr(self):
    """Check for interacting dark radiation (implementation detail, no docstring provided)."""
def has_dr(self):
    """Check for dark radiation (implementation detail, no docstring provided)."""
def spectral_distortion_amplitudes(self):
    """Distortion amplitudes (implementation detail, no docstring provided)."""
def get_transfer_functions_at_z(self, z, k_values, output_format='class'):
    """Return the density and velocity transfer functions.

    Return the density and velocity transfer functions for all initial
    conditions, at a given value of z.
    For more details, see section II of the CLASS notebook.
    By default, all transfer functions will be normalised to 'curvature R=1' at initial time
    (and not 'curvature R = -1/k^2' like in CAMB).
    You may switch to output_format='camb' for the CAMB definition and normalisation of transfer functions.
    When setting CLASS input parameters, include at least one of 'dTk' (for density transfer functions)
    or 'vTk' (for velocity transfer functions).

    Args:
        z (float): Redshift
        k_values:
        output_format (str, optional): Format transfer functions according to CLASS (default) or CAMB. Defaults to 'class'.
    """
def primordial_spec(self, k, return_power=True, lAccuracy=10):
    """Return the primordial power spectrum.

    This function switches between the scalar and tensor primordial power
    spectrum, and accepts as an argument a scale k.

    Args:
        k (float): wavenumber in units of 1/Mpc
        return_power (bool, optional): default value is true, which returns the power spectrum, otherwise the value of the scale is returned. Defaults to True.
        lAccuracy (int, optional): Level of accuracy of the integration. Defaults to 10.
    """
def primordial_scalar_pk(self, k, lAccuracy=10):
    """Return the primordial SCALAR power spectrum for a given k.

    Args:
        k (float): wavenumber in units of 1/Mpc
        lAccuracy (int, optional): Level of accuracy of the integration. Defaults to 10.
    """
def primordial_tensor_pk(self, k, lAccuracy=10):
    """Return the primordial TENSOR power spectrum for a given k.

    Args:
        k (float): wavenumber in units of 1/Mpc
        lAccuracy (int, optional): Level of accuracy of the integration. Defaults to 10.
    """
def angular_diamater_distance(self,z):
    """Return the angular diameter distance."""
def tangential_critical_density(self,z_l,z_s):
    """Returnthe critical density for tangential shear."""


```

