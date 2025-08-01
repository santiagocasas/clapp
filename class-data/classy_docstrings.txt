func:  def viewdictitems(d):
class: class CosmoError(Exception):
func:      def __str__(self):
class: class CosmoSevereError(CosmoError):
doc:   "
    Raised when Class failed to understand one or more input parameters.

    This case would not raise any problem in Class default behaviour. However,
    for parameter extraction, one has to be sure that all input parameters were
    understood, otherwise the wrong cosmological model would be selected.
    "

class: class CosmoComputationError(CosmoError):
doc:   "
    Raised when Class could not compute the cosmology at this point.

    This will be caught by the parameter extraction code to give an extremely
    unlikely value to this point
    "

class: cdef class Class:
doc:   "
    Class wrapping, creates the glue between C and python

    The actual Class wrapping, the only class we will call from MontePython
    (indeed the only one we will import, with the command:
    from classy import Class

    "

func:          def __get__(self):
func:          def __get__(self):
func:          def __get__(self):
func:          def __get__(self):
func:      def set_default(self):
func:      def __cinit__(self, default=False):
func:      def __dealloc__(self):
func:      def set(self,*pars,**kars):
func:      def empty(self):
func:      def _fillparfile(self):
func:      def _check_task_dependency(self, level):
doc:   "
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

        "

func:      def _pars_check(self, key, value, contains=False, add=""):
func:      def compute(self, level=["distortions"]):
doc:   "
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

        "

func:      def set_baseline(self, baseline_name):
doc:   "
        Set all input parameters to predefined baseline settings

        This function offers an opportunity to quickly set all input parameters in one single line.

        Currently, the possible predefined setting names are:

        1. 'p18lb', or alternatively, any name containing 'planck', '18', 'lens', and 'bao'
           Sets all parameters for computing the lensed and unlensed CMB l's and the matter power spectrum P(k) at z=0
           (linear and non-linear with Halofit corrections)
           for the best-fit model of the case 'Planck TTTEEE+lens + BAO' of the Planck 2018 papers.

        2. 'p18l', or alternatively, any name containing 'planck', '18', and 'lens'
           Sets all parameters for computing the lensed and unlensed CMB l's and the matter power spectrum P(k) at z=0
           (linear and non-linear with Halofit corrections)
           for the best-fit model of the case 'Planck TTTEEE+lens' of the Planck 2018 papers.

        3. 'p18', or alternatively, any name containing 'planck' and '18'
           Sets all parameters for computing the lensed and unlensed CMB l's and the matter power spectrum P(k) at z=0
           (linear and non-linear with Halofit corrections)
           for the best-fit model of the case 'Planck TTTEEE' of the Planck 2018 papers.

        This choice of parameter settings is the same as in the baseline input files (input/baseline*.param)
        of montepython [https://github.com/brinckmann/montepython_public] (see also 1210.7183, 1804.07261)

        Parameters
        ----------
        baseline_name : str
            Predefined setting name

        Returns
        -------
        None
        "

func:      def density_factor(self):
doc:   "
        Conversion factor from CLASS density to physical density in SI units

        Returns the factor to convert from the CLASS units of density, (8piG/3c^2 rho) in Mpc^-2,
        to the actual density rho in kg/m^3 (SI units)

        Parameters
        ----------
        None

        Returns
        -------
        float
            The conversion factor
        "

func:      def Mpc_to_m(self):
doc:   "
        Conversion factor from Mpc to m

        Returns the factor to convert from Megaparsecs to meters
        (factor defined internally in CLASS)

        Parameters
        ----------
        None

        Returns
        -------
        float
            The conversion factor
        "

func:      def kg_to_eV(self):
doc:   "
        Conversion factor from kg to eV

        Returns the factor to convert from kilograms to electron-volts when using natural units
        (inferred from factors defined internally in CLASS)

        Parameters
        ----------
        None

        Returns
        -------
        float
            The conversion factor
        "

func:      def kgm3_to_eVMpc3(self):
doc:   "
        Conversion factor from kg/m^3 to eV/Mpc^3

        Returns the factor to convert from kilograms per meter cube to electron-volts per Megaparsec cube when using natural units

        Parameters
        ----------
        None

        Returns
        -------
        float
            The conversion factor
        "

func:      def kg_to_Msol(self):
doc:   "
        Conversion factor from kg to Msol

        Returns the factor to convert from kilograms to solar mass

        Parameters
        ----------
        None

        Returns
        -------
        float
            The conversion factor
        "

func:      def kgm3_to_MsolMpc3(self):
doc:   "
        Conversion factor from kg/m^3 to Msol/Mpc^3

        Returns the factor to convert from kilograms per meter cube to solar mass per Megaparsec cube

        Parameters
        ----------
        None

        Returns
        -------
        float
            The conversion factor
        "

func:      def raw_cl(self, lmax=-1, nofail=False):
doc:   "
        Unlensed CMB spectra

        Return a dictionary of the unlensed CMB spectra C_l for all modes requested in the 'output' field
        (temperature, polarisation, lensing potential, cross-correlations...)
        This function requires that the 'ouput' field contains at least on of 'tCl', 'pCl', 'lCl'.
        The returned C_l's are all dimensionless.

        If lmax is not passed (default), the spectra are returned up to the maximum multipole that was set through input parameters.
        If lmax is passed, the code first checks whether it is smaller or equal to the maximum multipole that was set through input parameters.
        If it is smaller or equal, the spectra are returned up to lmax.
        If it is larger and nofail=False (default), the function raises an error.
        If it is larger and nofail=True, the functiona asks CLASS to recompute the Cl's up to lmax, and return the result up to lmax.

        Parameters
        ----------
        lmax : int, optional
            Define the maximum l for which the C_l will be returned (inclusively).

        nofail: bool, optional
            Even if lmax is larger than expected from the user input, check and enforce the computation of the C_l in the harmonic module up to lmax.

        Returns
        -------
        cl : dict
            Dictionary that contains the unlensed power spectrum for each auto-correlation and cross-correlation spectrum,
            as well the corresponding mutipoles l. The keys of the dictionary are: 'ell' for the multipoles,
            and 'tt', 'ee', 'te', 'bb', 'pp', 'tp' for each spectrum type.
        "

func:      def lensed_cl(self, lmax=-1,nofail=False):
doc:   "
        Lensed CMB spectra

        Return a dictionary of the lensed CMB spectra C_l for all modes requested in the 'output' field
        (temperature, polarisation, lensing potential, cross-correlations...)
        This function requires that the 'ouput' field contains at least one of 'tCl', 'pCl', 'lCl', and that 'lensing' is set to 'yes'.
        The returned spectra are all dimensionless.

        If lmax is not passed (default), the spectra are returned up to the maximum multipole that was set through input parameters.
        If lmax is passed, the code first checks whether it is smaller or equal to the maximum multipole that was set through input parameters.
        If it is smaller or equal, the spectra are returned up to lmax.
        If it is larger and nofail=False (default), the function raises an error.
        If it is larger and nofail=True, the functiona asks CLASS to recompute the Cl's up to lmax, and return the result up to lmax.

        Parameters
        ----------
        lmax : int, optional
            Define the maximum l for which the C_l will be returned (inclusively).

        nofail: bool, optional
            Even if lmax is larger than expected from the user input, check and enforce the computation of the C_l in the lensing module up to lmax.

        Returns
        -------
        cl : dict
            Dictionary that contains the lensed power spectrum for each auto-correlation and cross-correlation spectrum,
            as well the corresponding mutipoles l. The keys of the dictionary are: 'ell' for the multipoles,
            and 'tt', 'ee', 'te', 'bb', 'pp', 'tp' for each spectrum type.
        "

func:      def density_cl(self, lmax=-1, nofail=False):
doc:   "
        C_l spectra of density and lensing

        Return a dictionary of the spectra C_l of large scale
        structure observables, that is, density (e.g. from galaxy number count)
        and lensing potential (related to cosmic shear), for all the
        reshift bins defined by the user in input. If CMB temperature
        was also requested, this function also returns the
        density-temperature and lensing-temperature cross-correlation
        spectra.
        This function requires that the 'ouput' field contains at least one of 'dCl', 'sCl'.
        The returned spectra are all dimensionless.

        If lmax is not passed (default), the spectra are returned up to the maximum multipole that was set through input parameters.
        If lmax is passed, the code first checks whether it is smaller or equal to the maximum multipole that was set through input parameters.
        If it is smaller or equal, the spectra are returned up to lmax.
        If it is larger and nofail=False (default), the function raises an error.
        If it is larger and nofail=True, the functiona asks CLASS to recompute the Cl's up to lmax, and return the result up to lmax.

        Parameters
        ----------
        lmax : int, optional
            Define the maximum l for which the C_l will be returned (inclusively).

        nofail: bool, optional
            Even if lmax is larger than expected from the user input, check and enforce the computation of the C_l in the lensing module up to lmax.

        Returns
        -------
        cl : dict
            Dictionary that contains the power spectrum for each auto-correlation and cross-correlation spectrum,
            as well the corresponding mutipoles l. The keys of the dictionary are: 'ell' for the multipoles,
            and 'dens', 'td', 'll', 'dl', 'tl', where 'd' means density, 'l' means lensing, and 't' means temperature.
        "

func:      def z_of_r (self, z):
doc:   "
        Return the comoving radius r(z) and the derivative dz/dr.

        Return the comoving radius r(z) (in units of Mpc) of an object
        seen at redshift z (dimensionless), as well and the derivative
        dz/dr (in units of 1/Mpc). The name of the function is
        misleading, it should be r_of_z. This naming inconsistency
        propagated until now and could be fixed at some point. The
        function accepts single entries or arrays of z values.

        Parameters
        ----------
        z : float array
            Redshift

        Returns
        -------
        float array
            Comoving radius

        real array
            Derivative dz/dr
        "

func:      def luminosity_distance(self, z):
doc:   "
        Return the luminosity_distance d_L(z)

        Return the luminosity distance d_L(z) in units of Mpc.
        The function accepts single entries or arrays of z values.

        Parameters
        ----------
        z : float array
            Redshift

        Returns
        -------
        float array
            Luminosity distance
        "

func:      def pk(self,double k,double z):
doc:   "
        Return the total matter power spectrum P_m(k,z)

        Return the total matter power spectrum P_m(k,z) (in Mpc**3) for a given k (in
        1/Mpc) and z. The function returns the linear power spectrum
        if the user sets 'non_linear' to 'no', and the non-linear
        power spectrum otherwise.
        This function requires that the 'ouput' field contains at least 'mPk'.

        Parameters
        ----------
        k : float
            Wavenumber
        z : float
            Redshift

        Returns
        -------
        pk : float
            Total matter power spectrum
        "

func:      def pk_cb(self,double k,double z):
doc:   "
        Return the CDM+baryon power spectrum P_cb(k,z)

        Return the CDM+baryon power spectrum P_cb(k,z) (in Mpc**3) for a given k (in
        1/Mpc) and z. The function returns the linear power spectrum
        if the user sets 'non_linear' to 'no', and the non-linear
        power spectrum otherwise.
        This function requires that the 'ouput' field contains at least 'mPk'.

        Parameters
        ----------
        k : float
            Wavenumber
        z : float
            Redshift

        Returns
        -------
        pk_cb : float
            CDM+baryon power spectrum
        "

func:      def pk_lin(self,double k,double z):
doc:   "
        Return the linear total matter power spectrum P_m(k,z)

        Return the linear total matter power spectrum P_m(k,z) (in Mpc**3) for a given k (in
        1/Mpc) and z. The function returns the linear power spectrum
        in all cases, independently of what the user sets for 'non_linear'.
        This function requires that the 'ouput' field contains at least 'mPk'.

        Parameters
        ----------
        k : float
            Wavenumber
        z : float
            Redshift

        Returns
        -------
        pk_lin : float
            Linear total matter power spectrum
        "

func:      def pk_cb_lin(self,double k,double z):
doc:   "
        Return the linear CDM+baryon power spectrum P_cb(k,z)

        Return the linear CDM+baryon power spectrum P_cb(k,z) (in Mpc**3) for a given k (in
        1/Mpc) and z. The function returns the linear power spectrum
        in all cases, independently of what the user sets for 'non_linear'.
        This function requires that the 'ouput' field contains at least 'mPk'.

        Parameters
        ----------
        k : float
            Wavenumber
        z : float
            Redshift

        Returns
        -------
        pk_cb_lin : float
            Linear CDM+baryon power spectrum
        "

func:      def pk_numerical_nw(self,double k,double z):
doc:   "
        Return the no-wiggle (smoothed) linear matter power spectrum P_m(k,z)

        Return the no-wiggle (smoothed) linear matter power spectrum P_m(k,z) (in Mpc**3) for a given k (in
        1/Mpc) and z. A smoothing algorithm infers this spectrum from the full linear matter power spectrum.
        This function requires that the 'ouput' field contains at least 'mPk', and 'numerical_nowiggle' is set to 'yes'.

        Parameters
        ----------
        k : float
            Wavenumber
        z : float
            Redshift

        Returns
        -------
        pk_numerical_nw : float
            No-wiggle linear matter power spectrum
        "

func:      def pk_analytic_nw(self,double k):
doc:   "
        Return an analytic approximation to the no-wiggle linear matter power spectrum P_m(k) at z=0

        Return an analytic approximation to the linear matter power spectrum P_m(k) (in Mpc**3) for a given k (in
        1/Mpc) and at z=0, without BAO features. The calculation is based on the Eisenstein & Hu fitting formulas.
        This function requires that the 'ouput' field contains at least 'mPk', and 'analytic_nowiggle' is set to 'yes'.

        Parameters
        ----------
        k : float
            Wavenumber

        Returns
        -------
        pk_analytic_nw : float
            Analytic approximation to no-wiggle linear matter power spectrum
        "

func:      def get_pk(self, np.ndarray[DTYPE_t,ndim=3] k, np.ndarray[DTYPE_t,ndim=1] z, int k_size, int z_size, int mu_size):
doc:   "
        Return the total matter power spectrum P_m(k,z) for a 3D array of k and a 1D array of z

        Return the total matter power spectrum P_m(k,z) (in Mpc**3)
        for a 3D array of k values (in 1/Mpc) and z values. The array
        of k values must be indexed as k[index_k,index_z,index_mu],
        the one of z values as z[index_z]. This function is useful in
        the context of likelihoods describing spectroscopic galaxy
        redshift, when there is a different k-list for each redhsift
        and each value of the angle mu between the line-of-sight and
        the measured two-point correlation function. The function
        returns a grid of values for the linear power spectrum if the
        user sets 'non_linear' to 'no', and for the non-linear power
        spectrum otherwise.  This function requires that the 'ouput'
        field contains at least 'mPk'.

        Parameters
        ----------
        k : float array
            Wavenumber array indexed as k[index_k,index_z,index_mu]
        z : float array
            Redshift array indexed as z[index_z]
        k_size : int
            Number of k values for each index_z and index_mu
        z_size : int
            Number of redshift values
        mu_size : int
            Number of k values for each index_k and index_z

        Returns
        -------
        pk : float
            Grid of total matter power spectrum indexed as pk[index_k,index_z,index_mu]
        "

func:      def get_pk_cb(self, np.ndarray[DTYPE_t,ndim=3] k, np.ndarray[DTYPE_t,ndim=1] z, int k_size, int z_size, int mu_size):
doc:   "
        Return the CDM+baryon power spectrum P_cb(k,z) for a 3D array of k and a 1D array of z

        Return the CDM+baryon power spectrum P_m(k,z) (in Mpc**3)
        for a 3D array of k values (in 1/Mpc) and z values. The array
        of k values must be indexed as k[index_k,index_z,index_mu],
        the one of z values as z[index_z]. This function is useful in
        the context of likelihoods describing spectroscopic galaxy
        redshift, when there is a different k-list for each redhsift
        and each value of the angle mu between the line-of-sight and
        the measured two-point correlation function. The function
        returns a grid of values for the linear power spectrum if the
        user sets 'non_linear' to 'no', and for the non-linear power
        spectrum otherwise.  This function requires that the 'ouput'
        field contains at least 'mPk'.

        Parameters
        ----------
        k : float array
            Wavenumber array indexed as k[index_k,index_z,index_mu]
        z : float array
            Redshift array indexed as z[index_z]
        k_size : int
            Number of k values for each index_z and index_mu
        z_size : int
            Number of redshift values
        mu_size : int
            Number of k values for each index_k and index_z

        Returns
        -------
        pk_cb : float
            Grid of CDM+baryon power spectrum indexed as pk_cb[index_k,index_z,index_mu]
        "

func:      def get_pk_lin(self, np.ndarray[DTYPE_t,ndim=3] k, np.ndarray[DTYPE_t,ndim=1] z, int k_size, int z_size, int mu_size):
doc:   "
        Return the linear total matter power spectrum P_m(k,z) for a 3D array of k and a 1D array of z

        Return the linear total matter power spectrum P_m(k,z) (in
        Mpc**3) for a 3D array of k values (in 1/Mpc) and z
        values. The array of k values must be indexed as
        k[index_k,index_z,index_mu], the one of z values as
        z[index_z]. This function is useful in the context of
        likelihoods describing spectroscopic galaxy redshift, when
        there is a different k-list for each redhsift and each value
        of the angle mu between the line-of-sight and the measured
        two-point correlation function. The function always returns a
        grid of values for the linear power spectrum, independently of
        what the user sets for 'non_linear'.  This function requires
        that the 'ouput' field contains at least 'mPk'.

        Parameters
        ----------
        k : float array
            Wavenumber array indexed as k[index_k,index_z,index_mu]
        z : float array
            Redshift array indexed as z[index_z]
        k_size : int
            Number of k values for each index_z and index_mu
        z_size : int
            Number of redshift values
        mu_size : int
            Number of k values for each index_k and index_z

        Returns
        -------
        pk : float
            Grid of linear total matter power spectrum indexed as pk[index_k,index_z,index_mu]
        "

func:      def get_pk_cb_lin(self, np.ndarray[DTYPE_t,ndim=3] k, np.ndarray[DTYPE_t,ndim=1] z, int k_size, int z_size, int mu_size):
doc:   "
        Return the linear CDM+baryon power spectrum P_cb(k,z) for a 3D array of k and a 1D array of z

        Return the linear CDM+baryon matter power spectrum P_m(k,z) (in
        Mpc**3) for a 3D array of k values (in 1/Mpc) and z
        values. The array of k values must be indexed as
        k[index_k,index_z,index_mu], the one of z values as
        z[index_z]. This function is useful in the context of
        likelihoods describing spectroscopic galaxy redshift, when
        there is a different k-list for each redhsift and each value
        of the angle mu between the line-of-sight and the measured
        two-point correlation function. The function always returns a
        grid of values for the linear power spectrum, independently of
        what the user sets for 'non_linear'.  This function requires
        that the 'ouput' field contains at least 'mPk'.

        Parameters
        ----------
        k : float array
            Wavenumber array indexed as k[index_k,index_z,index_mu]
        z : float array
            Redshift array indexed as z[index_z]
        k_size : int
            Number of k values for each index_z and index_mu
        z_size : int
            Number of redshift values
        mu_size : int
            Number of k values for each index_k and index_z

        Returns
        -------
        pk_cb : float
            Grid of linear CDM+baryon power spectrum indexed as pk_cb[index_k,index_z,index_mu]
        "

func:      def get_pk_all(self, k, z, nonlinear = True, cdmbar = False, z_axis_in_k_arr = 0, interpolation_kind='cubic'):
doc:   "
        Return different versions of the power spectrum P(k,z) for arrays of k and z with arbitrary shapes

        Return different versions of the power spectrum P_m(k,z) (in Mpc**3)
        for arrays of k values (in 1/Mpc) and z values with arbitrary shape.
        The optional argument nonlinear can be set to True for non-linear (default) or False for non-linear.
        The optional argument cdmbar can be set to False for total matter (default) or True for CDM+baryon. It makes a difference only in presence of massive neutrinos.
        For multi-dimensional k-arrays, the function assumes that one of the dimensions is the z-axis.
        The optional argument z_axis_in_k_arr specifies the integer position of the z_axis within the n-dimensional k array (default:0).
        This function requires that the 'ouput' field contains at least 'mPk'.
        The function returns a grid of values for the power spectrum, with the same shape as the input k grid.

        Parameters
        ----------
        k : float array
            Wavenumber array, arbitrary shape
        z : float array
            Redshift array, arbitrary shape
        nonlinear: bool, optional
            Whether to return the non-linear spectrum instead of the linear one
        cdmbar : bool, optional
            Whether to return the CDM+baryon spectrum instead of the total matter one
        z_axis_in_k_arr : int, optional
            Position of the z_axis within the k array
        interpolation_kind : str, optional
            Flag for the interpolation kind, to be understood by interp1d() function of scipy.interpolate module

        Returns
        -------
        out_pk : float
            Grid of power spectrum with the same shape as input k grid
        "

func:          def _write_pk(z,islinear,ispkcb):
func:          def _islinear(z):
func:          def _interpolate_pk_at_z(karr,z):
func:      def get_pk_and_k_and_z(self, nonlinear=True, only_clustering_species = False, h_units=False):
doc:   "
        Return different versions of the power spectrum P(k,z), together with the corresponding list of k and z values

        Return different versions of the power spectrum P(k,z) (in Mpc**3) together with the corresponding list
        k values (in 1/Mpc) and z values. The values are those defined internally in CLASS before interpolation. This function is useful for building interpolators.
        The optional argument nonlinear can be set to True for non-linear (default) or False for non-linear.
        The optional argument only_clustering_species can be set to False for total matter (default) or True for CDM+baryon. It makes a difference only in presence of massive neutrinos.
        The optional argument h_units, set to False by default, can be set to True to have P(k,z) in (Mpc/h)**3 and k in (h/Mpc).
        This function requires that the 'ouput' field contains at least 'mPk'.
        The function returns a 2D grid of values for the power spectrum.

        Parameters
        ----------
        nonlinear : bool, optional
            Whether to return the non-linear spectrum instead of the linear one
        only_clustering_species : bool, optional
            Whether to return the CDM+baryon spectrum instead of the total matter one
        h_units : bool, optional
            Whether to use Mpc/h and h/Mpc units instead of Mpc and 1/Mpc

        Returns
        -------
        pk : float array
            Grid of power spectrum indexed as pk[index_k, index_z]
        k : float array
            Vector of wavenumbers indexed as k[index_k]
        z : float array
            Vector of redshifts indexed as k[index_z]
        "

func:      def get_transfer_and_k_and_z(self, output_format='class', h_units=False):
doc:   "
        Return a grid of transfer function of various perturbations T_i(k,z) arranged in a dictionary, together with the corresponding list of k and z values

        Return a grid of transfer function of various density and/or velocity perturbations T_i(k,z) arranged in a dictionary, together with the corresponding list of k values (in 1/Mpc) and z values. The values are those defined internally in CLASS before interpolation. This function is useful for building interpolators.
        The optional argument output_format can be set to 'class' (default) or 'camb'. With output_format='class', all transfer functions will be normalised to 'curvature R=1' at initial time and are dimensionless. With output_format='camb', they are normalised to 'curvature R = -1/k^2' like in CAMB, and they have units of Mpc**2. Then, 'dTk' must be in the input: the CAMB format only outputs density transfer functions.
        When sticking to output_format='class', you also get the newtonian metric fluctuations phi and psi.
        If you set the CLASS input parameter 'extra_metric_transfer_functions' to 'yes',
        you get additional metric fluctuations in the synchronous and N-body gauges.
        The optional argument h_units, set to False by default, can be set to True to have k in (h/Mpc).
        This function requires that the 'ouput' field contains at least one of 'dTk' (for density transfer functions) or 'vTk' (for velocity transfer functions). With output_format='camb', 'dTk' must be in the input: the CAMB format only outputs density transfer functions.
        The function returns a dictionary with, for each species, a 2D grid of transfer functions.

        Parameters
        ----------
        output_format : str, optional
            Format transfer functions according to 'class' or 'camb' convention
        h_units : bool, optional
            Whether to use h/Mpc units for k instead of 1/Mpc

        Returns
        -------
        tk : dict
            Dictionary containing all transfer functions.
            For instance, the grid of values of 'd_c' (= delta_cdm) is available in tk['d_c']
            All these grids are indexed as [index_k,index,z], for instance tk['d_c'][index_k,index,z]
        k : float array
            Vector of wavenumbers indexed as k[index_k]
        z : float array
            Vector of redshifts indexed as k[index_z]
        "

func:      def get_Weyl_pk_and_k_and_z(self, nonlinear=False, h_units=False):
doc:   "
        Return the power spectrum of the perturbation [k^2*(phi+psi)/2](k,z), together with the corresponding list of k and z values

        Return the power spectrum of the perturbation [k^2*(phi+psi)/2](k,z), where (phi+psi)/2 is the Weyl potential transfer function, together with the corresponding list of k values (in 1/Mpc) and z values. The values are those defined internally in CLASS before interpolation. This function is useful for building interpolators.
        Note that this function first gets P(k,z) and then corrects the output
        by the ratio of transfer functions [k^2(phi+psi)/d_m]^2.
        The optional argument nonlinear can be set to True for non-linear (default) or False for non-linear.
        The optional argument h_units, set to False by default, can be set to True to have k in (h/Mpc).
        This function requires that the 'ouput' field contains at least one of 'mTk' or 'vTk'.
        The function returns a 2D grid of values for the Weyl potential.

        Parameters
        ----------
        nonlinear : bool, optional
            Whether to return the non-linear spectrum instead of the linear one
        h_units : bool, optional
            Whether to use Mpc/h and h/Mpc units instead of Mpc and 1/Mpc

        Returns
        -------
        Weyl_pk : float array
            Grid of Weyl_pk indexed as Weyl_pk[index_k, index_z]
        k : float array
            Vector of wavenumbers indexed as k[index_k]
        z : float array
            Vector of redshifts indexed as k[index_z]
        "

func:      def sigma(self,R,z, h_units = False):
doc:   "
        Return sigma (total matter) for radius R and redhsift z

        Return sigma, the root mean square (rms) of the relative density fluctuation of
        total matter (dimensionless) in spheres of radius R at redshift z. If R= 8 h/Mpc this will be the usual sigma8(z).
        If the user passes a single value of R and z, the function returns sigma(R,z).
        If the use rpasses an array of R and/or z values, the function sets the shape of the returned grid of sigma(R,z) accordingly.

        If h_units = False (default), R is in unit of Mpc and sigma8 is obtained for R = 8/h.
        If h_units = True, R is in unit of Mpc/h and sigma8 is obtained for R = 8.

        This function requires that the 'ouput' field contains at
        least 'mPk' and that 'P_k_max_h/Mpc' or 'P_k_max_1/Mpc' are
        such that k_max is bigger or equal to 1 h/Mpc.

        Parameters
        ----------
        R : float
            Radius of the spheres in which the rms is computed (single value or array)
        z : float
            Redshift (single value or array)
        h_units : bool, optional
            Whether to use Mpc/h instead of Mpc for R

        Returns
        -------
        sigma : float
            Rms of density fluctuation of total matter (single value or grid of values)
        "

func:      def sigma_cb(self,double R,double z, h_units = False):
doc:   "
        Return sigma_cb (CDM+baryon) for radius R and redhsift z

        Return sigma_cb, the root mean square (rms) of the relative density fluctuation of
        CDM+baryon (dimensionless) in spheres of radius R at redshift z. If R= 8 h/Mpc this will be the usual sigma8_cb(z).
        If the user passes a single value of R and z, the function returns sigma_cb(R,z).
        If the user passes an array of R and/or z values, the function sets the shape of the returned grid of sigma_cb(R,z) accordingly.

        If h_units = False (default), R is in unit of Mpc and sigma8_cb is obtained for R = 8/h.
        If h_units = True, R is in unit of Mpc/h and sigma8_cb is obtained for R = 8.

        This function requires that the 'ouput' field contains at
        least 'mPk' and that 'P_k_max_h/Mpc' or 'P_k_max_1/Mpc' are
        such that k_max is bigger or equal to 1 h/Mpc.

        Parameters
        ----------
        R : float
            Radius of the spheres in which the rms is computed (single value or array)
        z : float
            Redshift (single value or array)
        h_units : bool, optional
            Whether to use Mpc/h instead of Mpc for R

        Returns
        -------
        sigma_cb : float
            Rms of density fluctuation of CDM+baryon (single value or grid of values)
        "

func:      def pk_tilt(self,double k,double z):
doc:   "
        Return the logarithmic slope of the linear matter power spectrum at k and z

        Return the logarithmic slope of the linear matter power
        spectrum, d ln P / d ln k (dimensionless), at a given
        wavenumber k (units of 1/Mpc) and redshift z.

        This function requires that the 'ouput' field contains at
        least 'mPk'.

        Parameters
        ----------
        k : float
            Wavenumber
        z : float
            Redshift

        Returns
        -------
        pk_tilt : float
            Logarithmic slope
        "

func:      def age(self):
doc:   "
        Return the age of the Universe

        Return the age of the universe in proper time (units of Gyr)

        Parameters
        ----------
        None

        Returns
        -------
        age : float
            Age
        "

func:      def h(self):
doc:   "
        Return the reduced Hubble paramneter

        Return the reduced Hubble paramneter h (dimensionless) such that H0 = 100*h km/s/Mpc

        Parameters
        ----------
        None

        Returns
        -------
        h : float
            Reduced Hubble paraneter
        "

func:      def n_s(self):
doc:   "
        Return the scalar tilt

        Return the scalar tilt n_s (dimensionless) of the primordial scalar spectrum

        Parameters
        ----------
        None

        Returns
        -------
        n_s : float
            Scalar tilt
        "

func:      def tau_reio(self):
doc:   "
        Return the reionization optical depth

        Return the reionization optical depth tau_reio (dimensionless)

        Parameters
        ----------
        None

        Returns
        -------
        tau_reio : float
            Reionization optical depth
        "

func:      def Omega_m(self):
doc:   "
        Return the Omega of total matter

        Return the fractional density Omega (dimensionless) of total non-relativistic matter (baryons, dark matter, massive neutrinos...) evaluated today

        Parameters
        ----------
        None

        Returns
        -------
        Omega_m : float
            Omega of total matter
        "

func:      def Omega_r(self):
doc:   "
        Return the Omega of total radiation

        Return the fractional density Omega (dimensionless) of total radiation, that is, ultra-relativistic matter (photons, massless neutrinos, ...) evaluated today

        Parameters
        ----------
        None

        Returns
        -------
        Omega_r : float
            Omega of total radiation
        "

func:      def theta_s_100(self):
doc:   "
        Sound horizon angle at recombination (peak of visibility) multiplied by 100

        Return the angular scale of the sound horizon at recombination
        multiplied by 100 (in radian, that is, dimensionless). The
        function uses theta_s = d_s(z_rec)/d_a(t_rec) =
        r_s(z_rec)/r_a(z_rec), where the d are physical distances and
        the r are comoving distances. CLASS defines recombination as
        the time at which the photon visibility function reaches its maximum.

        Parameters
        ----------
        None

        Returns
        -------
        theta_s_100 : float
            Sound horizon angle at recombination multiplied by 100
        "

func:      def theta_star_100(self):
doc:   "
        Sound horizon angle at photon decoupling (tau=1) multiplied by 100

        Return the angular scale of the sound horizon at photon
        decoupling multiplied by 100 (in radian, that is,
        dimensionless). The function uses theta_s =
        d_s(z_star)/d_a(t_star) = r_s(z_star)/r_a(z_star), where the d
        are physical distances and the r are comoving distances. CLASS
        defines decoupling as the time at which the photon optical
        depth crosses one.

        Parameters
        ----------
        None

        Returns
        -------
        theta_star_100 : float
            Sound horizon angle at decoupling multiplied by 100
        "

func:      def Omega_Lambda(self):
doc:   "
        Return the Omega of the comsological constant

        Return the fractional density Omega (dimensionless) of the cosmological constant

        Parameters
        ----------
        None

        Returns
        -------
        Omega_Lambda : float
            Omega of cosmological constant
        "

func:      def Omega_g(self):
doc:   "
        Return the Omega of photons

        Return the fractional density Omega (dimensionless) of photons  evaluated today

        Parameters
        ----------
        None

        Returns
        -------
        Omega_g : float
            Omega of photons
        "

func:      def Omega_b(self):
doc:   "
        Return the Omega of baryons

        Return the fractional density Omega (dimensionless) of baryons evaluated today

        Parameters
        ----------
        None

        Returns
        -------
        Omega_b : float
            Omega of baryons
        "

func:      def omega_b(self):
doc:   "
        Return the Omega h^2 of baryons

        Return the density parameter omega = Omega h^2 (dimensionless) of baryons

        Parameters
        ----------
        None

        Returns
        -------
        omega_b : float
            Omega h^2 of baryons
        "

func:      def Neff(self):
doc:   "
        Return the effective neutrino number

        Return the effective neutrino number N_eff (dimensionless) which parametrizes the density of radiation in the early universe, before non-cold dark matter particles become non-relativistic. Should be 3.044 in the standard model.

        Parameters
        ----------
        None

        Returns
        -------
        Neff : float
            Effective neutrino number
        "

func:      def k_eq(self):
doc:   "
        Return the equality wavenumber

        Return the wavenumber such that k=aH at the time of equality between radiation and matter (units of 1/Mpc)

        Parameters
        ----------
        None

        Returns
        -------
        k_eq : float
            Equality wavenumber
        "

func:      def z_eq(self):
doc:   "
        Return the equality redshift

        Return the redshift of the time of equality between radiation and matter (dimensionless)

        Parameters
        ----------
        None

        Returns
        -------
        z_eq : float
            Equality redshift
        "

func:      def sigma8(self):
doc:   "
        Return sigma8 today

        Return sigma8 (dimensionless), the root mean square (rms) of the relative density fluctuation of
        total matter in spheres of radius R= 8 h/Mpc at z=0

        This function requires that the 'ouput' field contains at least 'mPk'.

        Parameters
        ----------
        None

        Returns
        -------
        sigma8 : float
            sigma8 today
        "

func:      def S8(self):
doc:   "
        Return Sigma8

        Return Sigma8 = sigma8*(Omega_m/0.3)^0.5 (dimensionless), the root mean square (rms) of the relative density fluctuation of
        total matter in spheres of radius R= 8 h/Mpc at z=0 multiplied by (Omega_m/0.3)^0.5

        Parameters
        ----------
        None

        Returns
        -------
        Sigma8 : float
            Sigma8 today
        "

func:      def sigma8_cb(self):
doc:   "
        Return sigma8_cb today

        Return sigma8_cb (dimensionless), the root mean square (rms) of the relative density fluctuation of
        CDM+baryons in spheres of radius R= 8 h/Mpc at z=0

        This function requires that the 'ouput' field contains at least 'mPk'.

        Parameters
        ----------
        None

        Returns
        -------
        sigma8_cb : float
            sigma8_cb today
        "

func:      def rs_drag(self):
doc:   "
        Return the comoving sound horizon at baryon drag

        Return the comoving sound horizon (units of Mpc) at baryon
        drag time. CLASS defines this time as the time when the baryon
        optical depth crosses 1. This baryon optical depth is obtained
        by integrating the Thomson scattering rate of baryons, that
        is, the Thomson scattering rate of photons rescaled by 1/R,
        where R = 3 rho_b / 4 rho_gamma.

        Parameters
        ----------
        None

        Returns
        -------
        rs_drag : float
            Comoving sound horizon at baryon drag
        "

func:      def z_reio(self):
doc:   "
        Return the reionization redshift

        Return the reionization redshift (dimensionless). This
        redshift is a free parameter in the analytic form assummed for
        the free electron fraction x_e(z). It is either passed by the
        user, or adjusted by CLASS using a shooting method in order to
        match a required value of the reionization optical depth.

        Parameters
        ----------
        None

        Returns
        -------
        z_reio : float
            Reionization redshift
        "

func:      def angular_distance(self, z):
doc:   "
        Return the angular diameter distance at z

        Return the angular diameter distance (units of Mpc) at redshift z.
        If the user passes a single value of z, the function returns angular_distance(z).
        If the user passes an array of z values, the function sets the shape of angular_distance(z) accordingly.

        Parameters
        ----------
        z : float
            Redshift (single value or array)

        Returns
        -------
        angular_distance : float
            Angular distance (single value or array)
        "

func:      def angular_distance_from_to(self, z1, z2):
doc:   "
        Return the angular diameter distance of object at z2 as seen by observer at z1

        Return the angular diameter distance (units of Mpc) of object
        at z2 as seen by observer at z1, that is,
        sin_K((chi2-chi1)*np.sqrt(|k|))/np.sqrt(|k|)/(1+z2).  If
        z1>z2, returns zero.

        Parameters
        ----------
        z1 : float
            Observer redshift
        z2 : float
            Source redshift

        Returns
        -------
        angular_distance_from_to : float
            Angular distance
        "

func:      def comoving_distance(self, z):
doc:   "
        Return the comoving distance at z

        Return the comoving distance (units of Mpc) at redshift z.
        If the user passes a single value of z, the function returns comoving_distance(z).
        If the user passes an array of z values, the function sets the shape of comoving_distance(z) accordingly.

        Parameters
        ----------
        z : float
            Redshift (single value or array)

        Returns
        -------
        r : float
            Comoving distance (single value or array)
        "

func:      def scale_independent_growth_factor(self, z):
doc:   "
        Return the scale-independent growth factor at z

        Return the scale-independent growth factor D(z) for CDM
        perturbations (dimensionless).  CLASS finds this quantity by
        solving the background equation D'' + a H D' + 3/2 a^2 \rho_M
        D = 0, where ' stands for a derivative w.r.t conformal time. D
        is normalized to 1 today. By default, in this equation, rho_M
        is obtained by summing over baryons and cold dark matter
        (possibly including interacting CDM), but not over
        rho_ncdm. This means that in models with massive neutrinos D
        is a approximation for the growth factor in the small-scale
        (large-k) limit (the limit in which neutrinos free stream).

        Parameters
        ----------
        z : float
            Desired redshift

        Returns
        -------
        D : float
            Scale-independent growth factor D(z)
        "

func:      def scale_independent_growth_factor_f(self, z):
doc:   "
        Return the scale-independent growth rate at z

        Return the scale-independent growth rate f(z) = (d ln D / d ln
        a) for CDM perturbations (dimensionless). CLASS finds D by
        solving the background equation D'' + a H D' + 3/2 a^2 \rho_M
        D = 0, where ' stands for a derivative w.r.t conformal
        time. By default, in this equation, rho_M is obtained by
        summing over baryons and cold dark matter (possibly including
        interacting CDM), but not over rho_ncdm. This means that in
        models with massive neutrinos f is a approximation for the
        growth rate in the small-scale (large-k) limit (the limit in
        which neutrinos free stream).

        Parameters
        ----------
        z : float
            Desired redshift

        Returns
        -------
        f : float
            Scale-independent growth rate f(z)
        "

func:      def scale_dependent_growth_factor_f(self, k, z, h_units=False, nonlinear=False):
doc:   "
        Return the scale-dependent growth rate of total matter at z

        Return the scale-dependent growth rate for total matter perturbations (dimensionless):

        f(k,z) = (d ln delta_m / d ln a)

        The growth rate is actually inferred from the matter power
        spectrum rather than from the delta_m perturbation, as:

        f(k,z)= 1/2 * [d ln P_m(k,a) / d ln a] = - 0.5 * (1+z) * [d ln P_m(k,z) / d z]

        where P_m(k,z) is the total matter power spectrum. This is
        evaluated using a numerical derivative computed with the
        function UnivariateSpline of the module scipy.interpolate
        (with option s=0 in UnivariateSpline).

        If h_units = False (default), k is in unit of 1/Mpc.
        If h_units = True, k is in unit of h/Mpc.

        If nonlinear=False (default), estimate f from the linear power spectrum.
        If nonlinear=True, estimate f from the non-linear power spectrum.

        This function requires that the 'ouput' field contains at
        least 'mPk'. It also requires that the user asks for P(k,z) at redshifts
        larger than 0, or sets 'z_max_pk' to a non-zero value.

        Parameters
        ----------
        k : float
            Wavenumber
        z : float
            Redshift
        h_units : bool, optional
            Whether to pass k in units of h/Mpc instead of 1/Mpc
        nonlinear : bool, optional
            Whether to compute f from the non-linear power spectrum instead of the linear one

        Returns
        -------
        f : float
            Scale-dependent growth rate f(k,z)
        "

func:      def scale_dependent_growth_factor_f_cb(self, k, z, h_units=False, nonlinear=False):
doc:   "
        Return the scale-dependent growth rate of CDM+baryon at z

        Return the scale-dependent growth rate for CDM+baryon perturbations (dimensionless):

        f(k,z) = (d ln delta_cb / d ln a)

        The growth rate is actually inferred from the matter power
        spectrum rather than from the delta_cb perturbation, as:

        f(k,z)= 1/2 * [d ln P_cb(k,a) / d ln a] = - 0.5 * (1+z) * [d ln P_cb(k,z) / d z]

        where P_cb(k,z) is the CDM+baryon power spectrum. This is
        evaluated using a numerical derivative computed with the
        function UnivariateSpline of the module scipy.interpolate
        (with option s=0 in UnivariateSpline).

        If h_units = False (default), k is in unit of 1/Mpc.
        If h_units = True, k is in unit of h/Mpc.

        If nonlinear=False (default), estimate f from the linear power spectrum.
        If nonlinear=True, estimate f from the non-linear power spectrum.

        This function requires that the 'ouput' field contains at
        least 'mPk'. It also requires that the user asks for P(k,z) at redshifts
        larger than 0, or sets 'z_max_pk' to a non-zero value.

        Parameters
        ----------
        k : float
            Wavenumber
        z : float
            Redshift
        h_units : bool, optional
            Whether to pass k in units of h/Mpc instead of 1/Mpc
        nonlinear : bool, optional
            Whether to compute f from the non-linear power spectrum instead of the linear one

        Returns
        -------
        f : float
            Scale-dependent growth rate f_cb(k,z)
        "

func:      def scale_dependent_growth_factor_D(self, k, z, h_units=False, nonlinear=False):
doc:   "
        Return the scale-dependent growth factor of total matter at z

        Return the scale-dependent growth factor D for total matter perturbations (dimensionless).

        The growth factor is actually inferred from the matter power
        spectrum rather than from the delta_m perturbation, as:

        D(k,z) = sqrt[ P_m(k,z)/P_m(k,0) ]

        where P_m(k,z) is the total matter power spectrum.

        If h_units = False (default), k is in unit of 1/Mpc.
        If h_units = True, k is in unit of h/Mpc.

        If nonlinear=False (default), estimate f from the linear power spectrum.
        If nonlinear=True, estimate f from the non-linear power spectrum.

        This function requires that the 'ouput' field contains at least 'mPk'.

        Parameters
        ----------
        k : float
            Wavenumber
        z : float
            Redshift
        h_units : bool, optional
            Whether to pass k in units of h/Mpc instead of 1/Mpc
        nonlinear : bool, optional
            Whether to compute f from the non-linear power spectrum instead of the linear one

        Returns
        -------
        D : float
            Scale-dependent growth factor D(k,z)
        "

func:      def scale_dependent_growth_factor_D_cb(self, k, z, h_units=False, nonlinear=False):
doc:   "
        Return the scale-dependent growth factor of CDM+baryon at z

        Return the scale-dependent growth factor D_cb for CDM+baryon perturbations (dimensionless).

        The growth factor is actually inferred from the matter power
        spectrum rather than from the delta_m perturbation, as:

        D_cb(k,z) = sqrt[ P_cb(k,z)/P_cb(k,0) ]

        where P_cb(k,z) is the total matter power spectrum.

        If h_units = False (default), k is in unit of 1/Mpc.
        If h_units = True, k is in unit of h/Mpc.

        If nonlinear=False (default), estimate f from the linear power spectrum.
        If nonlinear=True, estimate f from the non-linear power spectrum.

        This function requires that the 'ouput' field contains at least 'mPk'.

        Parameters
        ----------
        k : float
            Wavenumber
        z : float
            Redshift
        h_units : bool, optional
            Whether to pass k in units of h/Mpc instead of 1/Mpc
        nonlinear : bool, optional
            Whether to compute f from the non-linear power spectrum instead of the linear one

        Returns
        -------
        D_cb : float
            Scale-dependent growth factor D_cb(k,z)
        "

func:      def scale_independent_f_sigma8(self, z):
doc:   "
        Return sigma8 * f(z)

        Return the scale-independent growth rate f(z) multiplied by
        sigma8(z) (dimensionless). The scale-independent growth rate
        f(z) is inferred from the scale-independent growth factor
        D(z), itself obtained by solving a background equation.

        Parameters
        ----------
        z : float
            Redshift (single value or array)

        Returns
        -------
        scale_independent_f_sigma8 : float
            f(z)*sigma8(z) (single value or array)
        "

func:      def effective_f_sigma8(self, z, z_step=0.1):
doc:   "
        Return sigma8(z) * f(k,z) estimated near k = 8 h/Mpc

        Return the growth rate f(k,z) estimated near k = 8 h/Mpc and multiplied by
        sigma8(z) (dimensionless). The product is actually directly inferred from sigma(R,z) as:

        sigma8(z) * f(k=8h/Mpc,z) = d sigma8 / d ln a = - (d sigma8 / dz)*(1+z)

        In the general case the quantity (d sigma8 / dz) is inferred
        from the two-sided finite difference between z+z_step and
        z-z_step. For z < z_step the step is reduced progressively
        down to z_step/10 while sticking to a double-sided
        derivative. For z< z_step/10 a single-sided derivative is used
        instead. (default: z_step=0.1)

        The input z can be a single value or an array of values. The
        output has the same shape.

        Parameters
        ----------
        z : float
            Redshift (single value or array)
        z_step : float, optional
            Step in redshift space for the finite difference method

        Returns
        -------
        effective_f_sigma8 : float
            sigma8(z) * f(k=8 h/Mpc,z) (single value or array)
        "

func:      def effective_f_sigma8_spline(self, z, Nz=20):
doc:   "
        Return sigma8(z) * f(k,z) estimated near k = 8 h/Mpc

        Return the growth rate f(k,z) estimated near k = 8 h/Mpc and multiplied by
        sigma8(z) (dimensionless). The product is actually directly inferred from sigma(R,z) as:

        sigma8(z) * f(k=8h/Mpc,z) = d sigma8 / d ln a = - (d sigma8 / dz)*(1+z)

        The derivative is inferred from a cubic spline method with Nz
        points, using the CubicSpline().derivative() method of the
        scipy.interpolate module (Nz=20 by default).

        The input z can be a single value or an array of values. The
        output has the same shape.

        Parameters
        ----------
        z : float
            Redshift (single value or array)
        Nz : float, optional
            Number of samples for the spline method

        Returns
        -------
        effective_f_sigma8 : float
            sigma8(z) * f(k=8 h/Mpc,z) (single value or array)
        "

func:      def z_of_tau(self, tau):
doc:   "
        Return the redshift corresponding to a given conformal time.

        Return the redshift z (units of Mpc) corresponding to a given conformal time tau (units of Mpc).
        The user can pass a single tau or an array of tau. The shape of the output z is set accordingly.

        Parameters
        ----------
        tau : float
            Conformal time (single value or array)

        Returns
        -------
        z : float
            Redshift (single value or array)
        "

func:      def Hubble(self, z):
doc:   "
        Return Hubble(z)

        Return the Hubble rate H(z) - more precisely, the inverse Hubble radius H(z)/c (units 1/Mpc).
        The user can pass a single z or an array of z. The shape of the output z is set accordingly.

        Parameters
        ----------
        z : float
            Redshift single value or array)

        Returns
        -------
        H : float
            Hubble rate (single value or array)
        "

func:      def Om_m(self, z):
doc:   "
        Return the fractional density Omega of matter at z

        Return the fractional density Omega(z) of total matter at redshift z (dimensionless).
        The user can pass a single z or an array of z. The shape of the output is set accordingly.

        Parameters
        ----------
        z : float
            Redshift single value or array)

        Returns
        -------
        Om_m : float
            Omega_m(z) (single value or array)
        "

func:      def Om_b(self, z):
doc:   "
        Return the fractional density Omega of baryons at z

        Return the fractional density Omega(z) of baryons at redshift z (dimensionless).
        The user can pass a single z or an array of z. The shape of the output is set accordingly.

        Parameters
        ----------
        z : float
            Redshift (single value or array)

        Returns
        -------
        Om_b : float
            Omega_b(z) (single value or array)
        "

func:      def Om_cdm(self, z):
doc:   "
        Return the fractional density Omega of CDM at z

        Return the fractional density Omega(z) of CDM at redshift z (dimensionless).
        The user can pass a single z or an array of z. The shape of the output is set accordingly.

        Parameters
        ----------
        z : float
            Redshift (single value or array)

        Returns
        -------
        Om_cdm : float
            Omega_cdm(z) (single value or array)
        "

func:      def Om_ncdm(self, z):
doc:   "
        Return the fractional density Omega of non-cold dark matter at z

        Return the fractional density Omega(z) of non-cold dark matter at redshift z (dimensionless).
        The user can pass a single z or an array of z. The shape of the output is set accordingly.

        Parameters
        ----------
        z : float
            Redshift (single value or array)

        Returns
        -------
        Om_ncdm : float
            Omega_ncdm(z) (single value or array)
        "

func:      def ionization_fraction(self, z):
doc:   "
        Return the electron ionization fraction x_e(z)

        Return the electron ionization fraction x_e (dimensionless)
        for a given redshift z. CLASS sticks to the standard
        definition x_e = n_free_electrons / n_H, such that x_e can be
        bigger than one due to Helium.
        The user can pass a single z or an array of z. The shape of the output is set accordingly.

        Parameters
        ----------
        z : float
            Redshift (single value or array)

        Returns
        -------
        xe : float
            Electron ionization fraction (single value or array)
        "

func:      def baryon_temperature(self, z):
doc:   "
        Return the baryon temperature T_b(z)

        Return the baryon temperature T_b (units of K) for a given redshift z.
        The user can pass a single z or an array of z. The shape of the output is set accordingly.

        Parameters
        ----------
        z : float
            Redshift (single value or array)

        Returns
        -------
        Tb : float
            Baryon temperature (single value or array)
        "

func:      def T_cmb(self):
doc:   "
        Return the photon temperature today

        Return the photon temperature T_cmb (units of K) evaluated at z=0

        Parameters
        ----------
        None

        Returns
        -------
        T_cmb : float
            Photon temperature today
        "

func:      def Omega0_m(self):
doc:   "
        Return the Omega of total matter

        Return the fractional density Omega (dimensionless) of total
        non-relativistic matter (baryons, dark matter, massive
        neutrinos...) evaluated today. Strictly identical to the
        previously defined function Omega_m(), but we leave it not to
        break compatibility.

        Parameters
        ----------
        None

        Returns
        -------
        Omega0_m : float
            Omega of total matter
        "

func:      def get_background(self):
doc:   "
        Return all background quantities

        Return a dictionary of background quantities at all times.
        The name and list of quantities in the returned dictionary are
        defined in CLASS, in background_output_titles() and
        background_output_data(). The keys of the dictionary refer to
        redshift 'z', proper time 'proper time [Gyr]', conformal time
        'conf. time [Mpc]', and many quantities such as the Hubble
        rate, distances, densities, pressures, or growth factors. For
        each key, the dictionary contains an array of values
        corresponding to each sampled value of time.

        This function works for whatever request in the 'output'
        field, and even if 'output' was not passed or left blank.

        Parameters
        ----------
        None

        Returns
        -------
        background : dict
            Dictionary of all background quantities at each time
        "

func:      def get_thermodynamics(self):
doc:   "
        Return all thermodynamics quantities

        Return a dictionary of thermodynbamics quantities at all
        times.  The name and list of quantities in the returned
        dictionary are defined in CLASS, in
        thermodynamics_output_titles() and
        thermodynamics_output_data(). The keys of the dictionary
        refer to the scale factor 'scale factor a', redshift 'z',
        conformal time 'conf. time [Mpc]', and many quantities such as
        the electron ionization fraction, scattering rates,
        temperatures, etc. For each key, the dictionary contains an
        array of values corresponding to each sampled value of time.

        This function works for whatever request in the 'output'
        field, and even if 'output' was not passed or left blank.

        Parameters
        ----------
        None

        Returns
        -------
        thermodynamics : dict
            Dictionary of all thermodynamics quantities at each time
        "

func:      def get_primordial(self):
doc:   "
        Return primordial spectra

        Return the primordial scalar spectrum, and possibly the
        primordial tensor spectrum if 'modes' includes 't'.  The name
        and list of quantities in the returned dictionary are defined
        in CLASS, in primordial_output_titles() and
        primordial_output_data().  The keys are 'k [1/Mpc]',
        'P_scalar(k)' and possibly 'P_tensor(k)'. For each key, the
        dictionary contains an array of values corresponding to each
        sampled wavenumber.

        This function requires that 'output' is set to something, e.g. 'tCl' or 'mPk'.

        Parameters
        ----------
        None

        Returns
        -------
        primordial : dict
            Dictionary of primordial spectra at each wavenumber
        "

func:      def get_perturbations(self, return_copy=True):
doc:   "
        Return transfer function evolution for selected wavenumbers

        Return an array of dictionaries of transfer functions at all
        times (dimensionless). There is one dictionary for each
        requested mode in the list 'scalars', 'vectors', tensors', and
        for each wavenumbers requested by the user with the input
        parameter 'k_output_values' (units of 1/Mpc). For instance,
        perturbations['scalars'][0] could be one of the dictionaries.

        For each requested mode and wavenumber, the name and list of
        quantities in the returned dictionary are defined in CLASS, in
        perturbations_output_titles() and perturbations_output_data(),
        sticking to the 'class' format. The keys of the dictionary
        refer to wavnumbers 'k (h/Mpc)', density fluctuations 'd_g',
        etc., velocity perturbations 't_g', etc., and metric
        perturbations. For each key, the dictionary contains an array
        of values corresponding to each sampled value of time. An
        example of such array would be:

        perturbations['scalars'][0]['d_g'].

        Do not enable 'return_copy=False' unless you know exactly what
        you are doing.  This will mean that you get access to the
        direct C pointers inside CLASS.  That also means that if CLASS
        is deallocated, your perturbations array will become invalid.

        This function requires setting 'output' to somethings, for instance 'tCl' or 'mPk'.

        Parameters
        ----------
        return_copy : bool, optional
            Whether to return an exact copy of a memory area defined by C pointers inside CLASS

        Returns
        -------
        perturbations : dict
            Array of dictionaries of all transfer functions at each time
        "

func:      def get_transfer(self, z=0., output_format='class'):
doc:   "
        Return all scalar transfer functions at z

        Return a dictionary of transfer functions at all wavenumbers
        for scalar perturbations and at redshift z.  The name and list
        of quantities in the returned dictionary are defined in CLASS,
        in perturbations_output_titles() and
        perturbations_output_data(), using either the 'class' format
        (default) or 'camb' format. The keys of the dictionary refer
        to wavenumbers 'k (h/Mpc)', and in 'class' format, to density
        fluctuations 'd_g', etc., velocity perturbations 't_g', etc.,
        and metric perturbations. In 'camb' format, besides
        wavenumbers 'k (h/Mpc)', the returned transfer functions are
        '-T_g/k2', etc.  For each key, the dictionary contains an
        array of values corresponding to each sampled value of
        wavenumber.

        This function works if 'output' contains at least 'mTk' for
        density transfer function and/or 'vTk' for velocity transfer
        functions. To get the transfer functions at some z>0, the user
        must set 'z_pk' or 'z_max_pk' to a value equal or bigger to
        the requested one.

        Parameters
        ----------
        z : float, optional
            Redshift (0 by default)

        output_format : str, optional
            Output format, one of 'class' (defualt) or 'camb'

        Returns
        -------
        transfers : dict
            Dictionary of all transfer functions at each wavenumber
        "

func:      def get_current_derived_parameters(self, names):
doc:   "
        Return a dictionary of numbers computed internally by CLASS

        Return a dictionary containing an entry for each of the names defined in the input list 'names'. These names have to match the list defined inside this function, that is, belong to the list: 'h', 'H0' (units of km/s/Mpc), 'Omega0_lambda', 'Omega_Lambda', 'Omega0_fld', 'age' (units of Gyr), 'conformal_age' (units of Mpc), 'm_ncdm_in_eV' (units of eV), 'm_ncdm_tot' (units of eV), 'Neff', 'Omega_m', 'omega_m', 'xi_idr', 'N_dg', 'Gamma_0_nadm', 'a_dark', 'tau_reio', 'z_reio', 'z_rec', 'tau_rec' (units of Mpc), 'rs_rec' (units of Mpc), 'rs_rec_h' (units of Mpc/h), 'ds_rec' (units of Mpc), 'ds_rec_h' (units of Mpc/h), 'ra_rec' (units of Mpc), 'ra_rec_h' (units of Mpc/h), 'da_rec'(units of Mpc), 'da_rec_h' (units of Mpc/h), 'z_star', 'tau_star' (units of Mpc), 'rs_star' (units of Mpc), 'ds_star' (units of Mpc), 'ra_star' (units of Mpc), 'da_star' (units of Mpc), 'rd_star' (units of Mpc), 'z_d', 'tau_d' (units of Mpc), 'ds_d' (units of Mpc), 'ds_d_h' (units of Mpc/h), 'rs_d' (units of Mpc), 'rs_d_h' (units of Mpc/h), 'conf_time_reio' (units of Mpc), '100*theta_s', '100*theta_star', 'theta_s_100', 'theta_star_100', 'YHe', 'n_e', 'A_s', 'ln10^{10}A_s', 'ln_A_s_1e10', 'n_s', 'alpha_s', 'beta_s', 'r', 'r_0002', 'n_t', 'alpha_t', 'V_0', 'V_1', 'V_2', 'V_3', 'V_4', 'epsilon_V', 'eta_V', 'ksi_V^2', 'exp_m_2_tau_As', 'phi_min', 'phi_max', 'sigma8', 'sigma8_cb', 'k_eq' (units of 1/Mpc), 'a_eq', 'z_eq', 'H_eq' (units of 1/Mpc), 'tau_eq' (units of Mpc), 'g_sd', 'y_sd', 'mu_sd'.

        For instance, to get the age of the universe in Gyr and the Hubble parameter in km/s/Mpc, the user may call .get_current_derived_parameters(['age','H0']), store the result in a dictionary 'derived', and retrieve these values as age=derived['age'] and H0=derived['H0'].

        Parameters
        ----------
        names : str list
            Names defined inside this function, associated to quantities computed internally by CLASS

        Returns
        -------
        derived : dict
            Dictionary whose keys are the input names, and whose values are numbers extracted from the CLASS structures
        "

func:      def nonlinear_scale(self, np.ndarray[DTYPE_t,ndim=1] z, int z_size):
doc:   "
        Return the wavenumber associated to the nonlinearity scale for an array of redshifts

        Return the wavenumber k_nl(z) associated to the nonlinearity
        scale for an array of z_size redshifts. The nonlinearity
        scale is defined and computed within the Halofit or HMcode
        external modules.

        Parameters
        ----------
        z : float array
            Array of requested redshifts
        z_size : int
            Size of the redshift array

        Returns
        -------
        k_nl : float array
            Array of k_nl (z)
        "

func:      def nonlinear_scale_cb(self, np.ndarray[DTYPE_t,ndim=1] z, int z_size):
doc:   "
        Return the wavenumber associated to the nonlinearity scale (computed using the CDM+baryon spectrum) for an array of redshifts

        Return the wavenumber k_nl_cb(z) associated to the
        nonlinearity scale (computed using the CDM+baryon spectrum)
        for an array of z_size redshifts. The nonlinearity scale is
        defined and computed within the Halofit or HMcode external
        modules.

        Parameters
        ----------
        z : float array
            Array of requested redshifts
        z_size : int
            Size of the redshift array

        Returns
        -------
        k_nl_cb  : float array
            Array of k_nl_cb(z)
        "

func:      def __call__(self, ctx):
doc:   "
        Function to interface with CosmoHammer

        Parameters
        ----------
        ctx : context
                Contains several dictionaries storing data and cosmological
                information

        "

func:      def get_pk_array(self, np.ndarray[DTYPE_t,ndim=1] k, np.ndarray[DTYPE_t,ndim=1] z, int k_size, int z_size, nonlinear):
doc:   "
        Return the total matter power spectrum P_m(k,z) computed with a fast method on a k and z array

        Return the total matter power spectrum P_m(k,z) (in Mpc**3)
        for an array of k values (in 1/Mpc) and z values, computed
        using a faster algorithm that with .get_pk(). The output is a
        flattened array index as pk[index_z*k_size+index_k].

        This function requires that the 'ouput' field contains at
        least 'mPk'.

        If 'non linear' is set to a non-linear method in CLASS, this
        function returns either the linear or non-linear power
        spectrum, depending on the argument 'nonlinear'. If 'non
        linear' is not set in CLASS, the argument 'nonlinear' should
        be set to 'False'.

        Parameters
        ----------
        k : float array
            Wavenumber array (one-dimensional, size k_size)
        z : float array
            Redshift array (one-dimensional, size z_size)
        k_size : int
            Number of k values
        z_size : int
            Number of redshift values
        nonlinear : bool
            Whether to return the nonlinear or linear power spectrum

        Returns
        -------
        pk : float array
            Flattened array of total matter power spectrum indexed as pk[index_z*k_size+index_k]
        "

func:      def get_pk_cb_array(self, np.ndarray[DTYPE_t,ndim=1] k, np.ndarray[DTYPE_t,ndim=1] z, int k_size, int z_size, nonlinear):
doc:   "
        Return the CDM+baryon power spectrum P_cb(k,z) computed with a fast method on a k and z array

        Return the CDM+baryon power spectrum P_cb(k,z) (in Mpc**3)
        for an array of k values (in 1/Mpc) and z values, computed
        using a faster algorithm that with .get_pk_cb(). The output is a
        flattened array index as pk_cb[index_z*k_size+index_k].

        This function requires that the 'ouput' field contains at
        least 'mPk'.

        If 'non linear' is set to a non-linear method in CLASS, this
        function returns either the linear or non-linear power
        spectrum, depending on the argument 'nonlinear'. If 'non
        linear' is not set in CLASS, the argument 'nonlinear' should
        be set to 'False'.

        Parameters
        ----------
        k : float array
            Wavenumber array (one-dimensional, size k_size)
        z : float array
            Redshift array (one-dimensional, size z_size)
        k_size : int
            Number of k values
        z_size : int
            Number of redshift values
        nonlinear : bool
            Whether to return the nonlinear or linear power spectrum

        Returns
        -------
        pk_cb : float array
            Flattened array of CDM+baryon power spectrum indexed as pk_cb[index_z*k_size+index_k]
        "

func:      def Omega0_k(self):
doc:   "
        Return the Omega of curvature

        Return the effective fractional density Omega (dimensionless) of curvature evaluated today.

        Parameters
        ----------
        None

        Returns
        -------
        Omega0_k : float
            Omega of curvature
        "

func:      def Omega0_cdm(self):
doc:   "
        Return the Omega of cdm

        Return the fractional density Omega (dimensionless) of CDM
        evaluated today. Strictly identical to the previously defined
        function Omega_cdm(), but we leave it not to break
        compatibility.

        Parameters
        ----------
        None

        Returns
        -------
        Omega0_cdm : float
            Omega of CDM
        "

func:      def spectral_distortion_amplitudes(self):
doc:   "
        Return the spectral distortion amplitudes g, mu, y, etc.

        Return the spectral distortion amplitude for each distorsion
        type g, mu, y, and residuals. The number of outputs beyond (g,
        mu, y) is the number of PCA components requested by the user
        using the input parameteer 'sd_PCA_size'

        This function requires that the 'ouput' field contains at least 'Sd'.

        Parameters
        ----------
        None

        Returns
        -------
        sd_type_amps : float array
            Spectral distortion amplitudes
        "

func:      def spectral_distortion(self):
doc:   "
        Return the shape of the total spectral distortion as a function of frequency

        Return the shape of the total spectral distortion (units of
        10^26 W m^-2 Hz^-1 sr^-1) as a function of frequency (units of
        GHz).

        This function requires that the 'ouput' field contains at least 'Sd'.

        Parameters
        ----------
        None

        Returns
        -------
        sd_nu : float array
            Array of frequencies nu/[1 GHz] (units of GHz)

        sd_amp : float array
            Array of total spectral distortion at these frequencies (units of 10^26 W m^-2 Hz^-1 sr^-1)
        "

func:      def get_sources(self):
doc:   "
        Return the source functions stored in the perturbation module at all sampled (k, tau)

        After integrating all perturbations (more precisely, all
        transfer functions) over time, the CLASS perturbation module
        stores a list of the most important combinations of them on a
        grid of (k,tau) values. This includes the combinations used
        later to compute the CMB and LSS observables. This function
        retrieves these source functions directly from the CLASS
        structure 'perturbations', and returns them together with the
        values of k and tau at which they are sampled.

        The output is a dictionary whose keys belong to the list:
        -'p' for the CMB polarization source
        -'phi' for the metric fluctuation phi of the Newtonian gauge
        -'phi_plus_psi' for the metric fluctuation phi+psi of the Newtonian gauge
        -'phi_prime' for the metric fluctuation phi' of the Newtonian gauge
        -'psi' for the metric fluctuation psi of the Newtonian gauge
        -'H_T_Nb_prime' for the metric fluctuation H_T' of the Nboisson gauge
        -'k2gamma_Nb' for the k^2 gamma of the Nboisson gauge
        -'h' for the metric fluctuation h of the synchronouys gauge
        -'h_prime'  for the metric fluctuation h' of the synchronouys gauge
        -'eta' for the metric fluctuation eta of the synchronouys gauge
        -'eta_prime'  for the metric fluctuation eta' of the synchronouys gauge
        -'delta_tot'  for the total density fluctuation
        -'delta_m' for the non-relativistic density fluctuation
        -'delta_cb' for the CDM+baryon density fluctuation
        -'delta_g' for the photon density fluctuation
        -'delta_b' for the baryon density fluctuation
        -'delta_cdm' for the CDM density fluctuation
        -'delta_idm' for the interacting DM density fluctuation
        -'delta_dcdm' for the decaying DM density fluctuation
        -'delta_fld' for the fluid density fluctuation
        -'delta_scf' for the scalar field density fluctuation
        -'delta_dr' for the decay radiation density fluctuation
        -'delta_ur' for the ultra-relativistic density fluctuation
        -'delta_idr' for the interacting dark radiation density fluctuation
        -'delta_ncdm[i]' for the ncdm[i] density fluctuation
        -'theta_tot' for the total velocity divergence
        -'theta_m' for the non-relativistic velocity divergence
        -'theta_cb' for the CDM+baryon velocity divergence
        -'theta_g' for the photon velocity divergence
        -'theta_b' for the baryon velocity divergence
        -'theta_cdm' for the CDM velocity divergence
        -'theta_idm' for the interacting DM velocity divergence
        -'theta_dcdm' for the decaying DM velocity divergence
        -'theta_fld' for the fluid velocity divergence
        -'theta_scf' for the scalar field velocity divergence
        -'theta_dr' for the decayt radiation velocity divergence
        -'theta_ur' for the ultra-relativistic velocity divergence
        -'theta_ncdm[i]' for the ncdm[i] velocity divergence

        The list of actual keys in the returned dictionary depends on
        what the user requested in the 'output' field.

        For each key, the returned dictionary contains an array of sources, indexed as:

        sources['key'][index_k][index_tau]

        Parameters
        ----------
        None

        Returns
        -------
        sources : dict
            Dictionary containing the source functions sources['key'][index_k][index_tau] (dimensionless)
        numpy array : k_array
            Array of k values (units of 1/Mpc)
        numpy array : tau_array
            Array of tau values (units of Mpc)
        "

