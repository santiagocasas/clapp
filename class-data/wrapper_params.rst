Parameters
----------


 .. function:: gt_current_derived_parameters(names)
  
  For a list of names, return a dictionary of those parameters (with those names) and the corresponding values
  Available options (currently, as of Jun 2025)
  * h
  * H0
  * Omega0_Lambda or Omega_Lambda
  * Omega0_fld
  * age (Gyrs)
  * conformal_age (Mpc)
  * m_ncdm_in_eV (only first massive neutrino species)
  * m_ncdm_tot (sum of all neutrino species, converted from :math:`\Omega_\mathrm{ncdm} h^2 \cdot 93.14\mathrm{eV}`
  * Neff
  * Omega_m
  * omega_m
  * xi_idr = T_idr/T_cmb
  * N_dg = Omega0_idr/Omega0_g * 8/7 * (11/4)^(4/3)
  * Gamma_0_nadm = a_idm_dr * (4/3) * Omega0_idr * h^2
  * a_dark = a_idm_dr
  * tau_reio
  * z_reio
  * z_rec
  * tau_rec
  * rs_rec
  * rs_rec_h = rs_rec * h
  * ds_rec = a_rec * rs_rec
  * ds_rec_h = ds_rec * h
  * ra_rec
  * ra_rec_H = ra_rec * h
  * da_rec = a_rec * ra_rec
  * da_rec_h = da_rec * h
  * z_star
  * tau_Star
  * rs_star
  * ds_star
  * ra_star
  * da_star
  * rd_star = damping scale (requires 'compute_damping_scale':'yes'
  * z_d = drag redshift
  * tau_d
  * rs_d
  * rs_d_h = rs_d * h
  * ds_d = a_d * rs_d
  * ds_d_h = ds_d * h
  * conf_time_reio
  * 100*theta_s = theta_s_100
  * 100*theta_star = theta_star_100
  * YHe
  * n_e
  * A_s
  * ln10^{10}A_s
  * exp_m_2_tau_As = exp(-2 tau_reio) * A_s
  * n_s
  * alpha_s
  * betA_s
  * r (at k=k_p=0.05/Mpc)
  * r_0002 (at k=0.002/Mpc)
  * n_t
  * alpha_t
  * V_0, V_1, V_2, V_3, V_4
  * epsilon_V (slow roll parameter)
  * eta_V (slow roll parameter)
  * ksi_V^2 (slow roll parameter)
  * phi_min
  * phi_max
  * sigma8
  * sigma8_cb
  * k_eq
  * a_eq
  * z_eq
  * H_eq
  * tau_eq
  * g_sd (spectral distortion)
  * y_sd (spectral distortion)
  * mu_sd (spectral distortion)

Convenience functions
^^^^^^^^^^^^^^^^^^^^^

These are convenience functions for you to get to your favorite quantities faster, but might be deprecated in the future. Also, for the use with samplers, please always use the :func:`get_derived_parameters`

 .. function:: age()
  
  Get total age of the universe (in Gyrs)

 .. function:: h()
  
  Get reduced Hubble constant :math:`h = H_0/(100\mathrm{km/s/Mpc})`

 .. function:: n_s()
  
  Get primordial power spectrum tilt

 .. function:: tau_reio()
  
  Get optical depth of reionization :math:`\tau_\mathrm{reio} = \int_{\eta_*}^{\eta_0} \frac{a \sigma_T n_e}{R}

 .. function:: z_reio()
  
  Get redshift of reionization

 .. function:: T_cmb()
  
  Get CMB temperature (can be used to convert :math:`C_\ell` to `\mu K` (by a factor of T_cmb**2 * 1e6**2)

 .. function:: N_eff()
  
  Get effective number of relativistic degrees of freedom :math:`\frac{7}{8} N_\mathrm{eff} \left(\frac{4}{11}\right)^{4/3} = \rho_\nu/\rho_g`

 .. function:: Omega_Lambda()
  
  Get cosmological constant energy density fraction :math:`\Omega_\Lambda = \rho_{\Lambda,0}/\mathrm{\rho_{\mathrm{crit},0}}`

 .. function:: Omega_m()
  
  Get matter energy density fraction :math:`\Omega_m = \rho_{m,0}/\mathrm{\rho_{\mathrm{crit},0}}`

 .. function:: Omega0_m()
  
  Depracated version of :func:`Omega_m`

 .. function:: Omega_g()
  
  Get photon energy density fraction :math:`\Omega_g = \rho_{g,0}/\mathrm{\rho_{\mathrm{crit},0}}`

 .. function:: Omega_b()
  
  Get baryon energy density fraction :math:`\Omega_b = \rho_{b,0}/\mathrm{\rho_{\mathrm{crit},0}}` 

 .. function:: Omega_r()
  
  Get radiation energy density fraction :math:`\Omega_r = \rho_{r,0}/\mathrm{\rho_{\mathrm{crit},0}}`

 .. function:: omega_b()
  
  Get reducded baryon energy density fraction :math:`\Omega_b h^2 = \rho_{b,0}/\mathrm{\rho_{\mathrm{crit},0}} \cdot h^2` 

 .. function:: Omega0_k()
  
  Get reducded curvature energy density fraction :math:`\Omega_k = \rho_{k,0}/\mathrm{\rho_{\mathrm{crit},0}}` 

 .. function:: Omega0_cdm()
  
  Get radiation energy density fraction :math:`\Omega_\mathrm{cdm} = \rho_{\mathrm{cdm},0}/\mathrm{\rho_{\mathrm{crit},0}}`
  

 .. function:: theta_s_100()
  
  Get sound horizon angle :math:`100 \theta_s = 100 r_s/r_A`

 .. function:: theta_star_100()
  
  Get sound horizon angle :math:`100 \theta_s = 100 r_*/r_A`

 .. function:: k_eq()
  
  Get equality wavenumbers :math:`k_\mathrm{eq} = a_\mathrm{eq} H_\mathrm{eq}`

 .. function:: z_eq()
  
  Get equality wavenumbers :math:`z_\mathrm{eq} = 1/a_\mathrm{eq} - 1`

 .. function:: rs_drag()
  
  Sound horizon :math:`r_s = \int_{z_\mathrm{drag}}^\infty c_s \mathrm{d}x/H(x)` 





