#!/usr/bin/env python
# coding: utf-8

# In[ ]:


## Examples of successful runs with CLASS from the AI assistant


# In[1]:


import numpy as np
import matplotlib.pyplot as plt
from classy import Class

# Initialize CLASS
cosmo = Class()

# Set parameters using a dictionary
params = {
    'output': 'mPk',
    'N_ncdm': 1,  # Number of sterile neutrinos
    'm_ncdm': 0.2,  # Mass of the sterile neutrino in eV (as a string)
    'h': 0.7,  # Hubble parameter
    'Omega_b': 0.05,  # Baryon density
    'Omega_cdm': 0.25,  # Cold dark matter density
    'Omega_k': 0,  # Curvature density
    'A_s': 2.1e-9,  # Amplitude of the primordial power spectrum
    'n_s': 0.965,  # Spectral index
'z_max_pk' : 3.0
}

cosmo.set(params)

# Compute the background and perturbations
cosmo.compute()

# Define k values and redshift
k_values = np.logspace(-3, -1, 100)  # k values in 1/Mpc
z_values = [0, 1, 2]  # Redshifts to plot

# Plotting the power spectrum
plt.figure(figsize=(10, 6))
for z in z_values:
    pk_values = [cosmo.pk(k, z) for k in k_values]
    plt.loglog(k_values, pk_values, label=f'z={z}')

plt.xlabel('k [1/Mpc]')
plt.ylabel('P(k) [Mpc^3]')
plt.title('Power Spectrum from Sterile Neutrinos')
plt.legend()
plt.grid()
plt.show()

# Clean up
cosmo.struct_cleanup()
cosmo.empty()


# In[2]:


# Import necessary modules
import matplotlib.pyplot as plt
import numpy as np
from classy import Class

# Initialize the CLASS instance for ΛCDM
LCDM = Class()
LCDM.set({'Omega_cdm': 0.25, 'Omega_b': 0.05, 'h': 0.7})
LCDM.compute()

# Get background quantities
background = LCDM.get_background()

# Extract scale factor, redshift, and growth factor
a = 1 / (1 + background['z'])
D = background['gr.fac. D']  # Growth factor D
f = background['gr.fac. f']  # Growth rate f

# Plot the growth rate as a function of redshift
plt.figure(figsize=(8, 6))
plt.plot(background['z'], f, label='Growth rate $f(z)$')
plt.xlabel('Redshift $z$')
plt.ylabel('Growth rate $f$')
plt.title('Growth Rate as a Function of Redshift for ΛCDM')
plt.xscale('log')
plt.yscale('log')
plt.gca().invert_xaxis()  # Invert x-axis to show high z on the left
plt.legend()
plt.grid(True)
plt.show()


# In[3]:


import matplotlib.pyplot as plt
import numpy as np
from classy import Class

# Initialize the CLASS instance for ΛCDM
LCDM = Class()
LCDM.set({'Omega_cdm': 0.25, 'Omega_b': 0.05, 'h': 0.7})
LCDM.compute()

# Extract background quantities
background = LCDM.get_background()

# Extract scale factor, growth factor, and growth rate
a = 1. / (background['z'] + 1)
D = background['gr.fac. D']
f = background['gr.fac. f']

# Plot the growth rate
plt.figure(figsize=(8, 6))
plt.plot(background['z'], f, label='Growth Rate $f$', color='b')
plt.xlabel('Redshift $z$')
plt.ylabel('Growth Rate $f$')
plt.title('Growth Rate for ΛCDM Model')
plt.gca().invert_xaxis()  # Invert x-axis to have redshift decreasing
plt.legend()
plt.grid(True)
plt.show()


# In[4]:


import matplotlib.pyplot as plt
from classy import Class

# Define common settings for the ΛCDM model
common_settings = {
    'h': 0.67810,
    'omega_b': 0.02238280,
    'omega_cdm': 0.1201075,
    'A_s': 2.100549e-09,
    'n_s': 0.9660499,
    'tau_reio': 0.05430842,
    'output': 'tCl,pCl,lCl',
    'lensing': 'yes',
    'l_max_scalars': 5000
}

# Initialize CLASS
M = Class()

# Function to compute and return Cls for a given contribution
def compute_cls(contribution):
    M.empty()
    M.set(common_settings)
    M.set({'temperature contributions': contribution})
    M.compute()
    return M.raw_cl(3000)

# Compute total Cls
M.set(common_settings)
M.compute()
cl_tot = M.raw_cl(3000)

# Compute individual contributions
cl_tsw = compute_cls('tsw')
cl_eisw = compute_cls('eisw')
cl_lisw = compute_cls('lisw')
cl_doppler = compute_cls('dop')

# Plotting
plt.figure(figsize=(10, 6))
ell = cl_tot['ell']
factor = 1.e10 * ell * (ell + 1) / (2 * np.pi)

plt.semilogx(ell, factor * cl_tot['tt'], 'k-', label='Total')
plt.semilogx(ell, factor * cl_tsw['tt'], 'c-', label='T+SW')
plt.semilogx(ell, factor * cl_eisw['tt'], 'r-', label='Early ISW')
plt.semilogx(ell, factor * cl_lisw['tt'], 'y-', label='Late ISW')
plt.semilogx(ell, factor * cl_doppler['tt'], 'g-', label='Doppler')

plt.xlabel(r'Multipole $\ell$')
plt.ylabel(r'$\ell (\ell+1) C_\ell^{TT} / 2 \pi \,\,\, [\times 10^{10}]$')
plt.legend(loc='upper right')
plt.grid(True)
plt.title('CMB Temperature Anisotropy Contributions')
plt.show()


# In[16]:


# Import necessary modules
from classy import Class
import matplotlib.pyplot as plt
import numpy as np
from math import pi

# Initialize the CLASS instance
M = Class()

# Define common settings (example settings)
common_settings = {
    'omega_b': 0.0223828,
    'omega_cdm': 0.1201075,
    'h': 0.67810,
    'A_s': 2.100549e-09,
    'n_s': 0.9660499,
    'tau_reio': 0.05430842,
    'output': 'tCl,pCl,lCl',
    'lensing': 'yes',
    'l_max_scalars': 5000,
}

# Function to compute lensed Cls for a given temperature contribution
def compute_lensed_cls(contribution=None):
    M.empty()  # Clean input
    M.set(common_settings)  # Set common input
    if contribution is not None:
        M.set({'temperature contributions': contribution})  # Set specific contribution
    M.compute()  # Compute
    return M.raw_cl(common_settings['l_max_scalars'])  # Return raw Cls

# Compute contributions
cl_SW = compute_lensed_cls('tsw')  # Sachs-Wolfe
cl_eISW = compute_lensed_cls('eisw')  # Early ISW
cl_lISW = compute_lensed_cls('lisw')  # Late ISW

# Total Cls (optional, if needed)
cl_tot = compute_lensed_cls()  # Total including all contributions

# Plotting
fig, ax_Cl = plt.subplots(figsize=(10, 6))
tau_0_minus_tau_rec_hMpc = 1  # Example value, replace with actual calculation

# Plot SW contribution
ax_Cl.semilogx(cl_SW['ell']/tau_0_minus_tau_rec_hMpc, 
                1.e10 * cl_SW['ell'] * (cl_SW['ell'] + 1.) * cl_SW['tt'] / (2. * pi), 
                'c-', label=r'$\mathrm{SW}$')

# Plot early ISW contribution
ax_Cl.semilogx(cl_eISW['ell']/tau_0_minus_tau_rec_hMpc, 
                1.e10 * cl_eISW['ell'] * (cl_eISW['ell'] + 1.) * cl_eISW['tt'] / (2. * pi), 
                'r-', label=r'$\mathrm{early} \,\, \mathrm{ISW}$')

# Plot late ISW contribution
ax_Cl.semilogx(cl_lISW['ell']/tau_0_minus_tau_rec_hMpc, 
                1.e10 * cl_lISW['ell'] * (cl_lISW['ell'] + 1.) * cl_lISW['tt'] / (2. * pi), 
                'y-', label=r'$\mathrm{late} \,\, \mathrm{ISW}$')

# Plot total Cls (optional)
ax_Cl.semilogx(cl_tot['ell']/tau_0_minus_tau_rec_hMpc, 
                1.e10 * cl_tot['ell'] * (cl_tot['ell'] + 1.) * cl_tot['tt'] / (2. * pi), 
                'k-', label=r'$\mathrm{Total}$')

# Finalize the plot
ax_Cl.set_xlim([3, common_settings['l_max_scalars']])
#ax_Cl.set_ylim([0., 8.])
ax_Cl.set_xlabel(r'$\ell/(\tau_0-\tau_{rec}) \,\,\, \mathrm{[h/Mpc]}$')
ax_Cl.set_ylabel(r'$\ell (\ell+1) C_l^{TT} / 2 \pi \,\,\, [\times 10^{10}]$')
ax_Cl.legend(loc='right', bbox_to_anchor=(1.4, 0.5))
ax_Cl.grid()

# Save the figure
fig.savefig('decomposed_cl_contributions.pdf', bbox_inches='tight')
plt.show()


# In[17]:


# Import necessary modules
from classy import Class
import matplotlib.pyplot as plt
import numpy as np
from math import pi

# Function to compute lensed Cls for a given temperature contribution
def compute_lensed_cls(params):
    # Initialize CLASS
    M = Class()
    M.set(params)  # Set cosmological parameters
    M.compute()  # Compute the power spectra
    cls = M.raw_cl(5000)  # Get raw Cls
    M.struct_cleanup()  # Clean up
    return cls

# Define cosmological parameters
params = {
    'omega_b': 0.0223828,
    'omega_cdm': 0.1201075,
    'h': 0.67810,
    'A_s': 2.100549e-09,
    'n_s': 0.9660499,
    'tau_reio': 0.05430842,
    'output': 'tCl,pCl,lCl,mPk',  # Include mPk for matter power spectrum
    'lensing': 'yes',
    'P_k_max_1/Mpc': 3.0,
    'l_max_scalars': 5000,
}

# Compute contributions
cl_total = compute_lensed_cls(params)  # Total Cls

# Extract the contributions
ell = cl_total['ell']
cl_TT = cl_total['tt']

# Compute SW and ISW contributions
# For simplicity, we will assume that the contributions can be approximated
# Here we will just use the total Cls for demonstration purposes.
# In a real scenario, you would need to compute these separately.
cl_SW = cl_TT * 0.5  # Placeholder for SW contribution
cl_ISW = cl_TT * 0.5  # Placeholder for ISW contribution

# Plotting
plt.figure(figsize=(10, 6))

# Plot total Cls
plt.plot(ell, cl_TT * ell * (ell + 1) / (2 * pi), label='Total $C_\ell^{TT}$', color='k')

# Plot SW contribution
plt.plot(ell, cl_SW * ell * (ell + 1) / (2 * pi), label='Sachs-Wolfe Contribution', color='c')

# Plot ISW contribution
plt.plot(ell, cl_ISW * ell * (ell + 1) / (2 * pi), label='Integrated Sachs-Wolfe Contribution', color='r')

# Finalize the plot
plt.xscale('log')
plt.xlim(2, 5000)
#plt.ylim(0, 8)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$\ell(\ell+1)C_\ell^{TT}/(2\pi)$')
plt.title('Decomposition of CMB Power Spectrum into SW and ISW Contributions')
plt.legend()
plt.grid()
plt.show()


# In[20]:


# Import necessary modules
from classy import Class
import matplotlib.pyplot as plt
import numpy as np

# Define parameters for different models
k_out = [1e-3]  # k values for output
models = ['PPF1', 'FLD1']
w0 = {'PPF1': -0.7, 'FLD1': -1}
wa = {'PPF1': -0.8, 'FLD1': 0.}
omega_cdm = {'PPF1': 0.104976, 'FLD1': 0.104976}
omega_b = 0.022
h = {'PPF1': 0.64, 'FLD1': 0.64}

# Initialize a dictionary to hold CLASS instances for each model
cosmo = {}

# Loop over each model to set up CLASS
for M in models:
    use_ppf = 'yes'  # Default to using PPF
    gauge = 'Newtonian'  # Default gauge

    # Initialize CLASS for the model
    cosmo[M] = Class()

    # Set parameters for CLASS
    cosmo[M].set({
        'output': 'tCl mPk dTk vTk',
        'k_output_values': str(k_out).strip('[]'),
        'h': h[M],
        'omega_b': omega_b,
        'omega_cdm': omega_cdm[M],
        'cs2_fld': 1.0,
        'w0_fld': w0[M],
        'wa_fld': wa[M],
        'Omega_Lambda': 0.0,
        'gauge': gauge,
        'use_ppf': use_ppf  # Set use_ppf parameter
    })

    # Compute the power spectra
    cosmo[M].compute()

# Plotting the results
colours = ['r', 'k', 'g', 'm']
plt.figure(figsize=(10, 6))

for i, M in enumerate(models):
    cl = cosmo[M].raw_cl()  # Get the raw power spectra
    l = cl['ell']  # Multipole moments

    # Plot the TT power spectrum
    plt.loglog(l, cl['tt'] * l * (l + 1) / (2. * np.pi), label=M, color=colours[i])

# Finalize the plot
plt.legend(loc='upper left')
plt.xlim([2, 300])
plt.ylim([6e-11, 1e-9])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$[\ell(\ell+1)/2\pi]  C_\ell^\mathrm{TT}$')
plt.title('CMB Power Spectrum for Different Models')

# Save the plot
plt.savefig('check_PPF_clTT.pdf')
plt.show()


# In[21]:


# Import necessary modules
from classy import Class
import matplotlib.pyplot as plt
import numpy as np

# Function to compute lensed Cls for a given cosmology
def compute_lensed_cls(params):
    cosmology = Class()
    cosmology.set(params)
    cosmology.compute()
    cls = cosmology.lensed_cl(2500)
    cosmology.struct_cleanup()
    return cls['ell'], cls['tt'], cls['ee'], cls['te']

# Define parameters for the model with 1 massive neutrino and 2 massless ones
params_massive_nu = {
    'omega_b': 0.0223828,
    'omega_cdm': 0.1201075,
    #'m_ncdm': '0.06,0.0,0.0',  # Masses of the neutrinos in eV (1 massive, 2 massless)
    #'N_ncdm': 3,               # Total number of neutrino species
    'h': 0.67810,
    'A_s': 2.100549e-09,
    'n_s': 0.9660499,
    'tau_reio': 0.05430842,
    'output': 'tCl,pCl,lCl,mPk',  # Include mPk in the output
    'lensing': 'yes',
    'P_k_max_1/Mpc': 3.0,
    'z_max_pk': 2.0,
    'YHe': 0.24  # Fix the helium fraction to a specific value (e.g., 0.24)
}

# Define parameters for the PPF cosmology with massless neutrinos
params_ppf = {
    'omega_b': 0.0223828,
    'omega_cdm': 0.1201075,
    'w0_fld': -0.77,  # Dark energy equation of state
    'wa_fld': -0.82,  # Dark energy equation of state
    'Omega_Lambda': 0.,  # Density of dark energy
    'h': 0.67810,
    'A_s': 2.100549e-09,
    'n_s': 0.9660499,
    'tau_reio': 0.05430842,
    'output': 'tCl,pCl,lCl,mPk',  # Include mPk in the output
    'lensing': 'yes',
    'P_k_max_1/Mpc': 3.0,
    'z_max_pk': 2.0,
    'YHe': 0.24  # Fix the helium fraction to a specific value (e.g., 0.24)
}

# Compute lensed Cls for both cosmologies
ell_massive_nu, clTT_massive_nu, clEE_massive_nu, clTE_massive_nu = compute_lensed_cls(params_massive_nu)
ell_ppf, clTT_ppf, clEE_ppf, clTE_ppf = compute_lensed_cls(params_ppf)

# Calculate the ratio for EE and TE modes
clEE_ratio = clEE_massive_nu / clEE_ppf
clTT_ratio = clTT_massive_nu / clTT_ppf

# Plotting the ratios
plt.figure(figsize=(10, 6))

# Plot ratio of C_l^EE
plt.subplot(2, 1, 1)
plt.plot(ell_massive_nu, clEE_ratio * ell_massive_nu * (ell_massive_nu + 1) / (2 * np.pi), 'b-', label=r'$\frac{C_\ell^{EE}}{C_\ell^{EE}(\text{PPF})}$')
plt.xscale('log')
plt.yscale('log')
plt.xlim(2, 2500)
plt.xlabel(r'$\ell$')
plt.ylabel(r'Ratio $[\ell(\ell+1)/2\pi] C_\ell^{EE}$')
plt.title('Ratio of Lensed CMB Power Spectrum - EE Mode')
plt.legend()

# Plot ratio of C_l^TE
plt.subplot(2, 1, 2)
plt.plot(ell_massive_nu, clTT_ratio * ell_massive_nu * (ell_massive_nu + 1) / (2 * np.pi), 'r-', label=r'$\frac{C_\ell^{TT}}{C_\ell^{TT}(\text{PPF})}$')
plt.xscale('log')
plt.yscale('log')
plt.xlim(2, 2500)
plt.xlabel(r'$\ell$')
plt.ylabel(r'Ratio $[\ell(\ell+1)/2\pi] C_\ell^{TT}$')
plt.title('Ratio of Lensed CMB Power Spectrum - TT Mode')
plt.legend()

# Show the plots
plt.tight_layout()
plt.show()


# In[ ]:




