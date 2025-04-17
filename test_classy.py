# Import necessary modules
from classy import Class
import matplotlib.pyplot as plt
import numpy as np

# Define common settings for the ΛCDM model, including a fixed Helium fraction
common_settings = {
    'h': 0.67810,
    'omega_b': 0.02238280,
    'omega_cdm': 0.1201075,
    'A_s': 2.100549e-09,
    'n_s': 0.9660499,
    'tau_reio': 0.05430842,
    'YHe': 0.24,  # Set the Helium fraction explicitly
    'output': 'tCl,lCl',  # Include 'lCl' for lensing potential
    'lensing': 'yes',
    'l_max_scalars': 2500
}

# Initialize CLASS
cosmo = Class()
cosmo.set(common_settings)
cosmo.compute()

# Get the C_l's
cls = cosmo.lensed_cl(2500)

# Extract ell and C_ell^TT
ell = cls['ell'][2:]  # Start from 2 to avoid the monopole and dipole
clTT = cls['tt'][2:]

# Plotting
plt.figure(figsize=(10, 6))
factor = ell * (ell + 1) / (2 * np.pi) * 1e12  # Factor to convert to D_ell

plt.plot(ell, factor * clTT, label='Temperature $C_\ell^{TT}$', color='b')
plt.xlabel(r'Multipole moment $\ell$')
plt.ylabel(r'$\ell (\ell + 1) C_\ell^{TT} / 2\pi \, [\mu K^2]$')
plt.title('CMB Temperature Power Spectrum')
plt.xscale('log')
plt.yscale('log')
plt.grid(True)
plt.legend()
plt.savefig('cmb_temperature_spectrum.png')  # Save the plot to a file

# Clean up
cosmo.struct_cleanup()
cosmo.empty()