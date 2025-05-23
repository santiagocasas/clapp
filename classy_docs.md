## CPU.py

```python
#!/usr/bin/env python
"""
.. module:: CPU
    :synopsis: CPU, a CLASS Plotting Utility
.. moduleauthor:: Benjamin Audren <benjamin.audren@gmail.com>
.. credits:: Benjamin Audren, Jesus Torrado
.. version:: 2.0

This is a small python program aimed to gain time when comparing two spectra,
e.g. from CAMB and CLASS, or a non-linear spectrum to a linear one.

It is designed to be used in a command line fashion, not being restricted to
your CLASS directory, though it recognizes mainly CLASS output format. Far from
perfect, or complete, it could use any suggestion for enhancing it,
just to avoid losing time on useless matters for others.

Be warned that, when comparing with other format, the following is assumed:
there are no empty line (especially at the end of file). Gnuplot comment lines
(starting with a # are allowed). This issue will cause a non-very descriptive
error in CPU, any suggestion for testing it is welcome.

Example of use:
- To superimpose two different spectra and see their global shape :
python CPU.py output/lcdm_z2_pk.dat output/lncdm_z2_pk.dat
- To see in details their ratio:
python CPU.py output/lcdm_z2_pk.dat output/lncdm_z2_pk.dat -r

The "PlanckScale" is taken with permission from Jesus Torrado's:
cosmo_mini_toolbox, available under GPLv3 at
https://github.com/JesusTorrado/cosmo_mini_toolbox

"""

from __future__ import unicode_literals, print_function

# System imports
import os
import sys
import argparse

# Numerics
import numpy as np
from numpy import ma
from scipy.interpolate import InterpolatedUnivariateSpline
from math import floor

# Plotting
import matplotlib.pyplot as plt
from matplotlib import scale as mscale
from matplotlib.transforms import Transform
from matplotlib.ticker import FixedLocator


def CPU_parser():
    parser = argparse.ArgumentParser(
        description=(
            'CPU, a CLASS Plotting Utility, specify wether you want\n'
            'to superimpose, or plot the ratio of different files.'),
        epilog=(
            'A standard usage would be, for instance:\n'
            'python CPU.py output/test_pk.dat output/test_pk_nl_density.dat'
            ' -r\npython CPU.py output/wmap_cl.dat output/planck_cl.dat'),
        formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument(
        'files', type=str, nargs='*', help='Files to plot')
    parser.add_argument('-r', '--ratio', dest='ratio', action='store_true',
                        help='Plot the ratio of the spectra')
    parser.add_argument('-y', '--y-axis', dest='y_axis', nargs='+',
                        help='specify the fields you want to plot.')
    parser.add_argument('-x', '--x-axis', dest='x_axis', type=str,
                        help='specify the field to be used on the x-axis')
    parser.add_argument('--scale', type=str,
                        choices=['lin', 'loglog', 'loglin', 'george'],
                        help='Specify the scale to use for the plot')
    parser.add_argument('--xlim', dest='xlim', nargs='+', type=float,
                        default=[], help='Specify the x range')
    parser.add_argument('--ylim', dest='ylim', nargs='+', type=float,
                        default=[], help='Specify the y range')
    parser.add_argument(
        '-p, --print',
        dest='printfile', default='',
        help=('print the graph directly in a file. If no name is specified, it'
              'uses the name of the first input file'))
    parser.add_argument(
        '--repeat',
        dest='repeat', action='store_true', default=False,
        help='repeat the step for all redshifts with same base name')
    return parser


def plot_CLASS_output(files, x_axis, y_axis, ratio=False, printing='',
                      output_name='', extension='', x_variable='',
                      scale='lin', xlim=[], ylim=[]):
    """
    Load the data to numpy arrays, write all the commands for plotting to a
    Python script for further refinment, and display them.

    Inspired heavily by the matlab version by Thomas Tram

    Parameters
    ----------
    files : list
        List of files to plot
    x-axis : string
        name of the column to use as the x coordinate
    y-axis : list, str
        List of items to plot, which should match the way they appear in the
        file, for instance: ['TT', 'BB]

    Keyword Arguments
    -----------------
    ratio : bool
        If set to yes, plots the ratio of the files, taking as a reference the
        first one
    output_name : str
        Specify a different name for the produced figure (by default, it takes
        the name of the first file, and replace the .dat by .pdf)
    extension : str

    """
    # Define the python script name, and the pdf path
    python_script_path = os.path.splitext(files[0])[0]+'.py'

    # The variable text will contain all the lines to be printed in the end to
    # the python script path, joined with newline characters. Beware of the
    # indentation.
    text = ['import matplotlib.pyplot as plt',
            'import numpy as np',
            'import itertools', '']

    # Load all the graphs
    data = []
    for data_file in files:
        data.append(np.loadtxt(data_file))

    # Create the full_path_files list, that contains the absolute path, so that
    # the future python script can import them directly.
    full_path_files = [os.path.abspath(elem) for elem in files]

    text += ['files = %s' % full_path_files]
    text += ['data = []',
             'for data_file in files:',
             '    data.append(np.loadtxt(data_file))']

    # Recover the base name of the files, everything before the dot
    roots = [elem.split(os.path.sep)[-1].split('.')[0] for elem in files]
    text += ['roots = [%s]' % ', '.join(["'%s'" % root for root in roots])]

    # Create the figure and ax objects
    fig, ax = plt.subplots()
    text += ['', 'fig, ax = plt.subplots()']

    # if ratio is not set, then simply plot them all
    original_y_axis = y_axis
    legend = []
    if not ratio:
        for index, curve in enumerate(data):
            # Recover the number of columns in the first file, as well as their
            # title.
            num_columns, names, tex_names = extract_headers(files[index])

            text += ['', 'index, curve = %i, data[%i]' % (index, index)]
            # Check if everything is in order
            if num_columns == 2:
                y_axis = [names[1]]
            elif num_columns > 2:
                # in case y_axis was only a string, cast it to a list
                if isinstance(original_y_axis, str):
                    y_axis = [original_y_axis]
                else:
                    y_axis = original_y_axis

            # Store the selected text and tex_names to the script
            selected = []
            for elem in y_axis:
                selected.extend(
                    [name for name in names if name.find(elem) != -1 and
                     name not in selected])
            if not y_axis:
                selected = names[1:]
            y_axis = selected

            # Decide for the x_axis, by default the index will be set to zero
            x_index = 0
            if x_axis:
                for index_name, name in enumerate(names):
                    if name.find(x_axis) != -1:
                        x_index = index_name
                        break
            # Store to text
            text += ['y_axis = %s' % selected]
            text += ['tex_names = %s' % [elem for (elem, name) in
                     zip(tex_names, names) if name in selected]]
            text += ["x_axis = '%s'" % tex_names[x_index]]
            text += ["ylim = %s" % ylim]
            text += ["xlim = %s" % xlim]

            for selec in y_axis:
                index_selec = names.index(selec)
                plot_line = 'ax.'
                if scale == 'lin':
                    plot_line += 'plot(curve[:, %i], curve[:, %i])' % (
                        x_index, index_selec)
                    ax.plot(curve[:, x_index], curve[:, index_selec])
                elif scale == 'loglog':
                    plot_line += 'loglog(curve[:, %i], abs(curve[:, %i]))' % (
                        x_index, index_selec)
                    ax.loglog(curve[:, x_index], abs(curve[:, index_selec]))
                elif scale == 'loglin':
                    plot_line += 'semilogx(curve[:, %i], curve[:, %i])' % (
                        x_index, index_selec)
                    ax.semilogx(curve[:, x_index], curve[:, index_selec])
                elif scale == 'george':
                    plot_line += 'plot(curve[:, %i], curve[:, %i])' % (
                        x_index, index_selec)
                    ax.plot(curve[:, x_index], curve[:, index_selec])
                    ax.set_xscale('planck')
                text += [plot_line]

            legend.extend([roots[index]+': '+elem for elem in y_axis])

        ax.legend(legend, loc='best')
        text += ["",
                 "ax.legend([root+': '+elem for (root, elem) in",
                 "    itertools.product(roots, y_axis)], loc='best')",
                 ""]
    else:
        ref = data[0]
        num_columns, ref_curve_names, ref_tex_names = extract_headers(files[0])
        # Check if everything is in order
        if num_columns == 2:
            y_axis_ref = [ref_curve_names[1]]
        elif num_columns > 2:
            # in case y_axis was only a string, cast it to a list
            if isinstance(original_y_axis, str):
                y_axis_ref = [original_y_axis]
            else:
                y_axis_ref = original_y_axis

        # Store the selected text and tex_names to the script
        selected = []
        for elem in y_axis_ref:
            selected.extend([name for name in ref_curve_names if name.find(elem) != -1 and
                             name not in selected])
        y_axis_ref = selected

        # Decide for the x_axis, by default the index will be set to zero
        x_index_ref = 0
        if x_axis:
            for index_name, name in enumerate(ref_curve_names):
                if name.find(x_axis) != -1:
                    x_index_ref = index_name
                    break

        for idx in range(1, len(data)):
            current = data[idx]
            num_columns, names, tex_names = extract_headers(files[idx])

            # Check if everything is in order
            if num_columns == 2:
                y_axis = [names[1]]
            elif num_columns > 2:
                # in case y_axis was only a string, cast it to a list
                if isinstance(original_y_axis, str):
                    y_axis = [original_y_axis]
                else:
                    y_axis = original_y_axis

            # Store the selected text and tex_names to the script
            selected = []
            for elem in y_axis:
                selected.extend([name for name in names if name.find(elem) != -1 and
                                 name not in selected])
            y_axis = selected

            text += ['y_axis = %s' % selected]
            text += ['tex_names = %s' % [elem for (elem, name) in
                                         zip(tex_names, names) if name in selected]]

            # Decide for the x_axis, by default the index will be set to zero
            x_index = 0
            if x_axis:
                for index_name, name in enumerate(names):
                    if name.find(x_axis) != -1:
                        x_index = index_name
                        break

            text += ["x_axis = '%s'" % tex_names[x_index]]
            for selec in y_axis:
                # Do the interpolation
                axis = ref[:, x_index_ref]
                reference = ref[:, ref_curve_names.index(selec)]
                #plt.loglog(current[:, x_index], current[:, names.index(selec)])
                #plt.show()
                #interpolated = splrep(current[:, x_index],
                                      #current[:, names.index(selec)])
                interpolated = InterpolatedUnivariateSpline(current[:, x_index],
                                      current[:, names.index(selec)])
                if scale == 'lin':
                    #ax.plot(axis, splev(ref[:, x_index_ref],
                                        #interpolated)/reference-1)
                    ax.plot(axis, interpolated(ref[:, x_index_ref])/reference-1)
                elif scale == 'loglin':
                    #ax.semilogx(axis, splev(ref[:, x_index_ref],
                                            #interpolated)/reference-1)
                    ax.semilogx(axis, interpolated(ref[:, x_index_ref])/reference-1)
                elif scale == 'loglog':
                    raise InputError(
                        "loglog plot is not available for ratios")

    if 'TT' in names:
        ax.set_xlabel('$\ell$', fontsize=16)
        text += ["ax.set_xlabel('$\ell$', fontsize=16)"]
    elif 'P' in names:
        ax.set_xlabel('$k$ [$h$/Mpc]', fontsize=16)
        text += ["ax.set_xlabel('$k$ [$h$/Mpc]', fontsize=16)"]
    else:
        ax.set_xlabel(tex_names[x_index], fontsize=16)
        text += ["ax.set_xlabel('%s', fontsize=16)" % tex_names[x_index]]
    if xlim:
        if len(xlim) > 1:
            ax.set_xlim(xlim)
            text += ["ax.set_xlim(xlim)"]
        else:
            ax.set_xlim(xlim[0])
            text += ["ax.set_xlim(xlim[0])"]
        ax.set_ylim()
        text += ["ax.set_ylim()"]
    if ylim:
        if len(ylim) > 1:
            ax.set_ylim(ylim)
            text += ["ax.set_ylim(ylim)"]
        else:
            ax.set_ylim(ylim[0])
            text += ["ax.set_ylim(ylim[0])"]
    text += ['plt.show()']
    plt.show()

    # If the use wants to print the figure to a file
    if printing:
        fig.savefig(printing)
        text += ["fig.savefig('%s')" % printing]

    # Write to the python file all the issued commands. You can then reproduce
    # the plot by running "python output/something_cl.dat.py"
    with open(python_script_path, 'w') as python_script:
        print('Creating a python script to reproduce the figure')
        print('--> stored in %s' % python_script_path)
        python_script.write('\n'.join(text))

    # If the use wants to print the figure to a file
    if printing:
        fig.savefig(printing)


class FormatError(Exception):
    """Format not recognised"""
    pass


class TypeError(Exception):
    """Spectrum type not recognised"""
    pass


class NumberOfFilesError(Exception):
    """Invalid number of files"""
    pass


class InputError(Exception):
    """Incompatible input requirements"""
    pass


def replace_scale(string):
    """
    This assumes that the string starts with "(.)", which will be replaced by
    (8piG/3)

    >>> print replace_scale('(.)toto')
    >>> '(8\\pi G/3)toto'
    """
    string_list = list(string)
    string_list.pop(1)
    string_list[1:1] = list('8\\pi G/3')
    return ''.join(string_list)


def process_long_names(long_names):
    """
    Given the names extracted from the header, return two arrays, one with the
    short version, and one tex version

    >>> names, tex_names = process_long_names(['(.)toto', 'proper time [Gyr]'])
    >>> print names
    >>> ['toto', 'proper time']
    >>> print tex_names
    >>> ['(8\\pi G/3)toto, 'proper time [Gyr]']

    """
    names = []
    tex_names = []
    # First pass, to remove the leading scales
    for name in long_names:
        # This can happen in the background file
        if name.startswith('(.)', 0):
            temp_name = name[3:]
            names.append(temp_name)
            tex_names.append(replace_scale(name))
        # Otherwise, we simply
        else:
            names.append(name)
            tex_names.append(name)

    # Finally, remove any extra spacing
    names = [''.join(elem.split()) for elem in names]
    return names, tex_names


def extract_headers(header_path):
    with open(header_path, 'r') as header_file:
        header = [line for line in header_file if line[0] == '#']
        header = header[-1]

    # Count the number of columns in the file, and recover their name. Thanks
    # Thomas Tram for the trick
    indices = [i+1 for i in range(len(header)) if
               header.startswith(':', i)]
    num_columns = len(indices)
    long_names = [header[indices[i]:indices[(i+1)]-3].strip() if i < num_columns-1
                  else header[indices[i]:].strip()
                  for i in range(num_columns)]

    # Process long_names further to handle special cases, and extract names,
    # which will correspond to the tags specified in "y_axis".
    names, tex_names = process_long_names(long_names)

    return num_columns, names, tex_names


def main():
    print('~~~ Running CPU, a CLASS Plotting Utility ~~~')
    parser = CPU_parser()
    # Parse the command line arguments
    args = parser.parse_args()

    # if there are no argument in the input, print usage
    if len(args.files) == 0:
        parser.print_usage()
        return

    # if the first file name contains cl or pk, infer the type of desired
    # spectrum
    if not args.y_axis:
        if args.files[0].rfind('cl') != -1:
            scale = 'loglog'
        elif args.files[0].rfind('pk') != -1:
            scale = 'loglog'
        else:
            scale = 'lin'
        args.y_axis = []
    else:
        scale = ''
    if not args.scale:
        if scale:
            args.scale = scale
        else:
            args.scale = 'lin'

    # Remove extra spacing in the y_axis list
    args.y_axis = [''.join(elem.split()) for elem in args.y_axis]
    # If ratio is asked, but only one file was passed in argument, politely
    # complain
    if args.ratio:
        if len(args.files) < 2:
            raise NumberOfFilesError(
                "If you want me to compute a ratio between two files, "
                "I strongly encourage you to give me at least two of them.")
    # actual plotting. By default, a simple superposition of the graph is
    # performed. If asked to be divided, the ratio is shown - whether a need
    # for interpolation arises or not.
    if args.ratio and args.scale == 'loglog':
        print("Defaulting to loglin scale")
        args.scale = 'loglin'

    plot_CLASS_output(args.files, args.x_axis, args.y_axis,
                      ratio=args.ratio, printing=args.printfile,
                      scale=args.scale, xlim=args.xlim, ylim=args.ylim)


# Helper code from cosmo_mini_toolbox, by Jesus Torrado, available fully at
# https://github.com/JesusTorrado/cosmo_mini_toolbox, to use the log then
# linear scale for the multipole axis when plotting Cl.
nonpos = "mask"
change = 50.0
factor = 500.


def _mask_nonpos(a):
    """
    Return a Numpy masked array where all non-positive 1 are
    masked. If there are no non-positive, the original array
    is returned.
    """
    mask = a <= 0.0
    if mask.any():
        return ma.MaskedArray(a, mask=mask)
    return a


def _clip_smaller_than_one(a):
    a[a <= 0.0] = 1e-300
    return a


class PlanckScale(mscale.ScaleBase):
    """
    Scale used by the Planck collaboration to plot Temperature power spectra:
    base-10 logarithmic up to l=50, and linear from there on.

    Care is taken so non-positive values are not plotted.
    """
    name = 'planck'

    def __init__(self, axis, **kwargs):
        pass

    def set_default_locators_and_formatters(self, axis):
        axis.set_major_locator(
            FixedLocator(
                np.concatenate((np.array([2, 10, change]),
                                np.arange(500, 2500, 500)))))
        axis.set_minor_locator(
            FixedLocator(
                np.concatenate((np.arange(2, 10),
                                np.arange(10, 50, 10),
                                np.arange(floor(change/100), 2500, 100)))))

    def get_transform(self):
        """
        Return a :class:`~matplotlib.transforms.Transform` instance
        appropriate for the given logarithm base.
        """
        return self.PlanckTransform(nonpos)

    def limit_range_for_scale(self, vmin, vmax, minpos):
        """
        Limit the domain to positive values.
        """
        return (vmin <= 0.0 and minpos or vmin,
                vmax <= 0.0 and minpos or vmax)

    class PlanckTransform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def __init__(self, nonpos):
            Transform.__init__(self)
            if nonpos == 'mask':
                self._handle_nonpos = _mask_nonpos
            else:
                self._handle_nonpos = _clip_nonpos

        def transform_non_affine(self, a):
            lower = a[np.where(a<=change)]
            greater = a[np.where(a> change)]
            if lower.size:
                lower = self._handle_nonpos(lower * 10.0)/10.0
                if isinstance(lower, ma.MaskedArray):
                    lower = ma.log10(lower)
                else:
                    lower = np.log10(lower)
                lower = factor*lower
            if greater.size:
                greater = (factor*np.log10(change) + (greater-change))
            # Only low
            if not(greater.size):
                return lower
            # Only high
            if not(lower.size):
                return greater
            return np.concatenate((lower, greater))

        def inverted(self):
            return PlanckScale.InvertedPlanckTransform()

    class InvertedPlanckTransform(Transform):
        input_dims = 1
        output_dims = 1
        is_separable = True
        has_inverse = True

        def transform_non_affine(self, a):
            lower = a[np.where(a<=factor*np.log10(change))]
            greater = a[np.where(a> factor*np.log10(change))]
            if lower.size:
                if isinstance(lower, ma.MaskedArray):
                    lower = ma.power(10.0, lower/float(factor))
                else:
                    lower = np.power(10.0, lower/float(factor))
            if greater.size:
                greater = (greater + change - factor*np.log10(change))
            # Only low
            if not(greater.size):
                return lower
            # Only high
            if not(lower.size):
                return greater
            return np.concatenate((lower, greater))

        def inverted(self):
            return PlanckTransform()

# Finished. Register the scale!
mscale.register_scale(PlanckScale)

if __name__ == '__main__':
    sys.exit(main())

```

## Growth_with_w.py

```python
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from classy import Class
from scipy import interpolate


# In[ ]:


w0vec = [-0.7, -1.0, -1.3]
wavec = [-0.2,0.0,0.2]
#w0vec = [-1.0]
#wavec = [0.0]

cosmo = {}
for w0 in w0vec:
    for wa in wavec:
        if w0==-1.0 and wa==0.0:
            M='LCDM'
        else:
            M = '('+str(w0)+','+str(wa)+')'
        cosmo[M] = Class()
        cosmo[M].set({'input_verbose':1,'background_verbose':1,'gauge' : 'Newtonian'})
        if M!='LCDM':
            cosmo[M].set({'Omega_Lambda':0.,'w0_fld':w0,'wa_fld':wa})
        cosmo[M].compute()


# In[ ]:


import scipy
import scipy.special
import scipy.integrate

def D_hypergeom(avec,csm):
    bg = csm.get_background()
    Om = csm.Omega0_m()
    if '(.)rho_lambda' in bg:
        Ol = bg['(.)rho_lambda'][-1]/bg['(.)rho_crit'][-1]
    else:
        Ol = bg['(.)rho_fld'][-1]/bg['(.)rho_crit'][-1]
        
    x = Ol/Om*avec**3
    D = avec*scipy.special.hyp2f1(1./3.,1,11./6.,-x)
    D_today = scipy.special.hyp2f1(1./3.,1,11./6.,-Ol/Om)
    return D/D_today

def f_hypergeom(avec,csm):
    bg = csm.get_background()
    Om = csm.Omega0_m()
    if '(.)rho_lambda' in bg:
        Ol = bg['(.)rho_lambda'][-1]/bg['(.)rho_crit'][-1]
    else:
        Ol = bg['(.)rho_fld'][-1]/bg['(.)rho_crit'][-1]
        
    x = Ol/Om*avec**3
    D = avec*scipy.special.hyp2f1(1./3.,1,11./6.,-x)
    f = 1.-6./11.*x*avec/D*scipy.special.hyp2f1(4./3.,2,17./6.,-x)
    return f

def D_integral2(avec,csm):
    bg = csm.get_background()
    Om = csm.Omega0_m()
    if '(.)rho_lambda' in bg:
        Ol = bg['(.)rho_lambda'][-1]/bg['(.)rho_crit'][-1]
        w0 = -1
        wa = 0.0
    else:
        Ol = bg['(.)rho_fld'][-1]/bg['(.)rho_crit'][-1]
        w0 = csm.pars['w0_fld']
        wa = csm.pars['wa_fld']
    D = np.zeros(avec.shape)
    for idx, a in enumerate(avec):
        Hc = a*np.sqrt(Om/a**3 + Ol*a**(-3*(1+w0+wa))*np.exp(-3.*(1.0-a)*wa) )
        Dintegrand2 = lambda a: (a*np.sqrt(Om/a**3 + Ol*a**(-3*(1+w0+wa))*np.exp(-3.*(1.0-a)*wa) ))**(-3)
        I = scipy.integrate.quad(Dintegrand2, 1e-15,a)
        D[idx] = Hc/a*I[0]
    D = D/scipy.integrate.quad(Dintegrand2,1e-15,1)[0]
    return D

def D_integral(avec,csm):
    bg = csm.get_background()
    Om = csm.Omega0_m()
    Ol = bg['(.)rho_lambda'][-1]/bg['(.)rho_crit'][-1]
    Or = 1-Om-Ol
    def Dintegrand(a):
        Hc = np.sqrt(Om/a+Ol*a*a+Or/a/a)
        #print a,Hc
        return Hc**(-3)
    D = np.zeros(avec.shape)
    for idx, a in enumerate(avec):
        #if a<1e-4:
        #    continue
        Hc = np.sqrt(Om/a+Ol*a*a+Or/a/a)
        I = scipy.integrate.quad(Dintegrand,1e-15,a,args=())
        D[idx] = Hc/a*I[0]
    D = D/scipy.integrate.quad(Dintegrand,1e-15,1,args=())[0]
    return D

def D_linder(avec,csm):
    bg = csm.get_background()
    if '(.)rho_lambda' in bg:
        Ol = bg['(.)rho_lambda'][-1]/bg['(.)rho_crit'][-1]
        w0 = -1
        wa = 0.0
    else:
        Ol = bg['(.)rho_fld'][-1]/bg['(.)rho_crit'][-1]
        w0 = csm.pars['w0_fld']
        wa = csm.pars['wa_fld']
        
    Om_of_a = (bg['(.)rho_cdm']+bg['(.)rho_b'])/bg['H [1/Mpc]']**2
    gamma = 0.55+0.05*(w0+0.5*wa)
    a_bg = 1./(1.+bg['z'])
    
    integ = (Om_of_a**gamma-1.)/a_bg
    
    integ_interp = interpolate.interp1d(a_bg,integ)
    D = np.zeros(avec.shape)
    amin = min(a_bg)
    amin = 1e-3
    for idx, a in enumerate(avec):
        if a<amin:
            D[idx] = a
        else:
            I = scipy.integrate.quad(integ_interp,amin,a,args=())
            D[idx] = a*np.exp(I[0])
#    D = D/scipy.integrate.quad(Dintegrand,1e-15,1,args=())[0]
    return D

def D_linder2(avec,csm):
    bg = csm.get_background()
    if '(.)rho_lambda' in bg:
        Ol = bg['(.)rho_lambda'][-1]/bg['(.)rho_crit'][-1]
        w0 = -1
        wa = 0.0
        rho_de = bg['(.)rho_lambda']
    else:
        Ol = bg['(.)rho_fld'][-1]/bg['(.)rho_crit'][-1]
        w0 = csm.pars['w0_fld']
        wa = csm.pars['wa_fld']
        rho_de = bg['(.)rho_fld']
        
    rho_M = bg['(.)rho_cdm']+bg['(.)rho_b']
    #Om_of_a = rho_M/bg['H [1/Mpc]']**2
    
    Om_of_a = rho_M/(rho_M+rho_de)
    gamma = 0.55+0.05*(1+w0+0.5*wa)
    #a_bg = 1./(1.+bg['z'])
    a_bg = avec
    integ = (Om_of_a**gamma-1.)/a_bg
    D = np.zeros(avec.shape)
    for idx, a in enumerate(avec):
        if idx<2:
            I=0
        else:
            I = np.trapz(integ[:idx],x=avec[:idx])
        D[idx] = a*np.exp(I)
#    D = D/scipy.integrate.quad(Dintegrand,1e-15,1,args=())[0]
    return D/D[-1]
    
    
def draw_vertical_redshift(csm, theaxis, var='tau',z=99,ls='-.',label='$z=99$'):
    if var=='z':
        xval = z
    elif var=='a':
        xval = 1./(z+1)
    elif var=='tau':
        bg = csm.get_background()
        f = interpolate.interp1d(bg['z'],bg['conf. time [Mpc]'])
        xval = f(z)
    theaxis.axvline(xval,lw=1,ls=ls,color='k',label=label)



# In[ ]:


figwidth1 = 4.4 #=0.7*6.3
figwidth2 = 6.3
figwidth15 = 0.5*(figwidth1+figwidth2)
ratio = 8.3/11.7
figheight1 = figwidth1*ratio
figheight2 = figwidth2*ratio
figheight15 = figwidth15*ratio

lw=2
fs=12
labelfs=16

fig, (ax1, ax2) = plt.subplots(2,1,figsize=(1.2*figwidth1,figheight1/(3./5.)),sharex=True,
                              gridspec_kw = {'height_ratios':[3, 2]})

if False:
    aminexp = -13
    amin = 10**aminexp
    ymin = 10**(aminexp/2.)
    ymax = 10**(-aminexp/2.)
elif False:
    aminexp = -7
    amin = 10**aminexp
    ymin = 10**(aminexp)
    ymax = 10**(-aminexp)
else:
    aminexp = -4
    amin = 10**aminexp
    ymin = 10**(aminexp-1)
    ymax = 10**(-aminexp+1)
    

bg = cosmo['LCDM'].get_background()

a = 1./(bg['z']+1)
H = bg['H [1/Mpc]']
D = bg['gr.fac. D']
f = bg['gr.fac. f']

ax1.loglog(a,D,lw=lw,label=r'$D_+^\mathrm{approx}$')
ax1.loglog(a,D_hypergeom(a,cosmo['LCDM']),lw=lw,label=r'$D_+^\mathrm{analytic}$')

ax1.loglog(a,a*ymax,'k--',lw=lw,label=r'$\propto a$')
ax1.loglog(a,1./a*ymin,'k:',lw=lw,label=r'$\propto a^{-1}$')

ax2.semilogx(a,D/D_hypergeom(a,cosmo['LCDM']),lw=lw,label=r'$D_+/D_+^\mathrm{analytic}$')
#ax2.semilogx(a,grow/grow[-1]/D_integral(a,cosmo['CDM']),'--',lw=5)
ax2.semilogx(a,f/f_hypergeom(a,cosmo['LCDM']),lw=lw,label=r'$f/f^{\,\mathrm{analytic}}$')


draw_vertical_redshift(cosmo['LCDM'], ax1, var='a',z=99,label='$z=99$')
draw_vertical_redshift(cosmo['LCDM'], ax1, var='a',z=49,label='$z=49$',ls='-')
draw_vertical_redshift(cosmo['LCDM'], ax2, var='a',z=99,label=None)
draw_vertical_redshift(cosmo['LCDM'], ax2, var='a',z=49,label=None,ls='-')

lgd1 = ax1.legend(fontsize=fs,ncol=1,loc='upper left',
           bbox_to_anchor=(1.02, 1.035))

#lgd2 = ax2.legend([r'$D_+/D_+^\mathrm{analytic}$','$z=99$'],
#           fontsize=fs,ncol=1,loc='upper left',
#           bbox_to_anchor=(1.0, 1.08))
lgd2 = ax2.legend(fontsize=fs,ncol=1,loc='upper left',
           bbox_to_anchor=(1.02, 0.83))

ax1.set_xlim([10**aminexp,1]) 
ax2.set_xlabel(r'$a$',fontsize=fs)
ax1.set_ylim([ymin,ymax])
ax2.set_ylim([0.9,1.099])

ax2.axhline(1,color='k')


fig.tight_layout()
fig.subplots_adjust(hspace=0.0)
fig.savefig('NewtonianGrowthFactor.pdf',bbox_extra_artists=(lgd1,lgd2), bbox_inches='tight')


# In[ ]:


lw=2
fs=14
fig, (ax1, ax2) = plt.subplots(2,1,figsize=(6,8),sharex=True,)
#                              gridspec_kw = {'height_ratios':[2, 1]})
for M, csm in iter(cosmo.items()):
    if M!='LCDM':
        w0, wa = M.strip('()').split(',')
        if float(wa)!=0.0:
            continue
    bg = csm.get_background()
    a = 1./(bg['z']+1)
    H = bg['H [1/Mpc]']
    #grow = bg['grow']
    #grow_prime = bg['grow_prime']
    D = bg['gr.fac. D']
    f = bg['gr.fac. f']
    #grow_interp = interpolate.interp1d(a,grow)
    #p = ax1.semilogx(a,grow/grow[-1]/a,lw=lw,label=M)
    #colour = p[0].get_color()
    
    p=ax1.semilogx(a,D_linder2(a,csm)/a,lw=lw,ls='--',label=M)
    colour = p[0].get_color()
    ax1.semilogx(a,D/a,lw=lw,ls='-',color=colour)
    ax1.semilogx(a,D_hypergeom(a,csm)/a,lw=lw,ls=':',color=colour)
    
    ax2.semilogx(a,D/D_integral2(a,csm),lw=lw,ls='-',color=colour)
    ax2.semilogx(a,D/D_hypergeom(a,csm),lw=lw,ls=':',color=colour)
    ax2.semilogx(a,D/D_linder2(a,csm),lw=lw,ls='--',color=colour)

ax1.set_xlim([1e-3,1]) 
ax2.set_xlabel(r'$a$',fontsize=fs)
ax1.set_ylim([0,2])
ax2.set_ylim([0.9,1.3])

lgd1 = ax1.legend(fontsize=fs,ncol=1,loc='lower left')
#           bbox_to_anchor=(1.0, 1.035))

fig.tight_layout()
fig.subplots_adjust(hspace=0.0)
fig.savefig('Growthrate_w0.pdf')


```

## base_2015_plikHM_TT_lowTEB_lensing.ini

```
# *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
# *  CLASS input parameter file  *
# *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*

# Best fit parameters from Planck 2015
# Case 2.59 of:
# https://wiki.cosmos.esa.int/planckpla2015/images/f/f7/Baseline_params_table_2015_limit68.pdf
# (but with more significant digits, directly from the chains)

#----------------------------
#----> background parameters:
#----------------------------

H0 = 67.86682
omega_b = 0.02227716
N_ur = 2.046
omega_cdm = 0.1184293
N_ncdm = 1
m_ncdm = 0.06
T_ncdm = 0.7137658555036082 # (4/11)^(1/3)

#--------------------------------
#----> thermodynamics parameters:
#--------------------------------

YHe = 0.245352
tau_reio = 0.06664549

#-------------------------------------
#----> primordial spectrum parameters:
#-------------------------------------

n_s = 0.9682903
A_s = 2.140509e-09

#-----------------------------
#----> non linear corrections:
#-----------------------------

non linear = halofit

#----------------------------------------
#----> parameters controlling the output:
#----------------------------------------

output = tCl,pCl,lCl,mPk
lensing = yes

root = output/base_2015_plikHM_TT_lowTEB_lensing

write warnings = yes
write parameters = yes

input_verbose = 1
background_verbose = 1
thermodynamics_verbose = 1
perturbations_verbose = 1
transfer_verbose = 1
primordial_verbose = 1
harmonic_verbose = 1
fourier_verbose = 1
lensing_verbose = 1
output_verbose = 1
```

## base_2018_plikHM_TTTEEE_lowl_lowE_lensing.ini

```
# *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
# *  CLASS input parameter file  *
# *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*

# Best fit parameters from Planck 2018
# Case 2.17 of:
# https://wiki.cosmos.esa.int/planck-legacy-archive/images/b/be/Baseline_params_table_2018_68pc.pdf
# (but with more significant digits, directly from the chains)

#----------------------------
#----> background parameters:
#----------------------------

H0 = 67.32117
omega_b = 0.02238280
N_ur = 2.046
omega_cdm = 0.1201075
N_ncdm = 1
m_ncdm = 0.06
T_ncdm = 0.7137658555036082 # (4/11)^(1/3)

#--------------------------------
#----> thermodynamics parameters:
#--------------------------------

YHe = 0.2454006
tau_reio = 0.05430842

#-------------------------------------
#----> primordial spectrum parameters:
#-------------------------------------

n_s = 0.9660499
A_s = 2.100549e-09

#-----------------------------
#----> non linear corrections:
#-----------------------------

non linear = halofit

#----------------------------------------
#----> parameters controlling the output:
#----------------------------------------

output = tCl,pCl,lCl,mPk
lensing = yes

root = output/base_2018_plikHM_TTTEEE_lowl_lowE_lensing

write warnings = yes
write parameters = yes

input_verbose = 1
background_verbose = 1
thermodynamics_verbose = 1
perturbations_verbose = 1
transfer_verbose = 1
primordial_verbose = 1
harmonic_verbose = 1
fourier_verbose = 1
lensing_verbose = 1
output_verbose = 1
```

## check_PPF_approx.py

```python
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from classy import Class


# In[ ]:


k_out = [5e-5, 5e-4, 5e-3]
models = ['PPF1','PPF2','FLD1','FLD1S']
w0 = {'PPF1':-0.7,'PPF2':-1.15,'FLD1':-0.7,'FLD1S':-0.7}
wa = {'PPF1':0.,'PPF2':0.5,'FLD1':0.,'FLD1S':0.}
omega_cdm = {'PPF1':0.104976,'PPF2':0.120376,'FLD1':0.104976,'FLD1S':0.104976}
omega_b = 0.022
##Omega_cdm = {'PPF1':0.26,'PPF2':0.21,'FLD1':0.26,'FLD1S':0.26}
##Omega_b = 0.05
h = {'PPF1':0.64,'PPF2':0.74,'FLD1':0.64,'FLD1S':0.64}
cosmo = {}

for M in models:
    use_ppf = 'yes'
    gauge = 'Newtonian'
    if 'FLD' in M:
        use_ppf = 'no'
    if 'S' in M:
        gauge = 'Synchronous'
        
    cosmo[M] = Class()
    
    cosmo[M].set({'output':'tCl mPk dTk vTk','k_output_values':str(k_out).strip('[]'),
                  'h':h[M],
                  'omega_b':omega_b,'omega_cdm':omega_cdm[M],
                  ##'Omega_b':Omega_b,'omega_cdm':Omega_cdm[M],
                  'cs2_fld':1.,
          'w0_fld':w0[M],'wa_fld':wa[M],'Omega_Lambda':0.,'gauge':gauge,
                 'use_ppf':use_ppf})
    cosmo[M].compute()


# In[ ]:


colours = ['r','k','g','m']
for i,M in enumerate(models):
    cl = cosmo[M].raw_cl()
    l = cl['ell']
    
    plt.loglog(l,cl['tt']*l*(l+1)/(2.*np.pi),label=M,color=colours[i])
    
plt.legend(loc='upper left')
plt.xlim([2,300])
plt.ylim([6e-11,1e-9])
plt.xlabel(r'$\ell$')
plt.ylabel(r'$[\ell(\ell+1)/2\pi]  C_\ell^\mathrm{TT}$')

plt.savefig('check_PPF_clTT.pdf')


# In[ ]:


for M in ['PPF1','FLD1']:
    csm = cosmo[M]
    pt = csm.get_perturbations()
    pts = pt['scalar']
    for i,k in enumerate(k_out):
        ptk = pts[i]
        a = ptk['a']
        phi = ptk['phi']
        psi = ptk['psi']
        if 'FLD' in M:
            ls = ':'
            lw=5
        else:
            ls = '-'
            lw=1
        plt.semilogx(a,0.5*(phi+psi),label=M+' '+'$k='+str(k)+'Mpc^{-1}$',ls=ls,lw=lw)
        
plt.legend(loc='lower left')
plt.xlim([1e-2,1])
plt.ylim([0.3,0.63])
plt.xlabel(r'$a/a_0$')
plt.ylabel(r'$\frac{1}{2} ~(\Phi+\Psi)$')

plt.savefig('check_PPF_metric.pdf')


# In[ ]:


#kminclosed = sqrt(-8*Omega_k)*(70/3e5) Mpc^(-1)

k_out = [1e-3] #[1e-4, 1e-3, 1e-2]
#models = ['PPF1','PPF2','FLD1']
models = ['PPF1','FLD1']
w0 = {'PPF1':-0.7,'PPF2':-1.15,'FLD1':-0.7,'FLD1S':-0.7}
wa = {'PPF1':0.,'PPF2':0.5,'FLD1':0.,'FLD1S':0.}
omega_cdm = {'PPF1':0.104976,'PPF2':0.120376,'FLD1':0.104976,'FLD1S':0.104976}
omega_b = 0.022
##Omega_cdm = {'PPF1':0.26,'PPF2':0.21,'FLD1':0.26,'FLD1S':0.26}
##Omega_b = 0.05
h = {'PPF1':0.64,'PPF2':0.74,'FLD1':0.64}

fig, axes = plt.subplots(1,2,figsize=(16,5))
for Omega_K in [-0.1, 0.0, 0.1]:
    for gauge in ['Synchronous','Newtonian']:
        cosmo = {}
        for M in models:
            use_ppf = 'yes'
            if 'FLD' in M:
                use_ppf = 'no'
        
            cosmo[M] = Class()
    
            cosmo[M].set({'output':'tCl mPk dTk vTk','k_output_values':str(k_out).strip('[]'),
                  'h':h[M],
                  'omega_b':omega_b,'omega_cdm':omega_cdm[M],'Omega_k':Omega_K,
                  ##'Omega_b':Omega_b,'omega_cdm':Omega_cdm[M],
                  'cs2_fld':1.,
          'w0_fld':w0[M],'wa_fld':wa[M],'Omega_Lambda':0.,'gauge':gauge,
                 'use_ppf':use_ppf,'hyper_sampling_curved_low_nu':10.0})
            cosmo[M].compute()
            
        label = r'$\Omega_k='+str(Omega_K)+'$, '+gauge[0]
        clfld = cosmo['FLD1'].raw_cl()
        clppf = cosmo['PPF1'].raw_cl()
        
        axes[0].semilogx(clfld['ell'][2:],clppf['tt'][2:]/clfld['tt'][2:],label=label)
        
        ptfld = cosmo['FLD1'].get_perturbations()['scalar']
        ptppf = cosmo['PPF1'].get_perturbations()['scalar']
        for i,k in enumerate(k_out):
            ptkfld = ptfld[i]
            a = ptkfld['a']
            phi_plus_phi_fld = ptkfld['phi']+ptkfld['psi']
            ptkppf = ptppf[i]
            phi_plus_phi_ppf = ptkppf['phi']+ptkppf['psi']
            axes[1].semilogx(ptkppf['a'],phi_plus_phi_ppf,label=label+'_ppf')
            axes[1].semilogx(ptkfld['a'],phi_plus_phi_fld,label=label+'_fld')
            print (len(ptkppf['a']),len(ptkfld['a']))
            
axes[0].legend(loc='lower left',ncol=2)
axes[0].set_xlim([2,300])
axes[0].set_ylim([0.98,1.02])
axes[0].set_xlabel(r'$\ell$')
axes[0].set_ylabel(r'$C_\ell^\mathrm{FLD1}/C_\ell^\mathrm{PPF1}$')

axes[1].legend(loc='lower left',ncol=2)
axes[1].set_xlim([1e-2,1])
axes[1].set_xlabel(r'$a/a_0$')
axes[1].set_ylabel(r'$(\Phi+\Psi)$')

fig.savefig('check_PPF_Omegak.pdf')        


# In[ ]:


colours = ['r','k','g','m']

k_out = [1e-1] #[1e-4, 1e-3, 1e-2]
#models = ['PPF1','PPF2','FLD1']
models = ['PPF1','FLD1']
w0 = {'PPF1':-0.7,'PPF2':-1.15,'FLD1':-0.7,'FLD1S':-0.7}
wa = {'PPF1':0.,'PPF2':0.5,'FLD1':0.,'FLD1S':0.}
omega_cdm = {'PPF1':0.104976,'PPF2':0.120376,'FLD1':0.104976,'FLD1S':0.104976}
omega_b = 0.022
##Omega_cdm = {'PPF1':0.26,'PPF2':0.21,'FLD1':0.26,'FLD1S':0.26}
##Omega_b = 0.05
h = {'PPF1':0.64,'PPF2':0.74,'FLD1':0.64}

fig, axes = plt.subplots(1,2,figsize=(18,8))

for Omega_K in [-0.1, 0.0, 0.1]:
    for ppfgauge in ['Synchronous','Newtonian']:
        cosmo = {}
        for M in models:
            use_ppf = 'yes'
            gauge = ppfgauge
            if 'FLD' in M:
                use_ppf = 'no'
        
            cosmo[M] = Class()
    
            cosmo[M].set({'output':'tCl mPk dTk vTk','k_output_values':str(k_out).strip('[]'),
                  'h':h[M],
                  'omega_b':omega_b,'omega_cdm':omega_cdm[M],'Omega_k':Omega_K,
                  ##'Omega_b':Omega_b,'omega_cdm':Omega_cdm[M],
                  'cs2_fld':1.,
          'w0_fld':w0[M],'wa_fld':wa[M],'Omega_Lambda':0.,'gauge':gauge,
                 'use_ppf':use_ppf,'hyper_sampling_curved_low_nu':6.1})
            cosmo[M].compute()
            
        #fig, axes = plt.subplots(1,2,figsize=(16,5))
        for j,M in enumerate(models):
            cl = cosmo[M].raw_cl()
            l = cl['ell']
            label = M+r'$\Omega_k='+str(Omega_K)+'$, '+gauge[0]
            axes[0].loglog(l,cl['tt']*l*(l+1)/(2.*np.pi),label=label,color=colours[j])
        
            csm = cosmo[M]
            pt = csm.get_perturbations()
            pts = pt['scalar']
            for i,k in enumerate(k_out):
                ptk = pts[i]
                a = ptk['a']
                phi = ptk['phi']
                psi = ptk['psi']
                if 'FLD' in M:
                    ls = ':'
                    lw=5
                else:
                    ls = '-'
                    lw=1
                axes[1].semilogx(a,0.5*abs(phi+psi),label=label+' '+'$k='+str(k)+'Mpc^{-1}$',ls=ls,lw=lw)

axes[0].legend(loc='upper left')
axes[0].set_xlim([2,300])
axes[0].set_ylim([6e-11,1e-9])
axes[0].set_xlabel(r'$\ell$')
axes[0].set_ylabel(r'$[\ell(\ell+1)/2\pi]  C_\ell^\mathrm{TT}$')

axes[1].legend(loc='upper right')
#axes[1].set_xlim([1e-2,1])
#axes[1].set_ylim([0.3,0.63])
axes[1].set_xlabel(r'$a/a_0$')
axes[1].set_ylabel(r'$\frac{1}{2}~(\Phi+\Psi)$')

fig.savefig('check_PPF_Omegak2.pdf')               


```

## cl_ST.py

```python
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import necessary modules
from classy import Class
from math import pi
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt


# In[ ]:


#####################################################
#
# Cosmological parameters and other CLASS parameters
#
#####################################################
common_settings = {# LambdaCDM parameters
                   'h':0.67810,
                   'omega_b':0.02238280,
                   'omega_cdm': 0.1201075,
                   'A_s':2.100549e-09,
                   'tau_reio': 0.05430842}

l_max_scalars = 3000
l_max_tensors = 600

# Note that for l_max_tensors =600 we can keep default precision,
# while for for l_max_tensors = 3000 we would need to import many high precision settings from the file cl_ref.pre    


# In[ ]:


###############
#    
# call CLASS : scalars only
#
###############
#
M = Class()
M.set(common_settings)
M.set({'output':'tCl,pCl','modes':'s','lensing':'no','n_s':0.9660499,
       'l_max_scalars':l_max_scalars})
M.compute()
cls = M.raw_cl(l_max_scalars)


# In[ ]:


###############
#    
# call CLASS : tensors only
#
###############
#
M.empty() # reset input parameters to default, before passing a new parameter set
M.set(common_settings)
M.set({'output':'tCl,pCl','modes':'t','lensing':'no','r':0.1,'n_t':0,
       'l_max_tensors':l_max_tensors})
M.compute()
clt = M.raw_cl(l_max_tensors)


# In[ ]:


###############
#    
# call CLASS : scalars + tensors (only in this case we can get the correct lensed ClBB)
#
###############
#
M.empty() # reset input parameters to default, before passing a new parameter set
M.set(common_settings)
M.set({'output':'tCl,pCl,lCl','modes':'s,t','lensing':'yes','n_s':0.9660499,'r':0.1,'n_t':0,
       'l_max_scalars':l_max_scalars,'l_max_tensors':l_max_tensors})
M.compute()
cl_tot = M.raw_cl(l_max_scalars)
cl_lensed = M.lensed_cl(l_max_scalars)


# In[ ]:


#################
#
# plotting
#
#################
#
plt.xlim([2,l_max_scalars])
plt.ylim([1.e-8,10])
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\ell (\ell+1) C_l^{XY} / 2 \pi \,\,\, [\times 10^{10}]$")
plt.title(r"$r=0.1$")
plt.grid()
#
ell = cl_tot['ell']
ellt = clt['ell']
factor = 1.e10*ell*(ell+1.)/2./pi
factort = 1.e10*ellt*(ellt+1.)/2./pi
#
plt.loglog(ell,factor*cls['tt'],'r-',label=r'$\mathrm{TT(s)}$')
plt.loglog(ellt,factort*clt['tt'],'r:',label=r'$\mathrm{TT(t)}$')
plt.loglog(ell,factor*cls['ee'],'b-',label=r'$\mathrm{EE(s)}$')
plt.loglog(ellt,factort*clt['ee'],'b:',label=r'$\mathrm{EE(t)}$')
plt.loglog(ellt,factort*clt['bb'],'g:',label=r'$\mathrm{BB(t)}$')
plt.loglog(ell,factor*(cl_lensed['bb']-cl_tot['bb']),'g-',label=r'$\mathrm{BB(lensing)}$')
plt.legend(loc='right',bbox_to_anchor=(1.4, 0.5))
plt.savefig('cl_ST.pdf',bbox_inches='tight')


```

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

## classy-py.py

```python
"""
.. module:: classy
    :synopsis: Python wrapper around CLASS
.. moduleauthor:: Karim Benabed <benabed@iap.fr>
.. moduleauthor:: Benjamin Audren <benjamin.audren@epfl.ch>
.. moduleauthor:: Julien Lesgourgues <lesgourg@cern.ch>

This module defines a class called Class. It is used with Monte Python to
extract cosmological parameters.

# JL 14.06.2017: TODO: check whether we should free somewhere the allocated fc.filename and titles, data (4 times)

"""
from math import exp,log
import numpy as np
cimport numpy as np
from libc.stdlib cimport *
from libc.stdio cimport *
from libc.string cimport *
import cython
cimport cython
from scipy.interpolate import CubicSpline
from scipy.interpolate import UnivariateSpline
from scipy.interpolate import interp1d

# Nils : Added for python 3.x and python 2.x compatibility
import sys
def viewdictitems(d):
    if sys.version_info >= (3,0):
        return d.items()
    else:
        return d.viewitems()

ctypedef np.float64_t DTYPE_t
ctypedef np.int32_t DTYPE_i



# Import the .pxd containing definitions
from cclassy cimport *

__version__ = _VERSION_.decode("utf-8")

# Implement a specific Exception (this might not be optimally designed, nor
# even acceptable for python standards. It, however, does the job).
# The idea is to raise either an AttributeError if the problem happened while
# reading the parameters (in the normal Class, this would just return a line in
# the unused_parameters file), or a NameError in other cases. This allows
# MontePython to handle things differently.
class CosmoError(Exception):
    def __init__(self, message=""):
        self.message = message.decode() if isinstance(message,bytes) else message

    def __str__(self):
        return '\n\nError in Class: ' + self.message


class CosmoSevereError(CosmoError):
    """
    Raised when Class failed to understand one or more input parameters.

    This case would not raise any problem in Class default behaviour. However,
    for parameter extraction, one has to be sure that all input parameters were
    understood, otherwise the wrong cosmological model would be selected.
    """
    pass


class CosmoComputationError(CosmoError):
    """
    Raised when Class could not compute the cosmology at this point.

    This will be caught by the parameter extraction code to give an extremely
    unlikely value to this point
    """
    pass


cdef class Class:
    """
    Class wrapping, creates the glue between C and python

    The actual Class wrapping, the only class we will call from MontePython
    (indeed the only one we will import, with the command:
    from classy import Class

    """
    # List of used structures, defined in the header file. They have to be
    # "cdefined", because they correspond to C structures
    cdef precision pr
    cdef background ba
    cdef thermodynamics th
    cdef perturbations pt
    cdef primordial pm
    cdef fourier fo
    cdef transfer tr
    cdef harmonic hr
    cdef output op
    cdef lensing le
    cdef distortions sd
    cdef file_content fc

    cdef int computed # Flag to see if classy has already computed with the given pars
    cdef int allocated # Flag to see if classy structs are allocated already
    cdef object _pars # Dictionary of the parameters
    cdef object ncp   # Keeps track of the structures initialized, in view of cleaning.

    _levellist = ["input","background","thermodynamics","perturbations", "primordial", "fourier", "transfer", "harmonic", "lensing", "distortions"]

    # Defining two new properties to recover, respectively, the parameters used
    # or the age (set after computation). Follow this syntax if you want to
    # access other quantities. Alternatively, you can also define a method, and
    # call it (see _T_cmb method, at the very bottom).
    property pars:
        def __get__(self):
            return self._pars
    property state:
        def __get__(self):
            return True
    property Omega_nu:
        def __get__(self):
            return self.ba.Omega0_ncdm_tot
    property nonlinear_method:
        def __get__(self):
            return self.fo.method

    def set_default(self):
        _pars = {
            "output":"tCl mPk",}
        self.set(**_pars)

    def __cinit__(self, default=False):
        cdef char* dumc
        self.allocated = False
        self.computed = False
        self._pars = {}
        self.fc.size=0
        self.fc.filename = <char*>malloc(sizeof(char)*30)
        assert(self.fc.filename!=NULL)
        dumc = "NOFILE"
        sprintf(self.fc.filename,"%s",dumc)
        self.ncp = set()
        if default: self.set_default()

    def __dealloc__(self):
        if self.allocated:
          self.struct_cleanup()
        self.empty()
        # Reset all the fc to zero if its not already done
        if self.fc.size !=0:
            self.fc.size=0
            free(self.fc.name)
            free(self.fc.value)
            free(self.fc.read)
            free(self.fc.filename)

    # Set up the dictionary
    def set(self,*pars,**kars):
        oldpars = self._pars.copy()
        if len(pars)==1:
            self._pars.update(dict(pars[0]))
        elif len(pars)!=0:
            raise CosmoSevereError("bad call")
        self._pars.update(kars)
        if viewdictitems(self._pars) <= viewdictitems(oldpars):
          return # Don't change the computed states, if the new dict was already contained in the previous dict
        self.computed=False
        return True

    def empty(self):
        self._pars = {}
        self.computed = False

    # Create an equivalent of the parameter file. Non specified values will be
    # taken at their default (in Class)
    def _fillparfile(self):
        cdef char* dumc

        if self.fc.size!=0:
            free(self.fc.name)
            free(self.fc.value)
            free(self.fc.read)
        self.fc.size = len(self._pars)
        self.fc.name = <FileArg*> malloc(sizeof(FileArg)*len(self._pars))
        assert(self.fc.name!=NULL)

        self.fc.value = <FileArg*> malloc(sizeof(FileArg)*len(self._pars))
        assert(self.fc.value!=NULL)

        self.fc.read = <short*> malloc(sizeof(short)*len(self._pars))
        assert(self.fc.read!=NULL)

        # fill parameter file
        i = 0
        for kk in self._pars:

            dumcp = kk.strip().encode()
            dumc = dumcp
            sprintf(self.fc.name[i],"%s",dumc)
            dumcp = str(self._pars[kk]).strip().encode()
            dumc = dumcp
            sprintf(self.fc.value[i],"%s",dumc)
            self.fc.read[i] = _FALSE_
            i+=1

    # Called at the end of a run, to free memory
    def struct_cleanup(self):
        if(self.allocated != True):
          return
        if self.sd.is_allocated:
            distortions_free(&self.sd)
        if self.le.is_allocated:
            lensing_free(&self.le)
        if self.hr.is_allocated:
            harmonic_free(&self.hr)
        if self.tr.is_allocated:
            transfer_free(&self.tr)
        if self.fo.is_allocated:
            fourier_free(&self.fo)
        if self.pm.is_allocated:
            primordial_free(&self.pm)
        if self.pt.is_allocated:
            perturbations_free(&self.pt)
        if self.th.is_allocated:
            thermodynamics_free(&self.th)
        if self.ba.is_allocated:
            background_free(&self.ba)
        self.ncp = set()

        self.allocated = False
        self.computed = False

    def _check_task_dependency(self, level):
        """
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

        """
        # If it's a string only, treat as a list
        if isinstance(level, str):
          level=[level]
        # For each item in the list
        levelset = set()
        for item in level:
          # If the item is not in the list of allowed levels, make error message
          if item not in self._levellist:
            raise CosmoSevereError("Unknown computation level: '{}'".format(item))
          # Otherwise, add to list of levels up to and including the specified level
          levelset.update(self._levellist[:self._levellist.index(item)+1])
        return levelset

    def _pars_check(self, key, value, contains=False, add=""):
        val = ""
        if key in self._pars:
            val = self._pars[key]
            if contains:
                if value in val:
                    return True
            else:
                if value==val:
                    return True
        if add:
            sep = " "
            if isinstance(add,str):
                sep = add

            if contains and val:
                    self.set({key:val+sep+value})
            else:
                self.set({key:value})
            return True
        return False

    def compute(self, level=["distortions"]):
        """
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

        """
        cdef ErrorMsg errmsg

        # Append to the list level all the modules necessary to compute.
        level = self._check_task_dependency(level)

        # Check if this function ran before (self.computed should be true), and
        # if no other modules were requested, i.e. if self.ncp contains (or is
        # equivalent to) level. If it is the case, simply stop the execution of
        # the function.
        if self.computed and self.ncp.issuperset(level):
            return

        # Check if already allocated to prevent memory leaks
        if self.allocated:
            self.struct_cleanup()

        # Otherwise, proceed with the normal computation.
        self.computed = False

        # Equivalent of writing a parameter file
        self._fillparfile()

        # self.ncp will contain the list of computed modules (under the form of
        # a set, instead of a python list)
        self.ncp=set()
        # Up until the empty set, all modules are allocated
        # (And then we successively keep track of the ones we allocate additionally)
        self.allocated = True

        # --------------------------------------------------------------------
        # Check the presence for all CLASS modules in the list 'level'. If a
        # module is found in level, executure its "_init" method.
        # --------------------------------------------------------------------
        # The input module should raise a CosmoSevereError, because
        # non-understood parameters asked to the wrapper is a problematic
        # situation.
        if "input" in level:
            if input_read_from_file(&self.fc, &self.pr, &self.ba, &self.th,
                                    &self.pt, &self.tr, &self.pm, &self.hr,
                                    &self.fo, &self.le, &self.sd, &self.op, errmsg) == _FAILURE_:
                raise CosmoSevereError(errmsg)
            self.ncp.add("input")
            # This part is done to list all the unread parameters, for debugging
            problem_flag = False
            problematic_parameters = []
            for i in range(self.fc.size):
                if self.fc.read[i] == _FALSE_:
                    problem_flag = True
                    problematic_parameters.append(self.fc.name[i].decode())
            if problem_flag:
                raise CosmoSevereError(
                    "Class did not read input parameter(s): %s\n" % ', '.join(
                    problematic_parameters))

        # The following list of computation is straightforward. If the "_init"
        # methods fail, call `struct_cleanup` and raise a CosmoComputationError
        # with the error message from the faulty module of CLASS.
        if "background" in level:
            if background_init(&(self.pr), &(self.ba)) == _FAILURE_:
                self.struct_cleanup()
                raise CosmoComputationError(self.ba.error_message)
            self.ncp.add("background")

        if "thermodynamics" in level:
            if thermodynamics_init(&(self.pr), &(self.ba),
                                   &(self.th)) == _FAILURE_:
                self.struct_cleanup()
                raise CosmoComputationError(self.th.error_message)
            self.ncp.add("thermodynamics")

        if "perturbations" in level:
            if perturbations_init(&(self.pr), &(self.ba),
                            &(self.th), &(self.pt)) == _FAILURE_:
                self.struct_cleanup()
                raise CosmoComputationError(self.pt.error_message)
            self.ncp.add("perturbations")

        if "primordial" in level:
            if primordial_init(&(self.pr), &(self.pt),
                               &(self.pm)) == _FAILURE_:
                self.struct_cleanup()
                raise CosmoComputationError(self.pm.error_message)
            self.ncp.add("primordial")

        if "fourier" in level:
            if fourier_init(&self.pr, &self.ba, &self.th,
                              &self.pt, &self.pm, &self.fo) == _FAILURE_:
                self.struct_cleanup()
                raise CosmoComputationError(self.fo.error_message)
            self.ncp.add("fourier")

        if "transfer" in level:
            if transfer_init(&(self.pr), &(self.ba), &(self.th),
                             &(self.pt), &(self.fo), &(self.tr)) == _FAILURE_:
                self.struct_cleanup()
                raise CosmoComputationError(self.tr.error_message)
            self.ncp.add("transfer")

        if "harmonic" in level:
            if harmonic_init(&(self.pr), &(self.ba), &(self.pt),
                            &(self.pm), &(self.fo), &(self.tr),
                            &(self.hr)) == _FAILURE_:
                self.struct_cleanup()
                raise CosmoComputationError(self.hr.error_message)
            self.ncp.add("harmonic")

        if "lensing" in level:
            if lensing_init(&(self.pr), &(self.pt), &(self.hr),
                            &(self.fo), &(self.le)) == _FAILURE_:
                self.struct_cleanup()
                raise CosmoComputationError(self.le.error_message)
            self.ncp.add("lensing")

        if "distortions" in level:
            if distortions_init(&(self.pr), &(self.ba), &(self.th),
                                &(self.pt), &(self.pm), &(self.sd)) == _FAILURE_:
                self.struct_cleanup()
                raise CosmoComputationError(self.sd.error_message)
            self.ncp.add("distortions")

        self.computed = True

        # At this point, the cosmological instance contains everything needed. The
        # following functions are only to output the desired numbers
        return

    def set_baseline(self, baseline_name):
        # Taken from montepython [https://github.com/brinckmann/montepython_public] (see also 1210.7183, 1804.07261)
        if ('planck' in baseline_name and '18' in baseline_name and 'lens' in baseline_name and 'bao' in baseline_name) or 'p18lb' in baseline_name.lower():
          self.set({'omega_b':2.255065e-02,
                    'omega_cdm':1.193524e-01,
                    'H0':6.776953e+01,
                    'A_s':2.123257e-09,
                    'n_s':9.686025e-01,
                    'z_reio':8.227371e+00,

                    'N_ur':2.0328,
                    'N_ncdm':1,
                    'm_ncdm':0.06,
                    'T_ncdm':0.71611,

                    'output':'mPk, tCl, pCl, lCl',
                    'lensing':'yes',
                    'P_k_max_h/Mpc':1.0,
                    'non_linear':'halofit'
                    })

        elif ('planck' in baseline_name and '18' in baseline_name and 'lens' in baseline_name) or 'p18l' in baseline_name.lower():
          self.set({'omega_b':2.236219e-02,
                    'omega_cdm':1.201668e-01,
                    'H0':6.726996e+01,
                    'A_s':2.102880e-09,
                    'n_s':9.661489e-01,
                    'z_reio':7.743057e+00,

                    'N_ur':2.0328,
                    'N_ncdm':1,
                    'm_ncdm':0.06,
                    'T_ncdm':0.71611,

                    'output':'mPk, tCl, pCl, lCl',
                    'lensing':'yes',
                    'P_k_max_h/Mpc':1.0,
                    'non_linear':'halofit'
                    })

        elif ('planck' in baseline_name and '18' in baseline_name) or 'p18' in baseline_name.lower():
          self.set({'omega_b':2.237064e-02,
                    'omega_cdm':1.214344e-01,
                    'H0':6.685836e+01,
                    'A_s':2.112203e-09,
                    'n_s':9.622800e-01,
                    'z_reio':7.795700e+00,

                    'N_ur':2.0328,
                    'N_ncdm':1,
                    'm_ncdm':0.06,
                    'T_ncdm':0.71611,

                    'output':'mPk, tCl, pCl, lCl',
                    'lensing':'yes',
                    'P_k_max_h/Mpc':1.0})
        else:
          raise CosmoSevereError("Unrecognized baseline case '{}'".format(baseline_name))

    @property
    def density_factor(self):
        """
        The density factor required to convert from the class-units of density to kg/m^3 (SI units)
        """
        return 3*_c_*_c_/(8*np.pi*_G_)/(_Mpc_over_m_*_Mpc_over_m_)

    @property
    def Mpc_to_m(self):
        return _Mpc_over_m_

    @property
    def kg_to_eV(self):
        return _c_*_c_/_eV_

    @property
    def kgm3_to_eVMpc3(self):
        """
        Convert from kg/m^3 to eV/Mpc^3
        """
        return self.kg_to_eV*self.Mpc_to_m**3

    @property
    def kg_to_Msol(self):
        return 1/(2.0e30)

    @property
    def kgm3_to_MsolMpc3(self):
        """
        Convert from kg/m^3 to Msol/Mpc^3
        """
        return self.kg_to_Msol*self.Mpc_to_m**3

    def raw_cl(self, lmax=-1, nofail=False):
        """
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
        """
        self.compute(["harmonic"])
        cdef int lmaxR

        # Define a list of integers, refering to the flags and indices of each
        # possible output Cl. It allows for a clear and concise way of looping
        # over them, checking if they are defined or not.
        has_flags = [
            (self.hr.has_tt, self.hr.index_ct_tt, 'tt'),
            (self.hr.has_ee, self.hr.index_ct_ee, 'ee'),
            (self.hr.has_te, self.hr.index_ct_te, 'te'),
            (self.hr.has_bb, self.hr.index_ct_bb, 'bb'),
            (self.hr.has_pp, self.hr.index_ct_pp, 'pp'),
            (self.hr.has_tp, self.hr.index_ct_tp, 'tp'),]
        spectra = []

        for flag, index, name in has_flags:
            if flag:
                spectra.append(name)

        # We need to be able to gracefully exit BEFORE allocating things (!)
        if not spectra:
            raise CosmoSevereError("No Cl computed")

        # We need to be able to gracefully exit BEFORE allocating things (!)
        lmaxR = self.hr.l_max_tot
        if lmax == -1:
            lmax = lmaxR
        if lmax > lmaxR:
            if nofail:
                self._pars_check("l_max_scalars",lmax)
                self.compute(["lensing"])
            else:
                raise CosmoSevereError("Can only compute up to lmax=%d"%lmaxR)

        # Now that the conditions are all checked, we can allocate and do what we want

        #temporary storage for the cls (total)
        cdef double *rcl = <double*> calloc(self.hr.ct_size,sizeof(double))

        # Quantities for tensor modes
        cdef double **cl_md = <double**> calloc(self.hr.md_size, sizeof(double*))
        for index_md in range(self.hr.md_size):
            cl_md[index_md] = <double*> calloc(self.hr.ct_size, sizeof(double))

        # Quantities for isocurvature modes
        cdef double **cl_md_ic = <double**> calloc(self.hr.md_size, sizeof(double*))
        for index_md in range(self.hr.md_size):
            cl_md_ic[index_md] = <double*> calloc(self.hr.ct_size*self.hr.ic_ic_size[index_md], sizeof(double))

        # Initialise all the needed Cls arrays
        cl = {}
        for elem in spectra:
            cl[elem] = np.zeros(lmax+1, dtype=np.double)

        success = True
        # Recover for each ell the information from CLASS
        for ell from 2<=ell<lmax+1:
            if harmonic_cl_at_l(&self.hr, ell, rcl, cl_md, cl_md_ic) == _FAILURE_:
                success = False
                break
            for flag, index, name in has_flags:
                if name in spectra:
                    cl[name][ell] = rcl[index]
        cl['ell'] = np.arange(lmax+1)

        free(rcl)
        for index_md in range(self.hr.md_size):
            free(cl_md[index_md])
            free(cl_md_ic[index_md])
        free(cl_md)
        free(cl_md_ic)

        # This has to be delayed until AFTER freeing the memory
        if not success:
          raise CosmoSevereError(self.hr.error_message)

        return cl

    def lensed_cl(self, lmax=-1,nofail=False):
        """
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
        """
        self.compute(["lensing"])
        cdef int lmaxR

        # Define a list of integers, refering to the flags and indices of each
        # possible output Cl. It allows for a clear and concise way of looping
        # over them, checking if they are defined or not.
        has_flags = [
            (self.le.has_tt, self.le.index_lt_tt, 'tt'),
            (self.le.has_ee, self.le.index_lt_ee, 'ee'),
            (self.le.has_te, self.le.index_lt_te, 'te'),
            (self.le.has_bb, self.le.index_lt_bb, 'bb'),
            (self.le.has_pp, self.le.index_lt_pp, 'pp'),
            (self.le.has_tp, self.le.index_lt_tp, 'tp'),]
        spectra = []

        for flag, index, name in has_flags:
            if flag:
                spectra.append(name)

        # We need to be able to gracefully exit BEFORE allocating things (!)
        if not spectra:
            raise CosmoSevereError("No lensed Cl computed")

        # We need to be able to gracefully exit BEFORE allocating things (!)
        lmaxR = self.le.l_lensed_max
        if lmax == -1:
            lmax = lmaxR
        if lmax > lmaxR:
            if nofail:
                self._pars_check("l_max_scalars",lmax)
                self.compute(["lensing"])
            else:
                raise CosmoSevereError("Can only compute up to lmax=%d"%lmaxR)

        # Now that the conditions are all checked, we can allocate and do what we want
        cdef double *lcl = <double*> calloc(self.le.lt_size,sizeof(double))

        cl = {}
        success = True
        # Simple Cls, for temperature and polarisation, are not so big in size
        for elem in spectra:
            cl[elem] = np.zeros(lmax+1, dtype=np.double)
        for ell from 2<=ell<lmax+1:
            if lensing_cl_at_l(&self.le,ell,lcl) == _FAILURE_:
                success = False
                break
            for flag, index, name in has_flags:
                if name in spectra:
                    cl[name][ell] = lcl[index]
        cl['ell'] = np.arange(lmax+1)

        free(lcl)

        # This has to be delayed until AFTER freeing the memory
        if not success:
          raise CosmoSevereError(self.le.error_message)

        return cl

    def density_cl(self, lmax=-1, nofail=False):
        """
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
        """
        self.compute(["harmonic"])
        cdef int lmaxR

        lmaxR = self.pt.l_lss_max
        has_flags = [
            (self.hr.has_dd, self.hr.index_ct_dd, 'dd'),
            (self.hr.has_td, self.hr.index_ct_td, 'td'),
            (self.hr.has_ll, self.hr.index_ct_ll, 'll'),
            (self.hr.has_dl, self.hr.index_ct_dl, 'dl'),
            (self.hr.has_tl, self.hr.index_ct_tl, 'tl')]
        spectra = []

        for flag, index, name in has_flags:
            if flag:
                spectra.append(name)
                l_max_flag = self.hr.l_max_ct[self.hr.index_md_scalars][index]
                if l_max_flag < lmax and lmax > 0:
                    raise CosmoSevereError(
                        "the %s spectrum was computed until l=%i " % (
                            name.upper(), l_max_flag) +
                        "but you asked a l=%i" % lmax)

        # We need to be able to gracefully exit BEFORE allocating things (!)
        if not spectra:
            raise CosmoSevereError("No density Cl computed")

        # We need to be able to gracefully exit BEFORE allocating things (!)
        if lmax == -1:
            lmax = lmaxR
        if lmax > lmaxR:
            if nofail:
                self._pars_check("l_max_lss",lmax)
                self._pars_check("output",'nCl')
                self.compute()
            else:
                raise CosmoSevereError("Can only compute up to lmax=%d"%lmaxR)

        # Now that the conditions are all checked, we can allocate and do what we want
        cdef double *dcl = <double*> calloc(self.hr.ct_size,sizeof(double))

        # Quantities for tensor modes
        cdef double **cl_md = <double**> calloc(self.hr.md_size, sizeof(double*))
        for index_md in range(self.hr.md_size):
            cl_md[index_md] = <double*> calloc(self.hr.ct_size, sizeof(double))

        # Quantities for isocurvature modes
        cdef double **cl_md_ic = <double**> calloc(self.hr.md_size, sizeof(double*))
        for index_md in range(self.hr.md_size):
            cl_md_ic[index_md] = <double*> calloc(self.hr.ct_size*self.hr.ic_ic_size[index_md], sizeof(double))

        cl = {}

        # For density Cls, we compute the names for each combination, which will also correspond to the size
        names = {'dd':[],'ll':[],'dl':[]}
        for index_d1 in range(self.hr.d_size):
          for index_d2 in range(index_d1, min(index_d1+self.hr.non_diag+1, self.hr.d_size)):
            names['dd'].append("dens[%d]-dens[%d]"%(index_d1+1, index_d2+1))
            names['ll'].append("lens[%d]-lens[%d]"%(index_d1+1, index_d2+1))
          for index_d2 in range(max(index_d1-self.hr.non_diag,0), min(index_d1+self.hr.non_diag+1, self.hr.d_size)):
            names['dl'].append("dens[%d]-lens[%d]"%(index_d1+1, index_d2+1))

        for elem in names:
            if elem in spectra:
                cl[elem] = {}
                for name in names[elem]:
                    cl[elem][name] = np.zeros(lmax+1, dtype=np.double)

        for elem in ['td', 'tl']:
            if elem in spectra:
                cl[elem] = np.zeros(lmax+1, dtype=np.double)

        success = True
        for ell from 2<=ell<lmax+1:
            if harmonic_cl_at_l(&self.hr, ell, dcl, cl_md, cl_md_ic) == _FAILURE_:
                success = False
                break
            if 'dd' in spectra:
                for index, name in enumerate(names['dd']):
                  cl['dd'][name][ell] = dcl[self.hr.index_ct_dd+index]
            if 'll' in spectra:
                for index, name in enumerate(names['ll']):
                  cl['ll'][name][ell] = dcl[self.hr.index_ct_ll+index]
            if 'dl' in spectra:
                for index, name in enumerate(names['dl']):
                  cl['dl'][name][ell] = dcl[self.hr.index_ct_dl+index]
            if 'td' in spectra:
                cl['td'][ell] = dcl[self.hr.index_ct_td]
            if 'tl' in spectra:
                cl['tl'][ell] = dcl[self.hr.index_ct_tl]
        cl['ell'] = np.arange(lmax+1)

        free(dcl)
        for index_md in range(self.hr.md_size):
            free(cl_md[index_md])
            free(cl_md_ic[index_md])
        free(cl_md)
        free(cl_md_ic)

        # This has to be delayed until AFTER freeing the memory
        if not success:
          raise CosmoSevereError(self.hr.error_message)
        return cl

    def z_of_r (self, z):
        self.compute(["background"])
        cdef int last_index=0 #junk
        cdef double * pvecback

        zarr = np.atleast_1d(z).astype(np.float64)

        r = np.zeros(len(zarr),'float64')
        dzdr = np.zeros(len(zarr),'float64')

        pvecback = <double*> calloc(self.ba.bg_size,sizeof(double))

        i = 0
        for redshift in zarr:

            if background_at_z(&self.ba,redshift,long_info,inter_normal,&last_index,pvecback)==_FAILURE_:
                free(pvecback) #manual free due to error
                raise CosmoSevereError(self.ba.error_message)

            # store r
            r[i] = pvecback[self.ba.index_bg_conf_distance]
            # store dz/dr = H
            dzdr[i] = pvecback[self.ba.index_bg_H]

            i += 1

        free(pvecback)

        return (r[0], dzdr[0]) if np.isscalar(z) else (r,dzdr)

    def luminosity_distance(self, z):
        """
        luminosity_distance(z)
        """
        self.compute(["background"])

        cdef int last_index = 0  # junk

        zarr = np.atleast_1d(z).astype(np.float64)

        pvecback = <double*> calloc(self.ba.bg_size,sizeof(double))

        lum_distance = np.empty_like(zarr)
        for iz, redshift in enumerate(zarr):
          if background_at_z(&self.ba, redshift, long_info,
                  inter_normal, &last_index, pvecback)==_FAILURE_:
              free(pvecback) #manual free due to error
              raise CosmoSevereError(self.ba.error_message)

          lum_distance[iz] = pvecback[self.ba.index_bg_lum_distance]
        free(pvecback)

        return (lum_distance[0] if np.isscalar(z) else lum_distance)

    # Gives the total matter pk for a given (k,z)
    def pk(self,double k,double z):
        """
        Gives the total matter pk (in Mpc**3) for a given k (in 1/Mpc) and z (will be non linear if requested to Class, linear otherwise)

        .. note::

            there is an additional check that output contains `mPk`,
            because otherwise a segfault will occur

        """
        self.compute(["fourier"])

        cdef double pk

        if (self.pt.has_pk_matter == _FALSE_):
            raise CosmoSevereError("No power spectrum computed. You must add mPk to the list of outputs.")

        if (self.fo.method == nl_none):
            if fourier_pk_at_k_and_z(&self.ba,&self.pm,&self.fo,pk_linear,k,z,self.fo.index_pk_m,&pk,NULL)==_FAILURE_:
                raise CosmoSevereError(self.fo.error_message)
        else:
            if fourier_pk_at_k_and_z(&self.ba,&self.pm,&self.fo,pk_nonlinear,k,z,self.fo.index_pk_m,&pk,NULL)==_FAILURE_:
                raise CosmoSevereError(self.fo.error_message)

        return pk

    # Gives the cdm+b pk for a given (k,z)
    def pk_cb(self,double k,double z):
        """
        Gives the cdm+b pk (in Mpc**3) for a given k (in 1/Mpc) and z (will be non linear if requested to Class, linear otherwise)

        .. note::

            there is an additional check that output contains `mPk`,
            because otherwise a segfault will occur

        """
        self.compute(["fourier"])

        cdef double pk_cb

        if (self.pt.has_pk_matter == _FALSE_):
            raise CosmoSevereError("No power spectrum computed. You must add mPk to the list of outputs.")
        if (self.fo.has_pk_cb == _FALSE_):
            raise CosmoSevereError("P_cb not computed (probably because there are no massive neutrinos) so you cannot ask for it")

        if (self.fo.method == nl_none):
            if fourier_pk_at_k_and_z(&self.ba,&self.pm,&self.fo,pk_linear,k,z,self.fo.index_pk_cb,&pk_cb,NULL)==_FAILURE_:
                raise CosmoSevereError(self.fo.error_message)
        else:
            if fourier_pk_at_k_and_z(&self.ba,&self.pm,&self.fo,pk_nonlinear,k,z,self.fo.index_pk_cb,&pk_cb,NULL)==_FAILURE_:
                raise CosmoSevereError(self.fo.error_message)

        return pk_cb

    # Gives the total matter pk for a given (k,z)
    def pk_lin(self,double k,double z):
        """
        Gives the linear total matter pk (in Mpc**3) for a given k (in 1/Mpc) and z

        .. note::

            there is an additional check that output contains `mPk`,
            because otherwise a segfault will occur

        """
        self.compute(["fourier"])

        cdef double pk_lin

        if (self.pt.has_pk_matter == _FALSE_):
            raise CosmoSevereError("No power spectrum computed. You must add mPk to the list of outputs.")

        if fourier_pk_at_k_and_z(&self.ba,&self.pm,&self.fo,pk_linear,k,z,self.fo.index_pk_m,&pk_lin,NULL)==_FAILURE_:
            raise CosmoSevereError(self.fo.error_message)

        return pk_lin

    # Gives the cdm+b pk for a given (k,z)
    def pk_cb_lin(self,double k,double z):
        """
        Gives the linear cdm+b pk (in Mpc**3) for a given k (in 1/Mpc) and z

        .. note::

            there is an additional check that output contains `mPk`,
            because otherwise a segfault will occur

        """
        self.compute(["fourier"])

        cdef double pk_cb_lin

        if (self.pt.has_pk_matter == _FALSE_):
            raise CosmoSevereError("No power spectrum computed. You must add mPk to the list of outputs.")

        if (self.fo.has_pk_cb == _FALSE_):
            raise CosmoSevereError("P_cb not computed by CLASS (probably because there are no massive neutrinos)")

        if fourier_pk_at_k_and_z(&self.ba,&self.pm,&self.fo,pk_linear,k,z,self.fo.index_pk_cb,&pk_cb_lin,NULL)==_FAILURE_:
            raise CosmoSevereError(self.fo.error_message)

        return pk_cb_lin

    # Gives the total matter pk for a given (k,z)
    def pk_numerical_nw(self,double k,double z):
        """
        Gives the nowiggle (smoothed) linear total matter pk (in Mpc**3) for a given k (in 1/Mpc) and z

        .. note::

            there is an additional check that `numerical_nowiggle` was set to `yes`,
            because otherwise a segfault will occur

        """
        self.compute(["fourier"])

        cdef double pk_numerical_nw

        if (self.fo.has_pk_numerical_nowiggle == _FALSE_):
            raise CosmoSevereError("No power spectrum computed. You must set `numerical_nowiggle` to `yes` in input")

        if fourier_pk_at_k_and_z(&self.ba,&self.pm,&self.fo,pk_numerical_nowiggle,k,z,0,&pk_numerical_nw,NULL)==_FAILURE_:
            raise CosmoSevereError(self.fo.error_message)

        return pk_numerical_nw

    # Gives the approximate analytic nowiggle power spectrum for a given k at z=0
    def pk_analytic_nw(self,double k):
        """
        Gives the linear total matter pk (in Mpc**3) for a given k (in 1/Mpc) and z

        .. note::

            there is an additional check that `analytic_nowiggle` was set to `yes`,
            because otherwise a segfault will occur

        """
        self.compute(["fourier"])

        cdef double pk_analytic_nw

        if (self.fo.has_pk_analytic_nowiggle == _FALSE_):
            raise CosmoSevereError("No analytic nowiggle spectrum computed. You must set `analytic_nowiggle` to `yes` in input")

        if fourier_pk_at_k_and_z(&self.ba,&self.pm,&self.fo,pk_analytic_nowiggle,k,0.,self.fo.index_pk_m,&pk_analytic_nw,NULL)==_FAILURE_:
            raise CosmoSevereError(self.fo.error_message)

        return pk_analytic_nw

    def get_pk(self, np.ndarray[DTYPE_t,ndim=3] k, np.ndarray[DTYPE_t,ndim=1] z, int k_size, int z_size, int mu_size):
        """ Fast function to get the power spectrum on a k and z array """
        self.compute(["fourier"])

        cdef np.ndarray[DTYPE_t, ndim=3] pk = np.zeros((k_size,z_size,mu_size),'float64')
        cdef int index_k, index_z, index_mu

        for index_k in range(k_size):
            for index_z in range(z_size):
                for index_mu in range(mu_size):
                    pk[index_k,index_z,index_mu] = self.pk(k[index_k,index_z,index_mu],z[index_z])
        return pk

    def get_pk_cb(self, np.ndarray[DTYPE_t,ndim=3] k, np.ndarray[DTYPE_t,ndim=1] z, int k_size, int z_size, int mu_size):
        """ Fast function to get the power spectrum on a k and z array """
        self.compute(["fourier"])

        cdef np.ndarray[DTYPE_t, ndim=3] pk_cb = np.zeros((k_size,z_size,mu_size),'float64')
        cdef int index_k, index_z, index_mu

        for index_k in range(k_size):
            for index_z in range(z_size):
                for index_mu in range(mu_size):
                    pk_cb[index_k,index_z,index_mu] = self.pk_cb(k[index_k,index_z,index_mu],z[index_z])
        return pk_cb

    def get_pk_lin(self, np.ndarray[DTYPE_t,ndim=3] k, np.ndarray[DTYPE_t,ndim=1] z, int k_size, int z_size, int mu_size):
        """ Fast function to get the linear power spectrum on a k and z array """
        self.compute(["fourier"])

        cdef np.ndarray[DTYPE_t, ndim=3] pk = np.zeros((k_size,z_size,mu_size),'float64')
        cdef int index_k, index_z, index_mu

        for index_k in range(k_size):
            for index_z in range(z_size):
                for index_mu in range(mu_size):
                    pk[index_k,index_z,index_mu] = self.pk_lin(k[index_k,index_z,index_mu],z[index_z])
        return pk

    def get_pk_cb_lin(self, np.ndarray[DTYPE_t,ndim=3] k, np.ndarray[DTYPE_t,ndim=1] z, int k_size, int z_size, int mu_size):
        """ Fast function to get the linear power spectrum on a k and z array """
        self.compute(["fourier"])

        cdef np.ndarray[DTYPE_t, ndim=3] pk_cb = np.zeros((k_size,z_size,mu_size),'float64')
        cdef int index_k, index_z, index_mu

        for index_k in range(k_size):
            for index_z in range(z_size):
                for index_mu in range(mu_size):
                    pk_cb[index_k,index_z,index_mu] = self.pk_cb_lin(k[index_k,index_z,index_mu],z[index_z])
        return pk_cb

    def get_pk_all(self, k, z, nonlinear = True, cdmbar = False, z_axis_in_k_arr = 0, interpolation_kind='cubic'):
        """ General function to get the P(k,z) for ARBITRARY shapes of k,z
            Additionally, it includes the functionality of selecting wether to use the non-linear parts or not,
            and wether to use the cdm baryon power spectrum only
            For Multi-Dimensional k-arrays, it assumes that one of the dimensions is the z-axis
            This is handled by the z_axis_in_k_arr integer, as described in the source code """
        self.compute(["fourier"])

        # z_axis_in_k_arr specifies the integer position of the z_axis wihtin the n-dimensional k_arr
        # Example: 1-d k_array -> z_axis_in_k_arr = 0
        # Example: 3-d k_array with z_axis being the first axis -> z_axis_in_k_arr = 0
        # Example: 3-d k_array with z_axis being the last axis  -> z_axis_in_k_arr = 2

        # 1) Define some utilities
        # Is the user asking for a valid cdmbar?
        ispkcb = cdmbar and not (self.ba.Omega0_ncdm_tot == 0.)

        # Allocate the temporary k/pk array used during the interaction with the underlying C code
        cdef np.float64_t[::1] pk_out = np.empty(self.fo.k_size, dtype='float64')
        k_out = np.asarray(<np.float64_t[:self.fo.k_size]> self.fo.k)

        # Define a function that can write the P(k) for a given z into the pk_out array
        def _write_pk(z,islinear,ispkcb):
          if fourier_pk_at_z(&self.ba,&self.fo,linear,(pk_linear if islinear else pk_nonlinear),z,(self.fo.index_pk_cb if ispkcb else self.fo.index_pk_m),&pk_out[0],NULL)==_FAILURE_:
              raise CosmoSevereError(self.fo.error_message)

        # Check what kind of non-linear redshift there is
        if nonlinear:
          if self.fo.index_tau_min_nl == 0:
            z_max_nonlinear = np.inf
          else:
            z_max_nonlinear = self.z_of_tau(self.fo.tau[self.fo.index_tau_min_nl])
        else:
          z_max_nonlinear = -1.

        # Only get the nonlinear function where the nonlinear treatment is possible
        def _islinear(z):
          if z > z_max_nonlinear or (self.fo.method == nl_none):
            return True
          else:
            return False

        # A simple wrapper for writing the P(k) in the given location and interpolating it
        def _interpolate_pk_at_z(karr,z):
          _write_pk(z,_islinear(z),ispkcb)
          interp_func = interp1d(k_out,np.log(pk_out),kind=interpolation_kind,copy=True)
          return np.exp(interp_func(karr))

        # 2) Check if z array, or z value
        if not isinstance(z,(list,np.ndarray)):
            # Only single z value was passed -> k could still be an array of arbitrary dimension
            if not isinstance(k,(list,np.ndarray)):
                # Only single z value AND only single k value -> just return a value
                # This iterates over ALL remaining dimensions
                return ((self.pk_cb if ispkcb else self.pk) if not _islinear(z) else (self.pk_cb_lin if ispkcb else self.pk_lin))(k,z)
            else:
                k_arr = np.array(k)
                result = _interpolate_pk_at_z(k_arr,z)
                return result

        # 3) An array of z values was passed
        k_arr = np.array(k)
        z_arr = np.array(z)
        if( z_arr.ndim != 1 ):
            raise CosmoSevereError("Can only parse one-dimensional z-arrays, not multi-dimensional")

        if( k_arr.ndim > 1 ):
            # 3.1) If there is a multi-dimensional k-array of EQUAL lenghts
            out_pk = np.empty(np.shape(k_arr))
            # Bring the z_axis to the front
            k_arr = np.moveaxis(k_arr, z_axis_in_k_arr, 0)
            out_pk = np.moveaxis(out_pk, z_axis_in_k_arr, 0)
            if( len(k_arr) != len(z_arr) ):
                raise CosmoSevereError("Mismatching array lengths of the z-array")
            for index_z in range(len(z_arr)):
                out_pk[index_z] = _interpolate_pk_at_z(k_arr[index_z],z[index_z])
            # Move the z_axis back into position
            k_arr = np.moveaxis(k_arr, 0, z_axis_in_k_arr)
            out_pk = np.moveaxis(out_pk, 0, z_axis_in_k_arr)
            return out_pk
        else:
            # 3.2) If there is a multi-dimensional k-array of UNEQUAL lenghts
            if isinstance(k_arr[0],(list,np.ndarray)):
                # A very special thing happened: The user passed a k array with UNEQUAL lengths of k arrays for each z
                out_pk = []
                for index_z in range(len(z_arr)):
                    k_arr_at_z = np.array(k_arr[index_z])
                    out_pk_at_z = _interpolate_pk_at_z(k_arr_at_z,z[index_z])
                    out_pk.append(out_pk_at_z)
                return out_pk

            # 3.3) If there is a single-dimensional k-array
            # The user passed a z-array, but only a 1-d k array
            # Assume thus, that the k array should be reproduced for all z
            out_pk = np.empty((len(z_arr),len(k_arr)))
            for index_z in range(len(z_arr)):
                out_pk[index_z] = _interpolate_pk_at_z(k_arr,z_arr[index_z])
            return out_pk

    #################################
    # Gives a grid of values of matter and/or cb power spectrum, together with the vectors of corresponding k and z values
    def get_pk_and_k_and_z(self, nonlinear=True, only_clustering_species = False, h_units=False):
        """
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
        """
        self.compute(["fourier"])

        cdef np.ndarray[DTYPE_t,ndim=2] pk = np.zeros((self.fo.k_size_pk, self.fo.ln_tau_size),'float64')
        cdef np.ndarray[DTYPE_t,ndim=1] k = np.zeros((self.fo.k_size_pk),'float64')
        cdef np.ndarray[DTYPE_t,ndim=1] z = np.zeros((self.fo.ln_tau_size),'float64')
        cdef int index_k, index_tau, index_pk
        cdef double z_max_nonlinear, z_max_requested
        # consistency checks

        if self.fo.has_pk_matter == False:
            raise CosmoSevereError("You ask classy to return an array of P(k,z) values, but the input parameters sent to CLASS did not require any P(k,z) calculations; add 'mPk' in 'output'")

        if nonlinear == True and self.fo.method == nl_none:
            raise CosmoSevereError("You ask classy to return an array of nonlinear P(k,z) values, but the input parameters sent to CLASS did not require any non-linear P(k,z) calculations; add e.g. 'halofit' or 'HMcode' in 'nonlinear'")

        # check wich type of P(k) to return (total or clustering only, i.e. without massive neutrino contribution)
        if (only_clustering_species == True):
            index_pk = self.fo.index_pk_cluster
        else:
            index_pk = self.fo.index_pk_total

        # get list of redshifts
        # the ln(times) of interest are stored in self.fo.ln_tau[index_tau]
        # For nonlinear, we have to additionally cut out the linear values

        if self.fo.ln_tau_size == 1:
            raise CosmoSevereError("You ask classy to return an array of P(k,z) values, but the input parameters sent to CLASS did not require any P(k,z) calculations for z>0; pass either a list of z in 'z_pk' or one non-zero value in 'z_max_pk'")
        else:
            for index_tau in range(self.fo.ln_tau_size):
                if index_tau == self.fo.ln_tau_size-1:
                    z[index_tau] = 0.
                else:
                    z[index_tau] = self.z_of_tau(np.exp(self.fo.ln_tau[index_tau]))

        # check consitency of the list of redshifts

        if nonlinear == True:
            # Check highest value of z at which nl corrections could be computed.
            # In the table tau_sampling it corresponds to index: self.fo.index_tau_min_nl
            z_max_nonlinear = self.z_of_tau(self.fo.tau[self.fo.index_tau_min_nl])

            # Check highest value of z in the requested output.
            z_max_requested = z[0]

            # The first z must be larger or equal to the second one, that is,
            # the first index must be smaller or equal to the second one.
            # If not, raise and error.
            if (z_max_requested > z_max_nonlinear and self.fo.index_tau_min_nl>0):
                raise CosmoSevereError("get_pk_and_k_and_z() is trying to return P(k,z) up to z_max=%e (the redshift range of computed pk); but the input parameters sent to CLASS (in particular ppr->nonlinear_min_k_max=%e) were such that the non-linear P(k,z) could only be consistently computed up to z=%e; increase the precision parameter 'nonlinear_min_k_max', or only obtain the linear pk"%(z_max_requested,self.pr.nonlinear_min_k_max,z_max_nonlinear))

        # get list of k

        if h_units:
            units=1./self.ba.h
        else:
            units=1

        for index_k in range(self.fo.k_size_pk):
            k[index_k] = self.fo.k[index_k]*units

        # get P(k,z) array

        for index_tau in range(self.fo.ln_tau_size):
            for index_k in range(self.fo.k_size_pk):
                if nonlinear == True:
                    pk[index_k, index_tau] = np.exp(self.fo.ln_pk_nl[index_pk][index_tau * self.fo.k_size + index_k])
                else:
                    pk[index_k, index_tau] = np.exp(self.fo.ln_pk_l[index_pk][index_tau * self.fo.k_size + index_k])

        return pk, k, z

    #################################
    # Gives a grid of each transfer functions arranged in a dictionary, together with the vectors of corresponding k and z values
    def get_transfer_and_k_and_z(self, output_format='class', h_units=False):
        """
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
        """
        self.compute(["transfer"])

        cdef np.ndarray[DTYPE_t,ndim=1] k = np.zeros((self.pt.k_size_pk),'float64')
        cdef np.ndarray[DTYPE_t,ndim=1] z = np.zeros((self.pt.ln_tau_size),'float64')
        cdef int index_k, index_tau
        cdef char * titles
        cdef double * data
        cdef file_format outf

        # consistency checks
        if (self.pt.has_density_transfers == False) and (self.pt.has_velocity_transfers == False):
            raise CosmoSevereError("You ask classy to return transfer functions, but the input parameters sent to CLASS did not require any T(k,z) calculations; add 'dTk' and/or 'vTk' in 'output'")

        index_md = self.pt.index_md_scalars;

        if (self.pt.ic_size[index_md] > 1):
            raise CosmoSevereError("For simplicity, get_transfer_and_k_and_z() has been written assuming only adiabatic initial conditions. You need to write the generalisation to cases with multiple initial conditions.")

        # check out put format
        if output_format == 'camb':
            outf = camb_format
        else:
            outf = class_format

        # check name and number of trnasfer functions computed ghy CLASS

        titles = <char*>calloc(_MAXTITLESTRINGLENGTH_,sizeof(char))

        if perturbations_output_titles(&self.ba,&self.pt, outf, titles)==_FAILURE_:
            free(titles) # manual free due to error
            raise CosmoSevereError(self.pt.error_message)

        tmp = <bytes> titles
        tmp = str(tmp.decode())
        names = tmp.split("\t")[:-1]

        free(titles)

        number_of_titles = len(names)

        # get list of redshifts
        # the ln(times) of interest are stored in self.fo.ln_tau[index_tau]

        if self.pt.ln_tau_size == 1:
            raise CosmoSevereError("You ask classy to return an array of T_x(k,z) values, but the input parameters sent to CLASS did not require any transfer function calculations for z>0; pass either a list of z in 'z_pk' or one non-zero value in 'z_max_pk'")
        else:
            for index_tau in range(self.pt.ln_tau_size):
                if index_tau == self.pt.ln_tau_size-1:
                    z[index_tau] = 0.
                else:
                    z[index_tau] = self.z_of_tau(np.exp(self.pt.ln_tau[index_tau]))

        # get list of k

        if h_units:
            units=1./self.ba.h
        else:
            units=1

        k_size = self.pt.k_size_pk
        for index_k in range(k_size):
            k[index_k] = self.pt.k[index_md][index_k]*units

        # create output dictionary

        tk = {}
        for index_type,name in enumerate(names):
            if index_type > 0:
                tk[name] = np.zeros((k_size, len(z)),'float64')

        # allocate the vector in wich the transfer functions will be stored temporarily for all k and types at a given z
        data = <double*>malloc(sizeof(double)*number_of_titles*self.pt.k_size[index_md])

        # get T(k,z) array

        for index_tau in range(len(z)):
            if perturbations_output_data_at_index_tau(&self.ba, &self.pt, outf, index_tau, number_of_titles, data)==_FAILURE_:
                free(data) # manual free due to error
                raise CosmoSevereError(self.pt.error_message)

            for index_type,name in enumerate(names):
                if index_type > 0:
                    for index_k in range(k_size):
                        tk[name][index_k, index_tau] = data[index_k*number_of_titles+index_type]

        free(data)
        return tk, k, z

    #################################
    # Gives a grid of values of the power spectrum of the quantity [k^2*(phi+psi)/2], where (phi+psi)/2 is the Weyl potential, together with the vectors of corresponding k and z values
    def get_Weyl_pk_and_k_and_z(self, nonlinear=False, h_units=False):
        """
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
        """
        self.compute(["fourier"])

        cdef np.ndarray[DTYPE_t,ndim=2] pk = np.zeros((self.fo.k_size_pk,self.fo.ln_tau_size),'float64')
        cdef np.ndarray[DTYPE_t,ndim=1] z = np.zeros((self.fo.ln_tau_size),'float64')
        cdef np.ndarray[DTYPE_t,ndim=2] k4 = np.zeros((self.fo.k_size_pk, self.fo.ln_tau_size),'float64')
        cdef np.ndarray[DTYPE_t,ndim=2] phi = np.zeros((self.fo.k_size_pk, self.fo.ln_tau_size),'float64')
        cdef np.ndarray[DTYPE_t,ndim=2] psi = np.zeros((self.fo.k_size_pk, self.fo.ln_tau_size),'float64')
        cdef np.ndarray[DTYPE_t,ndim=2] d_m = np.zeros((self.fo.k_size_pk, self.fo.ln_tau_size),'float64')
        cdef np.ndarray[DTYPE_t,ndim=2] Weyl_pk = np.zeros((self.fo.k_size_pk, self.fo.ln_tau_size),'float64')

        cdef bint input_nonlinear = nonlinear
        cdef bint input_h_units = h_units

        cdef int index_z

        # get total matter power spectrum
        pk, k, z = self.get_pk_and_k_and_z(nonlinear=input_nonlinear, only_clustering_species = False, h_units=input_h_units)

        # get transfer functions
        tk_and_k_and_z = {}
        tk_and_k_and_z, k, z = self.get_transfer_and_k_and_z(output_format='class',h_units=input_h_units)
        phi = tk_and_k_and_z['phi']
        psi = tk_and_k_and_z['psi']
        d_m = tk_and_k_and_z['d_m']

        # get an array containing k**4 (same for all redshifts)
        for index_z in range(self.fo.ln_tau_size):
            k4[:,index_z] = k**4

        # rescale total matter power spectrum to get the Weyl power spectrum times k**4
        # (the latter factor is just a convention. Since there is a factor k**2 in the Poisson equation,
        # this rescaled Weyl spectrum has a shape similar to the matter power spectrum).
        Weyl_pk = pk * ((phi+psi)/2./d_m)**2 * k4

        return Weyl_pk, k, z

    #################################
    # Gives sigma(R,z) for a given (R,z)
    def sigma(self,R,z, h_units = False):
        """
        Gives sigma (total matter) for a given R and z
        (R is the radius in units of Mpc, so if R=8/h this will be the usual sigma8(z).
         This is unless h_units is set to true, in which case R is the radius in units of Mpc/h,
         and R=8 corresponds to sigma8(z))

        .. note::

            there is an additional check to verify whether output contains `mPk`,
            and whether k_max > ...
            because otherwise a segfault will occur

        """
        self.compute(["fourier"])

        cdef double sigma

        zarr = np.atleast_1d(z).astype(np.float64)
        Rarr = np.atleast_1d(R).astype(np.float64)

        if (self.pt.has_pk_matter == _FALSE_):
            raise CosmoSevereError("No power spectrum computed. In order to get sigma(R,z) you must add mPk to the list of outputs.")

        if (self.pt.k_max_for_pk < self.ba.h):
            raise CosmoSevereError("In order to get sigma(R,z) you must set 'P_k_max_h/Mpc' to 1 or bigger, in order to have k_max > 1 h/Mpc.")

        R_in_Mpc = (Rarr if not h_units else Rarr/self.ba.h)

        pairs = np.array(np.meshgrid(zarr,R_in_Mpc)).T.reshape(-1,2)

        sigmas = np.empty(pairs.shape[0])
        for ip, pair in enumerate(pairs):
          if fourier_sigmas_at_z(&self.pr,&self.ba,&self.fo,pair[1],pair[0],self.fo.index_pk_m,out_sigma,&sigma)==_FAILURE_:
              raise CosmoSevereError(self.fo.error_message)
          sigmas[ip] = sigma

        return (sigmas[0] if (np.isscalar(z) and np.isscalar(R)) else np.squeeze(sigmas.reshape(len(zarr),len(Rarr))))

    # Gives sigma_cb(R,z) for a given (R,z)
    def sigma_cb(self,double R,double z, h_units = False):
        """
        Gives sigma (cdm+b) for a given R and z
        (R is the radius in units of Mpc, so if R=8/h this will be the usual sigma8(z)
         This is unless h_units is set to true, in which case R is the radius in units of Mpc/h,
         and R=8 corresponds to sigma8(z))

        .. note::

            there is an additional check to verify whether output contains `mPk`,
            and whether k_max > ...
            because otherwise a segfault will occur

        """
        self.compute(["fourier"])

        cdef double sigma_cb

        zarr = np.atleast_1d(z).astype(np.float64)
        Rarr = np.atleast_1d(R).astype(np.float64)

        if (self.pt.has_pk_matter == _FALSE_):
            raise CosmoSevereError("No power spectrum computed. In order to get sigma(R,z) you must add mPk to the list of outputs.")

        if (self.fo.has_pk_cb == _FALSE_):
            raise CosmoSevereError("sigma_cb not computed by CLASS (probably because there are no massive neutrinos)")

        if (self.pt.k_max_for_pk < self.ba.h):
            raise CosmoSevereError("In order to get sigma(R,z) you must set 'P_k_max_h/Mpc' to 1 or bigger, in order to have k_max > 1 h/Mpc.")

        R_in_Mpc = (Rarr if not h_units else Rarr/self.ba.h)

        pairs = np.array(np.meshgrid(zarr,R_in_Mpc)).T.reshape(-1,2)

        sigmas_cb = np.empty(pairs.shape[0])
        for ip, pair in enumerate(pairs):
          if fourier_sigmas_at_z(&self.pr,&self.ba,&self.fo,R,z,self.fo.index_pk_cb,out_sigma,&sigma_cb)==_FAILURE_:
            raise CosmoSevereError(self.fo.error_message)
          sigmas_cb[ip] = sigma_cb

        return (sigmas_cb[0] if (np.isscalar(z) and np.isscalar(R)) else np.squeeze(sigmas_cb.reshape(len(zarr),len(Rarr))))

    # Gives effective logarithmic slope of P_L(k,z) (total matter) for a given (k,z)
    def pk_tilt(self,double k,double z):
        """
        Gives effective logarithmic slope of P_L(k,z) (total matter) for a given k and z
        (k is the wavenumber in units of 1/Mpc, z is the redshift, the output is dimensionless)

        .. note::

            there is an additional check to verify whether output contains `mPk` and whether k is in the right range

        """
        self.compute(["fourier"])

        cdef double pk_tilt

        if (self.pt.has_pk_matter == _FALSE_):
            raise CosmoSevereError("No power spectrum computed. In order to get pk_tilt(k,z) you must add mPk to the list of outputs.")

        if (k < self.fo.k[1] or k > self.fo.k[self.fo.k_size-2]):
            raise CosmoSevereError("In order to get pk_tilt at k=%e 1/Mpc, you should compute P(k,z) in a wider range of k's"%k)

        if fourier_pk_tilt_at_k_and_z(&self.ba,&self.pm,&self.fo,pk_linear,k,z,self.fo.index_pk_total,&pk_tilt)==_FAILURE_:
            raise CosmoSevereError(self.fo.error_message)

        return pk_tilt

    def age(self):
        self.compute(["background"])
        return self.ba.age

    def h(self):
        return self.ba.h

    def n_s(self):
        return self.pm.n_s

    def tau_reio(self):
        self.compute(["thermodynamics"])
        return self.th.tau_reio

    def Omega_m(self):
        return self.ba.Omega0_m

    def Omega_r(self):
        return self.ba.Omega0_r

    def theta_s_100(self):
        self.compute(["thermodynamics"])
        return 100.*self.th.rs_rec/self.th.da_rec/(1.+self.th.z_rec)

    def theta_star_100(self):
        self.compute(["thermodynamics"])
        return 100.*self.th.rs_star/self.th.da_star/(1.+self.th.z_star)

    def Omega_Lambda(self):
        return self.ba.Omega0_lambda

    def Omega_g(self):
        return self.ba.Omega0_g

    def Omega_b(self):
        return self.ba.Omega0_b

    def omega_b(self):
        return self.ba.Omega0_b * self.ba.h * self.ba.h

    def Neff(self):
        self.compute(["background"])
        return self.ba.Neff

    def k_eq(self):
        self.compute(["background"])
        return self.ba.a_eq*self.ba.H_eq

    def z_eq(self):
        self.compute(["background"])
        return 1./self.ba.a_eq-1.

    def sigma8(self):
        self.compute(["fourier"])
        if (self.pt.has_pk_matter == _FALSE_):
            raise CosmoSevereError("No power spectrum computed. In order to get sigma8, you must add mPk to the list of outputs.")
        return self.fo.sigma8[self.fo.index_pk_m]

    def S8(self):
        return self.sigma8()*np.sqrt(self.Omega_m()/0.3)

    #def neff(self):
    #    self.compute(["harmonic"])
    #    return self.hr.neff

    def sigma8_cb(self):
        self.compute(["fourier"])
        if (self.pt.has_pk_matter == _FALSE_):
            raise CosmoSevereError("No power spectrum computed. In order to get sigma8_cb, you must add mPk to the list of outputs.")
        return self.fo.sigma8[self.fo.index_pk_cb]

    def rs_drag(self):
        self.compute(["thermodynamics"])
        return self.th.rs_d

    def z_reio(self):
        self.compute(["thermodynamics"])
        return self.th.z_reio

    def angular_distance(self, z):
        """
        angular_distance(z)

        Return the angular diameter distance (exactly, the quantity defined by Class
        as index_bg_ang_distance in the background module)

        Parameters
        ----------
        z : float
                Desired redshift
        """
        self.compute(["background"])

        cdef int last_index #junk
        cdef double * pvecback

        zarr = np.atleast_1d(z).astype(np.float64)

        pvecback = <double*> calloc(self.ba.bg_size,sizeof(double))

        D_A = np.empty_like(zarr)
        for iz, redshift in enumerate(zarr):
          if background_at_z(&self.ba,redshift,long_info,inter_normal,&last_index,pvecback)==_FAILURE_:
              free(pvecback) #Manual free due to error
              raise CosmoSevereError(self.ba.error_message)

          D_A[iz] = pvecback[self.ba.index_bg_ang_distance]

        free(pvecback)

        return (D_A[0] if np.isscalar(z) else D_A)

    #################################
    # Get angular diameter distance of object at z2 as seen by observer at z1,
    def angular_distance_from_to(self, z1, z2):
        """
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
        """
        self.compute(["background"])

        cdef int last_index #junk
        cdef double * pvecback

        if z1>=z2:
            return 0.

        else:
            pvecback = <double*> calloc(self.ba.bg_size,sizeof(double))

            if background_at_z(&self.ba,z1,long_info,inter_normal,&last_index,pvecback)==_FAILURE_:
                free(pvecback) #manual free due to error
                raise CosmoSevereError(self.ba.error_message)

            # This is the comoving distance to object at z1
            chi1 = pvecback[self.ba.index_bg_conf_distance]

            if background_at_z(&self.ba,z2,long_info,inter_normal,&last_index,pvecback)==_FAILURE_:
                free(pvecback) #manual free due to error
                raise CosmoSevereError(self.ba.error_message)

            # This is the comoving distance to object at z2
            chi2 = pvecback[self.ba.index_bg_conf_distance]

            free(pvecback)

            if self.ba.K == 0:
                return (chi2-chi1)/(1+z2)
            elif self.ba.K > 0:
                return np.sin(np.sqrt(self.ba.K)*(chi2-chi1))/np.sqrt(self.ba.K)/(1+z2)
            elif self.ba.K < 0:
                return np.sinh(np.sqrt(-self.ba.K)*(chi2-chi1))/np.sqrt(-self.ba.K)/(1+z2)

    def comoving_distance(self, z):
        """
        comoving_distance(z)

        Return the comoving distance

        Parameters
        ----------
        z : float
                Desired redshift
        """
        self.compute(["background"])

        cdef int last_index #junk
        cdef double * pvecback

        zarr = np.atleast_1d(z).astype(np.float64)

        pvecback = <double*> calloc(self.ba.bg_size,sizeof(double))

        r = np.empty_like(zarr)
        for iz, redshift in enumerate(zarr):
          if background_at_z(&self.ba,redshift,long_info,inter_normal,&last_index,pvecback)==_FAILURE_:
              free(pvecback) #manual free due to error
              raise CosmoSevereError(self.ba.error_message)

          r[iz] = pvecback[self.ba.index_bg_conf_distance]

        free(pvecback)

        return (r[0] if np.isscalar(z) else r)

    def scale_independent_growth_factor(self, z):
        """
        scale_independent_growth_factor(z)

        Return the scale invariant growth factor D(a) for CDM perturbations
        (exactly, the quantity defined by Class as index_bg_D in the background module)

        Parameters
        ----------
        z : float
                Desired redshift
        """
        self.compute(["background"])

        cdef int last_index #junk
        cdef double * pvecback

        zarr = np.atleast_1d(z).astype(np.float64)

        pvecback = <double*> calloc(self.ba.bg_size,sizeof(double))

        D = np.empty_like(zarr)
        for iz, redshift in enumerate(zarr):
          if background_at_z(&self.ba,redshift,long_info,inter_normal,&last_index,pvecback)==_FAILURE_:
              free(pvecback) #manual free due to error
              raise CosmoSevereError(self.ba.error_message)

          D[iz] = pvecback[self.ba.index_bg_D]

        free(pvecback)

        return (D[0] if np.isscalar(z) else D)

    def scale_independent_growth_factor_f(self, z):
        """
        scale_independent_growth_factor_f(z)

        Return the scale independent growth factor f(z)=d ln D / d ln a for CDM perturbations
        (exactly, the quantity defined by Class as index_bg_f in the background module)

        Parameters
        ----------
        z : float
                Desired redshift
        """
        self.compute(["background"])

        cdef int last_index #junk
        cdef double * pvecback

        zarr = np.atleast_1d(z).astype(np.float64)

        pvecback = <double*> calloc(self.ba.bg_size,sizeof(double))

        f = np.empty_like(zarr)
        for iz, redshift in enumerate(zarr):
          if background_at_z(&self.ba,redshift,long_info,inter_normal,&last_index,pvecback)==_FAILURE_:
              free(pvecback) #manual free due to error
              raise CosmoSevereError(self.ba.error_message)

          f[iz] = pvecback[self.ba.index_bg_f]

        free(pvecback)

        return (f[0] if np.isscalar(z) else f)

    #################################
    def scale_dependent_growth_factor_f(self, k, z, h_units=False, nonlinear=False, Nz=20):
        """
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
        """
        self.compute(["fourier"])

        # build array of z values at wich P(k,z) was pre-computed by class (for numerical derivative)
        # check that P(k,z) was stored at different zs
        if self.fo.ln_tau_size > 1:
            # check that input z is in stored range
            z_max = self.z_of_tau(np.exp(self.fo.ln_tau[0]))
            if (z<0) or (z>z_max):
                raise CosmoSevereError("You asked for f(k,z) at a redshift %e outside of the computed range [0,%e]"%(z,z_max))
            # create array of zs in growing z order (decreasing tau order)
            z_array = np.empty(self.fo.ln_tau_size)
            # first redshift is exactly zero
            z_array[0]=0.
            # next values can be inferred from ln_tau table
            if (self.fo.ln_tau_size>1):
                for i in range(1,self.fo.ln_tau_size):
                    z_array[i] = self.z_of_tau(np.exp(self.fo.ln_tau[self.fo.ln_tau_size-1-i]))
        else:
            raise CosmoSevereError("You asked for the scale-dependent growth factor: this requires numerical derivation of P(k,z) w.r.t z, and thus passing a non-zero input parameter z_max_pk")

        # if needed, convert k to units of 1/Mpc
        if h_units:
            k = k*self.ba.h

        # Allocate an array of P(k,z[...]) values
        Pk_array = np.empty_like(z_array)

        # Choose whether to use .pk() or .pk_lin()
        # The linear pk is in .pk_lin if nonlinear corrections have been computed, in .pk otherwise
        # The non-linear pk is in .pk if nonlinear corrections have been computed
        if nonlinear == False:
            if self.fo.method == nl_none:
                use_pk_lin = False
            else:
                use_pk_lin = True
        else:
            if self.fo.method == nl_none:
                raise CosmoSevereError("You asked for the scale-dependent growth factor of non-linear matter fluctuations, but you did not ask for non-linear calculations at all")
            else:
                use_pk_lin = False

        # Get P(k,z) and array P(k,z[...])
        if use_pk_lin == False:
            Pk = self.pk(k,z)
            for iz, zval in enumerate(z_array):
                Pk_array[iz] = self.pk(k,zval)
        else:
            Pk = self.pk_lin(k,z)
            for iz, zval in enumerate(z_array):
                Pk_array[iz] = self.pk_lin(k,zval)

        # Compute derivative (d ln P / d ln z)
        dPkdz = UnivariateSpline(z_array,Pk_array,s=0).derivative()(z)

        # Compute growth factor f
        f = -0.5*(1+z)*dPkdz/Pk

        return f

    #################################
    def scale_dependent_growth_factor_f_cb(self, k, z, h_units=False, nonlinear=False, Nz=20):
        """
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
        """

        # build array of z values at wich P_cb(k,z) was pre-computed by class (for numerical derivative)
        # check that P_cb(k,z) was stored at different zs
        if self.fo.ln_tau_size > 1:
            # check that input z is in stored range
            z_max = self.z_of_tau(np.exp(self.fo.ln_tau[0]))
            if (z<0) or (z>z_max):
                raise CosmoSevereError("You asked for f_cb(k,z) at a redshift %e outside of the computed range [0,%e]"%(z,z_max))
            # create array of zs in growing z order (decreasing tau order)
            z_array = np.empty(self.fo.ln_tau_size)
            # first redshift is exactly zero
            z_array[0]=0.
            # next values can be inferred from ln_tau table
            if (self.fo.ln_tau_size>1):
                for i in range(1,self.fo.ln_tau_size):
                    z_array[i] = self.z_of_tau(np.exp(self.fo.ln_tau[self.fo.ln_tau_size-1-i]))
        else:
            raise CosmoSevereError("You asked for the scale-dependent growth factor: this requires numerical derivation of P(k,z) w.r.t z, and thus passing a non-zero input parameter z_max_pk")

        # if needed, convert k to units of 1/Mpc
        if h_units:
            k = k*self.ba.h

        # Allocate an array of P(k,z[...]) values
        Pk_array = np.empty_like(z_array)

        # Choose whether to use .pk() or .pk_lin()
        # The linear pk is in .pk_lin if nonlinear corrections have been computed, in .pk otherwise
        # The non-linear pk is in .pk if nonlinear corrections have been computed
        if nonlinear == False:
            if self.fo.method == nl_none:
                use_pk_lin = False
            else:
                use_pk_lin = True
        else:
            if self.fo.method == nl_none:
                raise CosmoSevereError("You asked for the scale-dependent growth factor of non-linear matter fluctuations, but you did not ask for non-linear calculations at all")
            else:
                use_pk_lin = False

        # Get P(k,z) and array P(k,z[...])
        if use_pk_lin == False:
            Pk = self.pk(k,z)
            for iz, zval in enumerate(z_array):
                Pk_array[iz] = self.pk_cb(k,zval)
        else:
            Pk = self.pk_lin(k,z)
            for iz, zval in enumerate(z_array):
                Pk_array[iz] = self.pk_cb_lin(k,zval)

        # Compute derivative (d ln P / d ln z)
        dPkdz = UnivariateSpline(z_array,Pk_array,s=0).derivative()(z)

        # Compute growth factor f
        f = -0.5*(1+z)*dPkdz/Pk

        return f

    #################################
    # gives f(z)*sigma8(z) where f(z) is the scale-independent growth factor
    def scale_independent_f_sigma8(self, z):
        """
        scale_independent_f_sigma8(z)

        Return the scale independent growth factor f(z) multiplied by sigma8(z)

        Parameters
        ----------
        z : float
                Desired redshift

        Returns
        -------
        f(z)*sigma8(z) (dimensionless)
        """
        return self.scale_independent_growth_factor_f(z)*self.sigma(8,z,h_units=True)

    #################################
    # gives an estimation of f(z)*sigma8(z) at the scale of 8 h/Mpc, computed as (d sigma8/d ln a)
    def effective_f_sigma8(self, z, z_step=0.1):
        """
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
        """

        # we need d sigma8/d ln a = - (d sigma8/dz)*(1+z)
        if hasattr(z, "__len__"):
          out_array = np.empty_like(z,dtype=np.float64)
          for iz, redshift in enumerate(z):
            out_array[iz] = self.effective_f_sigma8(redshift, z_step=z_step)
          return out_array

        # if possible, use two-sided derivative with default value of z_step
        if z >= z_step:
            return (self.sigma(8,z-z_step,h_units=True)-self.sigma(8,z+z_step,h_units=True))/(2.*z_step)*(1+z)
        else:
            # if z is between z_step/10 and z_step, reduce z_step to z, and then stick to two-sided derivative
            if (z > z_step/10.):
                z_step = z
                return (self.sigma(8,z-z_step,h_units=True)-self.sigma(8,z+z_step,h_units=True))/(2.*z_step)*(1+z)
            # if z is between 0 and z_step/10, use single-sided derivative with z_step/10
            else:
                z_step /=10
                return (self.sigma(8,z,h_units=True)-self.sigma(8,z+z_step,h_units=True))/z_step*(1+z)

    #################################
    # gives an estimation of f(z)*sigma8(z) at the scale of 8 h/Mpc, computed as (d sigma8/d ln a)
    def effective_f_sigma8_spline(self, z, Nz=20):
        """
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
        """
        self.compute(["fourier"])

        if hasattr(z, "__len__"):
          out_array = np.empty_like(z,dtype=np.float64)
          for iz, redshift in enumerate(z):
            out_array[iz] = self.effective_f_sigma8_spline(redshift, Nz=Nz)
          return out_array

        # we need d sigma8/d ln a = - (d sigma8/dz)*(1+z)
        if self.fo.ln_tau_size>0:
          z_max = self.z_of_tau(np.exp(self.fo.ln_tau[0]))
        else:
          z_max = 0

        if (z<0) or (z>z_max):
            raise CosmoSevereError("You asked for effective_f_sigma8 at a redshift %e outside of the computed range [0,%e]"%(z,z_max))

        if (z<0.1):
            z_array = np.linspace(0, 0.2, num = Nz)
        elif (z<z_max-0.1):
            z_array = np.linspace(z-0.1, z+0.1, num = Nz)
        else:
            z_array = np.linspace(z_max-0.2, z_max, num = Nz)

        sig8_array = self.sigma(8,z_array,h_units=True)
        return -CubicSpline(z_array,sig8_array).derivative()(z)*(1+z)

   #################################
    def z_of_tau(self, tau):
        """
        Redshift corresponding to a given conformal time.

        Parameters
        ----------
        tau : float
                Conformal time
        """
        self.compute(["background"])

        cdef int last_index #junk
        cdef double * pvecback

        tauarr = np.atleast_1d(tau).astype(np.float64)

        pvecback = <double*> calloc(self.ba.bg_size,sizeof(double))

        z = np.empty_like(tauarr)
        for itau, tauval in enumerate(tauarr):
          if background_at_tau(&self.ba,tauval,long_info,inter_normal,&last_index,pvecback)==_FAILURE_:
              free(pvecback) #manual free due to error
              raise CosmoSevereError(self.ba.error_message)

          z[itau] = 1./pvecback[self.ba.index_bg_a]-1.

        free(pvecback)

        return (z[0] if np.isscalar(tau) else z)

    def Hubble(self, z):
        """
        Hubble(z)

        Return the Hubble rate (exactly, the quantity defined by Class as index_bg_H
        in the background module)

        Parameters
        ----------
        z : float
                Desired redshift
        """
        self.compute(["background"])

        cdef int last_index #junk
        cdef double * pvecback

        zarr = np.atleast_1d(z).astype(np.float64)

        pvecback = <double*> calloc(self.ba.bg_size,sizeof(double))

        H = np.empty_like(zarr)
        for iz, redshift in enumerate(zarr):
          if background_at_z(&self.ba,redshift,long_info,inter_normal,&last_index,pvecback)==_FAILURE_:
              free(pvecback) #manual free due to error
              raise CosmoSevereError(self.ba.error_message)

          H[iz] = pvecback[self.ba.index_bg_H]

        free(pvecback)

        return (H[0] if np.isscalar(z) else H)

    def Om_m(self, z):
        """
        Omega_m(z)

        Return the matter density fraction (exactly, the quantity defined by Class as index_bg_Omega_m
        in the background module)

        Parameters
        ----------
        z : float
                Desired redshift
        """
        self.compute(["background"])

        cdef int last_index #junk
        cdef double * pvecback

        zarr = np.atleast_1d(z).astype(np.float64)

        pvecback = <double*> calloc(self.ba.bg_size,sizeof(double))

        Om_m = np.empty_like(zarr)
        for iz, redshift in enumerate(zarr):
          if background_at_z(&self.ba,redshift,long_info,inter_normal,&last_index,pvecback)==_FAILURE_:
              free(pvecback) #manual free due to error
              raise CosmoSevereError(self.ba.error_message)

          Om_m[iz] = pvecback[self.ba.index_bg_Omega_m]

        free(pvecback)

        return (Om_m[0] if np.isscalar(z) else Om_m)

    def Om_b(self, z):
        """
        Omega_b(z)

        Return the baryon density fraction (exactly, the ratio of quantities defined by Class as
        index_bg_rho_b and index_bg_rho_crit in the background module)

        Parameters
        ----------
        z : float
                Desired redshift
        """
        self.compute(["background"])

        cdef int last_index #junk
        cdef double * pvecback

        zarr = np.atleast_1d(z).astype(np.float64)

        pvecback = <double*> calloc(self.ba.bg_size,sizeof(double))

        Om_b = np.empty_like(zarr)
        for iz, redshift in enumerate(zarr):
          if background_at_z(&self.ba,redshift,long_info,inter_normal,&last_index,pvecback)==_FAILURE_:
              free(pvecback) #manual free due to error
              raise CosmoSevereError(self.ba.error_message)

          Om_b[iz] = pvecback[self.ba.index_bg_rho_b]/pvecback[self.ba.index_bg_rho_crit]

        free(pvecback)

        return (Om_b[0] if np.isscalar(z) else Om_b)

    def Om_cdm(self, z):
        """
        Omega_cdm(z)

        Return the cdm density fraction (exactly, the ratio of quantities defined by Class as
        index_bg_rho_cdm and index_bg_rho_crit in the background module)

        Parameters
        ----------
        z : float
                Desired redshift
        """
        self.compute(["background"])

        cdef int last_index #junk
        cdef double * pvecback

        zarr = np.atleast_1d(z).astype(np.float64)

        Om_cdm = np.zeros_like(zarr)

        if self.ba.has_cdm == True:

          pvecback = <double*> calloc(self.ba.bg_size,sizeof(double))
          for iz, redshift in enumerate(zarr):

              if background_at_z(&self.ba,redshift,long_info,inter_normal,&last_index,pvecback)==_FAILURE_:
                  free(pvecback) #manual free due to error
                  raise CosmoSevereError(self.ba.error_message)

              Om_cdm[iz] = pvecback[self.ba.index_bg_rho_cdm]/pvecback[self.ba.index_bg_rho_crit]

          free(pvecback)

        return (Om_cdm[0] if np.isscalar(z) else Om_cdm)

    def Om_ncdm(self, z):
        """
        Omega_ncdm(z)

        Return the ncdm density fraction (exactly, the ratio of quantities defined by Class as
        Sum_m [ index_bg_rho_ncdm1 + n ], with n=0...N_ncdm-1, and index_bg_rho_crit in the background module)

        Parameters
        ----------
        z : float
                Desired redshift
        """
        self.compute(["background"])

        cdef int last_index #junk
        cdef double * pvecback

        zarr = np.atleast_1d(z).astype(np.float64)

        Om_ncdm = np.zeros_like(zarr)

        if self.ba.has_ncdm == True:

            pvecback = <double*> calloc(self.ba.bg_size,sizeof(double))

            for iz, redshift in enumerate(zarr):
              if background_at_z(&self.ba,redshift,long_info,inter_normal,&last_index,pvecback)==_FAILURE_:
                  free(pvecback) #manual free due to error
                  raise CosmoSevereError(self.ba.error_message)

              rho_ncdm = 0.
              for n in range(self.ba.N_ncdm):
                  rho_ncdm += pvecback[self.ba.index_bg_rho_ncdm1+n]
              Om_ncdm[iz] = rho_ncdm/pvecback[self.ba.index_bg_rho_crit]

            free(pvecback)

        return (Om_ncdm[0] if np.isscalar(z) else Om_ncdm)

    def ionization_fraction(self, z):
        """
        ionization_fraction(z)

        Return the ionization fraction for a given redshift z

        Parameters
        ----------
        z : float
                Desired redshift
        """
        self.compute(["thermodynamics"])

        cdef int last_index #junk
        cdef double * pvecback
        cdef double * pvecthermo

        zarr = np.atleast_1d(z).astype(np.float64)
        xe = np.empty_like(zarr)

        pvecback = <double*> calloc(self.ba.bg_size,sizeof(double))
        pvecthermo = <double*> calloc(self.th.th_size,sizeof(double))

        for iz, redshift in enumerate(zarr):
          if background_at_z(&self.ba,redshift,long_info,inter_normal,&last_index,pvecback)==_FAILURE_:
              free(pvecback) #manual free due to error
              free(pvecthermo) #manual free due to error
              raise CosmoSevereError(self.ba.error_message)

          if thermodynamics_at_z(&self.ba,&self.th,redshift,inter_normal,&last_index,pvecback,pvecthermo) == _FAILURE_:
              free(pvecback) #manual free due to error
              free(pvecthermo) #manual free due to error
              raise CosmoSevereError(self.th.error_message)

          xe[iz] = pvecthermo[self.th.index_th_xe]

        free(pvecback)
        free(pvecthermo)

        return (xe[0] if np.isscalar(z) else xe)

    def baryon_temperature(self, z):
        """
        baryon_temperature(z)

        Give the baryon temperature for a given redshift z

        Parameters
        ----------
        z : float
                Desired redshift
        """
        self.compute(["thermodynamics"])

        cdef int last_index #junk
        cdef double * pvecback
        cdef double * pvecthermo

        zarr = np.atleast_1d(z).astype(np.float64)
        Tb = np.empty_like(zarr)

        pvecback = <double*> calloc(self.ba.bg_size,sizeof(double))
        pvecthermo = <double*> calloc(self.th.th_size,sizeof(double))

        for iz, redshift in enumerate(zarr):
          if background_at_z(&self.ba,redshift,long_info,inter_normal,&last_index,pvecback)==_FAILURE_:
              free(pvecback) #manual free due to error
              free(pvecthermo) #manual free due to error
              raise CosmoSevereError(self.ba.error_message)

          if thermodynamics_at_z(&self.ba,&self.th,redshift,inter_normal,&last_index,pvecback,pvecthermo) == _FAILURE_:
              free(pvecback) #manual free due to error
              free(pvecthermo) #manual free due to error
              raise CosmoSevereError(self.th.error_message)

          Tb[iz] = pvecthermo[self.th.index_th_Tb]

        free(pvecback)
        free(pvecthermo)

        return (Tb[0] if np.isscalar(z) else Tb)

    def T_cmb(self):
        """
        Return the CMB temperature
        """
        return self.ba.T_cmb

    # redundent with a previous Omega_m() funciton,
    # but we leave it not to break compatibility
    def Omega0_m(self):
        """
        Return the sum of Omega0 for all non-relativistic components
        """
        return self.ba.Omega0_m

    def get_background(self):
        """
        Return an array of the background quantities at all times.

        Parameters
        ----------

        Returns
        -------
        background : dictionary containing background.
        """
        self.compute(["background"])

        cdef char *titles
        cdef double* data
        titles = <char*>calloc(_MAXTITLESTRINGLENGTH_,sizeof(char))

        if background_output_titles(&self.ba, titles)==_FAILURE_:
            free(titles) #manual free due to error
            raise CosmoSevereError(self.ba.error_message)

        tmp = <bytes> titles
        tmp = str(tmp.decode())
        names = tmp.split("\t")[:-1]
        number_of_titles = len(names)
        timesteps = self.ba.bt_size

        data = <double*>malloc(sizeof(double)*timesteps*number_of_titles)

        if background_output_data(&self.ba, number_of_titles, data)==_FAILURE_:
            free(titles) #manual free due to error
            free(data) #manual free due to error
            raise CosmoSevereError(self.ba.error_message)

        background = {}

        for i in range(number_of_titles):
            background[names[i]] = np.zeros(timesteps, dtype=np.double)
            for index in range(timesteps):
                background[names[i]][index] = data[index*number_of_titles+i]

        free(titles)
        free(data)
        return background

    def get_thermodynamics(self):
        """
        Return the thermodynamics quantities.

        Returns
        -------
        thermodynamics : dictionary containing thermodynamics.
        """
        self.compute(["thermodynamics"])

        cdef char *titles
        cdef double* data

        titles = <char*>calloc(_MAXTITLESTRINGLENGTH_,sizeof(char))

        if thermodynamics_output_titles(&self.ba, &self.th, titles)==_FAILURE_:
            free(titles) #manual free due to error
            raise CosmoSevereError(self.th.error_message)

        tmp = <bytes> titles
        tmp = str(tmp.decode())
        names = tmp.split("\t")[:-1]
        number_of_titles = len(names)
        timesteps = self.th.tt_size

        data = <double*>malloc(sizeof(double)*timesteps*number_of_titles)

        if thermodynamics_output_data(&self.ba, &self.th, number_of_titles, data)==_FAILURE_:
            free(titles) #manual free due to error
            free(data) #manual free due to error
            raise CosmoSevereError(self.th.error_message)

        thermodynamics = {}

        for i in range(number_of_titles):
            thermodynamics[names[i]] = np.zeros(timesteps, dtype=np.double)
            for index in range(timesteps):
                thermodynamics[names[i]][index] = data[index*number_of_titles+i]

        free(titles)
        free(data)
        return thermodynamics

    def get_primordial(self):
        """
        Return the primordial scalar and/or tensor spectrum depending on 'modes'.
        'output' must be set to something, e.g. 'tCl'.

        Returns
        -------
        primordial : dictionary containing k-vector and primordial scalar and tensor P(k).
        """
        self.compute(["primordial"])

        cdef char *titles
        cdef double* data

        titles = <char*>calloc(_MAXTITLESTRINGLENGTH_,sizeof(char))

        if primordial_output_titles(&self.pt, &self.pm, titles)==_FAILURE_:
            free(titles) #manual free due to error
            raise CosmoSevereError(self.pm.error_message)

        tmp = <bytes> titles
        tmp = str(tmp.decode())
        names = tmp.split("\t")[:-1]
        number_of_titles = len(names)
        timesteps = self.pm.lnk_size

        data = <double*>malloc(sizeof(double)*timesteps*number_of_titles)

        if primordial_output_data(&self.pt, &self.pm, number_of_titles, data)==_FAILURE_:
            free(titles) #manual free due to error
            free(data) #manual free due to error
            raise CosmoSevereError(self.pm.error_message)

        primordial = {}

        for i in range(number_of_titles):
            primordial[names[i]] = np.zeros(timesteps, dtype=np.double)
            for index in range(timesteps):
                primordial[names[i]][index] = data[index*number_of_titles+i]

        free(titles)
        free(data)
        return primordial

    def get_perturbations(self, return_copy=True):
        """
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
        """
        self.compute(["perturbations"])

        perturbations = {}

        if self.pt.k_output_values_num<1:
            return perturbations

        cdef:
            Py_ssize_t j
            Py_ssize_t i
            Py_ssize_t number_of_titles
            Py_ssize_t timesteps
            list names
            list tmparray
            dict tmpdict
            double[:,::1] data_mv
            double ** thedata
            int * thesizes

        # Doing the exact same thing 3 times, for scalar, vector and tensor. Sorry
        # for copy-and-paste here, but I don't know what else to do.
        for mode in ['scalar','vector','tensor']:
            if mode=='scalar' and self.pt.has_scalars:
                thetitles = <bytes> self.pt.scalar_titles
                thedata = self.pt.scalar_perturbations_data
                thesizes = self.pt.size_scalar_perturbation_data
            elif mode=='vector' and self.pt.has_vectors:
                thetitles = <bytes> self.pt.vector_titles
                thedata = self.pt.vector_perturbations_data
                thesizes = self.pt.size_vector_perturbation_data
            elif mode=='tensor' and self.pt.has_tensors:
                thetitles = <bytes> self.pt.tensor_titles
                thedata = self.pt.tensor_perturbations_data
                thesizes = self.pt.size_tensor_perturbation_data
            else:
                continue
            thetitles = str(thetitles.decode())
            names = thetitles.split("\t")[:-1]
            number_of_titles = len(names)
            tmparray = []
            if number_of_titles != 0:
                for j in range(self.pt.k_output_values_num):
                    timesteps = thesizes[j]//number_of_titles
                    tmpdict={}
                    data_mv = <double[:timesteps,:number_of_titles]> thedata[j]
                    for i in range(number_of_titles):
                        tmpdict[names[i]] = (np.asarray(data_mv[:,i]).copy() if return_copy else np.asarray(data_mv[:,i]))
                    tmparray.append(tmpdict)
            perturbations[mode] = tmparray

        return perturbations

    def get_transfer(self, z=0., output_format='class'):
        """
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
        """
        self.compute(["transfer"])

        cdef char *titles
        cdef double* data
        cdef char ic_info[1024]
        cdef FileName ic_suffix
        cdef file_format outf

        if (not self.pt.has_density_transfers) and (not self.pt.has_velocity_transfers):
            return {}

        if output_format == 'camb':
            outf = camb_format
        else:
            outf = class_format

        index_md = self.pt.index_md_scalars;
        titles = <char*>calloc(_MAXTITLESTRINGLENGTH_,sizeof(char))

        if perturbations_output_titles(&self.ba,&self.pt, outf, titles)==_FAILURE_:
            free(titles) #manual free due to error
            raise CosmoSevereError(self.pt.error_message)

        tmp = <bytes> titles
        tmp = str(tmp.decode())
        names = tmp.split("\t")[:-1]
        number_of_titles = len(names)
        timesteps = self.pt.k_size[index_md]

        size_ic_data = timesteps*number_of_titles;
        ic_num = self.pt.ic_size[index_md];

        data = <double*>malloc(sizeof(double)*size_ic_data*ic_num)

        if perturbations_output_data_at_z(&self.ba, &self.pt, outf, <double> z, number_of_titles, data)==_FAILURE_:
            raise CosmoSevereError(self.pt.error_message)

        transfers = {}

        for index_ic in range(ic_num):
            if perturbations_output_firstline_and_ic_suffix(&self.pt, index_ic, ic_info, ic_suffix)==_FAILURE_:
                free(titles) #manual free due to error
                free(data) #manual free due to error
                raise CosmoSevereError(self.pt.error_message)
            ic_key = <bytes> ic_suffix

            tmpdict = {}
            for i in range(number_of_titles):
                tmpdict[names[i]] = np.zeros(timesteps, dtype=np.double)
                for index in range(timesteps):
                    tmpdict[names[i]][index] = data[index_ic*size_ic_data+index*number_of_titles+i]

            if ic_num==1:
                transfers = tmpdict
            else:
                transfers[ic_key] = tmpdict

        free(titles)
        free(data)

        return transfers

    def get_current_derived_parameters(self, names):
        """
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

        """
        if type(names) != type([]):
            raise TypeError("Deprecated")

        self.compute(["thermodynamics"])

        derived = {}
        for name in names:
            if name == 'h':
                value = self.ba.h
            elif name == 'H0':
                value = self.ba.h*100
            elif name == 'Omega0_lambda' or name == 'Omega_Lambda':
                value = self.ba.Omega0_lambda
            elif name == 'Omega0_fld':
                value = self.ba.Omega0_fld
            elif name == 'age':
                value = self.ba.age
            elif name == 'conformal_age':
                value = self.ba.conformal_age
            elif name == 'm_ncdm_in_eV':
                value = self.ba.m_ncdm_in_eV[0]
            elif name == 'm_ncdm_tot':
                value = self.ba.Omega0_ncdm_tot*self.ba.h*self.ba.h*93.14
            elif name == 'Neff':
                value = self.ba.Neff
            elif name == 'Omega_m':
                value = self.ba.Omega0_m
            elif name == 'omega_m':
                value = self.ba.Omega0_m*self.ba.h**2
            elif name == 'xi_idr':
                value = self.ba.T_idr/self.ba.T_cmb
            elif name == 'N_dg':
                value = self.ba.Omega0_idr/self.ba.Omega0_g*8./7.*pow(11./4.,4./3.)
            elif name == 'Gamma_0_nadm':
                value = self.th.a_idm_dr*(4./3.)*(self.ba.h*self.ba.h*self.ba.Omega0_idr)
            elif name == 'a_dark':
                value = self.th.a_idm_dr
            elif name == 'tau_reio':
                value = self.th.tau_reio
            elif name == 'z_reio':
                value = self.th.z_reio
            elif name == 'z_rec':
                value = self.th.z_rec
            elif name == 'tau_rec':
                value = self.th.tau_rec
            elif name == 'rs_rec':
                value = self.th.rs_rec
            elif name == 'rs_rec_h':
                value = self.th.rs_rec*self.ba.h
            elif name == 'ds_rec':
                value = self.th.ds_rec
            elif name == 'ds_rec_h':
                value = self.th.ds_rec*self.ba.h
            elif name == 'ra_rec':
                value = self.th.da_rec*(1.+self.th.z_rec)
            elif name == 'ra_rec_h':
                value = self.th.da_rec*(1.+self.th.z_rec)*self.ba.h
            elif name == 'da_rec':
                value = self.th.da_rec
            elif name == 'da_rec_h':
                value = self.th.da_rec*self.ba.h
            elif name == 'z_star':
                value = self.th.z_star
            elif name == 'tau_star':
                value = self.th.tau_star
            elif name == 'rs_star':
                value = self.th.rs_star
            elif name == 'ds_star':
                value = self.th.ds_star
            elif name == 'ra_star':
                value = self.th.ra_star
            elif name == 'da_star':
                value = self.th.da_star
            elif name == 'rd_star':
                value = self.th.rd_star
            elif name == 'z_d':
                value = self.th.z_d
            elif name == 'tau_d':
                value = self.th.tau_d
            elif name == 'ds_d':
                value = self.th.ds_d
            elif name == 'ds_d_h':
                value = self.th.ds_d*self.ba.h
            elif name == 'rs_d':
                value = self.th.rs_d
            elif name == 'rs_d_h':
                value = self.th.rs_d*self.ba.h
            elif name == 'conf_time_reio':
                value = self.th.conf_time_reio
            elif name == '100*theta_s':
                value = 100.*self.th.rs_rec/self.th.da_rec/(1.+self.th.z_rec)
            elif name == '100*theta_star':
                value = 100.*self.th.rs_star/self.th.da_star/(1.+self.th.z_star)
            elif name == 'theta_s_100':
                value = 100.*self.th.rs_rec/self.th.da_rec/(1.+self.th.z_rec)
            elif name == 'theta_star_100':
                value = 100.*self.th.rs_star/self.th.da_star/(1.+self.th.z_star)
            elif name == 'YHe':
                value = self.th.YHe
            elif name == 'n_e':
                value = self.th.n_e
            elif name == 'A_s':
                value = self.pm.A_s
            elif name == 'ln10^{10}A_s':
                value = log(1.e10*self.pm.A_s)
            elif name == 'ln_A_s_1e10':
                value = log(1.e10*self.pm.A_s)
            elif name == 'n_s':
                value = self.pm.n_s
            elif name == 'alpha_s':
                value = self.pm.alpha_s
            elif name == 'beta_s':
                value = self.pm.beta_s
            elif name == 'r':
                # This is at the pivot scale
                value = self.pm.r
            elif name == 'r_0002':
                # at k_pivot = 0.002/Mpc
                value = self.pm.r*(0.002/self.pm.k_pivot)**(
                    self.pm.n_t-self.pm.n_s-1+0.5*self.pm.alpha_s*log(
                        0.002/self.pm.k_pivot))
            elif name == 'n_t':
                value = self.pm.n_t
            elif name == 'alpha_t':
                value = self.pm.alpha_t
            elif name == 'V_0':
                value = self.pm.V0
            elif name == 'V_1':
                value = self.pm.V1
            elif name == 'V_2':
                value = self.pm.V2
            elif name == 'V_3':
                value = self.pm.V3
            elif name == 'V_4':
                value = self.pm.V4
            elif name == 'epsilon_V':
                eps1 = self.pm.r*(1./16.-0.7296/16.*(self.pm.r/8.+self.pm.n_s-1.))
                eps2 = -self.pm.n_s+1.-0.7296*self.pm.alpha_s-self.pm.r*(1./8.+1./8.*(self.pm.n_s-1.)*(-0.7296-1.5))-(self.pm.r/8.)**2*(-0.7296-1.)
                value = eps1*((1.-eps1/3.+eps2/6.)/(1.-eps1/3.))**2
            elif name == 'eta_V':
                eps1 = self.pm.r*(1./16.-0.7296/16.*(self.pm.r/8.+self.pm.n_s-1.))
                eps2 = -self.pm.n_s+1.-0.7296*self.pm.alpha_s-self.pm.r*(1./8.+1./8.*(self.pm.n_s-1.)*(-0.7296-1.5))-(self.pm.r/8.)**2*(-0.7296-1.)
                eps23 = 1./8.*(self.pm.r**2/8.+(self.pm.n_s-1.)*self.pm.r-8.*self.pm.alpha_s)
                value = (2.*eps1-eps2/2.-2./3.*eps1**2+5./6.*eps1*eps2-eps2**2/12.-eps23/6.)/(1.-eps1/3.)
            elif name == 'ksi_V^2':
                eps1 = self.pm.r*(1./16.-0.7296/16.*(self.pm.r/8.+self.pm.n_s-1.))
                eps2 = -self.pm.n_s+1.-0.7296*self.pm.alpha_s-self.pm.r*(1./8.+1./8.*(self.pm.n_s-1.)*(-0.7296-1.5))-(self.pm.r/8.)**2*(-0.7296-1.)
                eps23 = 1./8.*(self.pm.r**2/8.+(self.pm.n_s-1.)*self.pm.r-8.*self.pm.alpha_s)
                value = 2.*(1.-eps1/3.+eps2/6.)*(2.*eps1**2-3./2.*eps1*eps2+eps23/4.)/(1.-eps1/3.)**2
            elif name == 'exp_m_2_tau_As':
                value = exp(-2.*self.th.tau_reio)*self.pm.A_s
            elif name == 'phi_min':
                value = self.pm.phi_min
            elif name == 'phi_max':
                value = self.pm.phi_max
            elif name == 'sigma8':
                self.compute(["fourier"])
                if (self.pt.has_pk_matter == _FALSE_):
                    raise CosmoSevereError("No power spectrum computed. In order to get sigma8, you must add mPk to the list of outputs.")
                value = self.fo.sigma8[self.fo.index_pk_m]
            elif name == 'sigma8_cb':
                self.compute(["fourier"])
                if (self.pt.has_pk_matter == _FALSE_):
                    raise CosmoSevereError("No power spectrum computed. In order to get sigma8_cb, you must add mPk to the list of outputs.")
                value = self.fo.sigma8[self.fo.index_pk_cb]
            elif name == 'k_eq':
                value = self.ba.a_eq*self.ba.H_eq
            elif name == 'a_eq':
                value = self.ba.a_eq
            elif name == 'z_eq':
                value = 1./self.ba.a_eq-1.
            elif name == 'H_eq':
                value = self.ba.H_eq
            elif name == 'tau_eq':
                value = self.ba.tau_eq
            elif name == 'g_sd':
                self.compute(["distortions"])
                if (self.sd.has_distortions == _FALSE_):
                    raise CosmoSevereError("No spectral distortions computed. In order to get g_sd, you must add sd to the list of outputs.")
                value = self.sd.sd_parameter_table[0]
            elif name == 'y_sd':
                self.compute(["distortions"])
                if (self.sd.has_distortions == _FALSE_):
                    raise CosmoSevereError("No spectral distortions computed. In order to get y_sd, you must add sd to the list of outputs.")
                value = self.sd.sd_parameter_table[1]
            elif name == 'mu_sd':
                self.compute(["distortions"])
                if (self.sd.has_distortions == _FALSE_):
                    raise CosmoSevereError("No spectral distortions computed. In order to get mu_sd, you must add sd to the list of outputs.")
                value = self.sd.sd_parameter_table[2]
            else:
                raise CosmoSevereError("%s was not recognized as a derived parameter" % name)
            derived[name] = value
        return derived

    def nonlinear_scale(self, np.ndarray[DTYPE_t,ndim=1] z, int z_size):
        """
        nonlinear_scale(z, z_size)

        Return the nonlinear scale for all the redshift specified in z, of size
        z_size

        Parameters
        ----------
        z : numpy array
                Array of requested redshifts
        z_size : int
                Size of the redshift array
        """
        self.compute(["fourier"])

        cdef int index_z
        cdef np.ndarray[DTYPE_t, ndim=1] k_nl = np.zeros(z_size,'float64')
        cdef np.ndarray[DTYPE_t, ndim=1] k_nl_cb = np.zeros(z_size,'float64')
        #cdef double *k_nl
        #k_nl = <double*> calloc(z_size,sizeof(double))
        for index_z in range(z_size):
            if fourier_k_nl_at_z(&self.ba,&self.fo,z[index_z],&k_nl[index_z],&k_nl_cb[index_z]) == _FAILURE_:
                raise CosmoSevereError(self.fo.error_message)

        return k_nl

    def nonlinear_scale_cb(self, np.ndarray[DTYPE_t,ndim=1] z, int z_size):
        """

make        nonlinear_scale_cb(z, z_size)

        Return the nonlinear scale for all the redshift specified in z, of size

        z_size

        Parameters
        ----------
        z : numpy array
                Array of requested redshifts
        z_size : int
                Size of the redshift array
        """
        self.compute(["fourier"])

        cdef int index_z
        cdef np.ndarray[DTYPE_t, ndim=1] k_nl = np.zeros(z_size,'float64')
        cdef np.ndarray[DTYPE_t, ndim=1] k_nl_cb = np.zeros(z_size,'float64')
        #cdef double *k_nl
        #k_nl = <double*> calloc(z_size,sizeof(double))
        if (self.ba.Omega0_ncdm_tot == 0.):
            raise CosmoSevereError(
                "No massive neutrinos. You must use pk, rather than pk_cb."
                )
        for index_z in range(z_size):
            if fourier_k_nl_at_z(&self.ba,&self.fo,z[index_z],&k_nl[index_z],&k_nl_cb[index_z]) == _FAILURE_:
                raise CosmoSevereError(self.fo.error_message)

        return k_nl_cb

    def __call__(self, ctx):
        """
        Function to interface with CosmoHammer

        Parameters
        ----------
        ctx : context
                Contains several dictionaries storing data and cosmological
                information

        """
        data = ctx.get('data')  # recover data from the context

        # If the module has already been called once, clean-up
        if self.state:
            self.struct_cleanup()

        # Set the module to the current values
        self.set(data.cosmo_arguments)
        self.compute(["lensing"])

        # Compute the derived paramter value and store them
        params = ctx.getData()
        self.get_current_derived_parameters(
            data.get_mcmc_parameters(['derived']))
        for elem in data.get_mcmc_parameters(['derived']):
            data.mcmc_parameters[elem]['current'] /= \
                data.mcmc_parameters[elem]['scale']
            params[elem] = data.mcmc_parameters[elem]['current']

        ctx.add('boundary', True)
        # Store itself into the context, to be accessed by the likelihoods
        ctx.add('cosmo', self)

    def get_pk_array(self, np.ndarray[DTYPE_t,ndim=1] k, np.ndarray[DTYPE_t,ndim=1] z, int k_size, int z_size, nonlinear):
        """ Fast function to get the power spectrum on a k and z array """
        self.compute(["fourier"])
        cdef np.ndarray[DTYPE_t, ndim=1] pk = np.zeros(k_size*z_size,'float64')
        cdef np.ndarray[DTYPE_t, ndim=1] pk_cb = np.zeros(k_size*z_size,'float64')

        if nonlinear == 0:
            fourier_pks_at_kvec_and_zvec(&self.ba, &self.fo, pk_linear, <double*> k.data, k_size, <double*> z.data, z_size, <double*> pk.data, <double*> pk_cb.data)

        else:
            fourier_pks_at_kvec_and_zvec(&self.ba, &self.fo, pk_nonlinear, <double*> k.data, k_size, <double*> z.data, z_size, <double*> pk.data, <double*> pk_cb.data)

        return pk

    def get_pk_cb_array(self, np.ndarray[DTYPE_t,ndim=1] k, np.ndarray[DTYPE_t,ndim=1] z, int k_size, int z_size, nonlinear):
        """ Fast function to get the power spectrum on a k and z array """
        self.compute(["fourier"])
        cdef np.ndarray[DTYPE_t, ndim=1] pk = np.zeros(k_size*z_size,'float64')
        cdef np.ndarray[DTYPE_t, ndim=1] pk_cb = np.zeros(k_size*z_size,'float64')

        if nonlinear == 0:
            fourier_pks_at_kvec_and_zvec(&self.ba, &self.fo, pk_linear, <double*> k.data, k_size, <double*> z.data, z_size, <double*> pk.data, <double*> pk_cb.data)

        else:
            fourier_pks_at_kvec_and_zvec(&self.ba, &self.fo, pk_nonlinear, <double*> k.data, k_size, <double*> z.data, z_size, <double*> pk.data, <double*> pk_cb.data)

        return pk_cb

    def Omega0_k(self):
        """ Curvature contribution """
        return self.ba.Omega0_k

    def Omega0_cdm(self):
        return self.ba.Omega0_cdm

    def spectral_distortion_amplitudes(self):
        self.compute(["distortions"])
        if self.sd.type_size == 0:
          raise CosmoSevereError("No spectral distortions have been calculated. Check that the output contains 'Sd' and the compute level is at least 'distortions'.")
        cdef np.ndarray[DTYPE_t, ndim=1] sd_type_amps = np.zeros(self.sd.type_size,'float64')
        for i in range(self.sd.type_size):
          sd_type_amps[i] = self.sd.sd_parameter_table[i]
        return sd_type_amps

    def spectral_distortion(self):
        self.compute(["distortions"])
        if self.sd.x_size == 0:
          raise CosmoSevereError("No spectral distortions have been calculated. Check that the output contains 'Sd' and the compute level is at least 'distortions'.")
        cdef np.ndarray[DTYPE_t, ndim=1] sd_amp = np.zeros(self.sd.x_size,'float64')
        cdef np.ndarray[DTYPE_t, ndim=1] sd_nu = np.zeros(self.sd.x_size,'float64')
        for i in range(self.sd.x_size):
          sd_amp[i] = self.sd.DI[i]*self.sd.DI_units*1.e26
          sd_nu[i] = self.sd.x[i]*self.sd.x_to_nu
        return sd_nu,sd_amp


    def get_sources(self):
        """
        Return the source functions for all k, tau in the grid.

        Returns
        -------
        sources : dictionary containing source functions.
        k_array : numpy array containing k values.
        tau_array: numpy array containing tau values.
        """
        self.compute(["fourier"])
        sources = {}

        cdef:
            int index_k, index_tau, i_index_type;
            int index_type;
            int index_md = self.pt.index_md_scalars;
            double * k = self.pt.k[index_md];
            double * tau = self.pt.tau_sampling;
            int index_ic = self.pt.index_ic_ad;
            int k_size = self.pt.k_size[index_md];
            int tau_size = self.pt.tau_size;
            int tp_size = self.pt.tp_size[index_md];
            double *** sources_ptr = self.pt.sources;
            double [:,:] tmparray = np.zeros((k_size, tau_size)) ;
            double [:] k_array = np.zeros(k_size);
            double [:] tau_array = np.zeros(tau_size);

        names = []

        for index_k in range(k_size):
            k_array[index_k] = k[index_k]
        for index_tau in range(tau_size):
            tau_array[index_tau] = tau[index_tau]

        indices = []

        if self.pt.has_source_t:
            indices.extend([
                self.pt.index_tp_t0,
                self.pt.index_tp_t1,
                self.pt.index_tp_t2
                ])
            names.extend([
                "t0",
                "t1",
                "t2"
                ])
        if self.pt.has_source_p:
            indices.append(self.pt.index_tp_p)
            names.append("p")
        if self.pt.has_source_phi:
            indices.append(self.pt.index_tp_phi)
            names.append("phi")
        if self.pt.has_source_phi_plus_psi:
            indices.append(self.pt.index_tp_phi_plus_psi)
            names.append("phi_plus_psi")
        if self.pt.has_source_phi_prime:
            indices.append(self.pt.index_tp_phi_prime)
            names.append("phi_prime")
        if self.pt.has_source_psi:
            indices.append(self.pt.index_tp_psi)
            names.append("psi")
        if self.pt.has_source_H_T_Nb_prime:
            indices.append(self.pt.index_tp_H_T_Nb_prime)
            names.append("H_T_Nb_prime")
        if self.pt.index_tp_k2gamma_Nb:
            indices.append(self.pt.index_tp_k2gamma_Nb)
            names.append("k2gamma_Nb")
        if self.pt.has_source_h:
            indices.append(self.pt.index_tp_h)
            names.append("h")
        if self.pt.has_source_h_prime:
            indices.append(self.pt.index_tp_h_prime)
            names.append("h_prime")
        if self.pt.has_source_eta:
            indices.append(self.pt.index_tp_eta)
            names.append("eta")
        if self.pt.has_source_eta_prime:
            indices.append(self.pt.index_tp_eta_prime)
            names.append("eta_prime")
        if self.pt.has_source_delta_tot:
            indices.append(self.pt.index_tp_delta_tot)
            names.append("delta_tot")
        if self.pt.has_source_delta_m:
            indices.append(self.pt.index_tp_delta_m)
            names.append("delta_m")
        if self.pt.has_source_delta_cb:
            indices.append(self.pt.index_tp_delta_cb)
            names.append("delta_cb")
        if self.pt.has_source_delta_g:
            indices.append(self.pt.index_tp_delta_g)
            names.append("delta_g")
        if self.pt.has_source_delta_b:
            indices.append(self.pt.index_tp_delta_b)
            names.append("delta_b")
        if self.pt.has_source_delta_cdm:
            indices.append(self.pt.index_tp_delta_cdm)
            names.append("delta_cdm")
        if self.pt.has_source_delta_idm:
            indices.append(self.pt.index_tp_delta_idm)
            names.append("delta_idm")
        if self.pt.has_source_delta_dcdm:
            indices.append(self.pt.index_tp_delta_dcdm)
            names.append("delta_dcdm")
        if self.pt.has_source_delta_fld:
            indices.append(self.pt.index_tp_delta_fld)
            names.append("delta_fld")
        if self.pt.has_source_delta_scf:
            indices.append(self.pt.index_tp_delta_scf)
            names.append("delta_scf")
        if self.pt.has_source_delta_dr:
            indices.append(self.pt.index_tp_delta_dr)
            names.append("delta_dr")
        if self.pt.has_source_delta_ur:
            indices.append(self.pt.index_tp_delta_ur)
            names.append("delta_ur")
        if self.pt.has_source_delta_idr:
            indices.append(self.pt.index_tp_delta_idr)
            names.append("delta_idr")
        if self.pt.has_source_delta_ncdm:
            for incdm in range(self.ba.N_ncdm):
              indices.append(self.pt.index_tp_delta_ncdm1+incdm)
              names.append("delta_ncdm[{}]".format(incdm))
        if self.pt.has_source_theta_tot:
            indices.append(self.pt.index_tp_theta_tot)
            names.append("theta_tot")
        if self.pt.has_source_theta_m:
            indices.append(self.pt.index_tp_theta_m)
            names.append("theta_m")
        if self.pt.has_source_theta_cb:
            indices.append(self.pt.index_tp_theta_cb)
            names.append("theta_cb")
        if self.pt.has_source_theta_g:
            indices.append(self.pt.index_tp_theta_g)
            names.append("theta_g")
        if self.pt.has_source_theta_b:
            indices.append(self.pt.index_tp_theta_b)
            names.append("theta_b")
        if self.pt.has_source_theta_cdm:
            indices.append(self.pt.index_tp_theta_cdm)
            names.append("theta_cdm")
        if self.pt.has_source_theta_idm:
            indices.append(self.pt.index_tp_theta_idm)
            names.append("theta_idm")
        if self.pt.has_source_theta_dcdm:
            indices.append(self.pt.index_tp_theta_dcdm)
            names.append("theta_dcdm")
        if self.pt.has_source_theta_fld:
            indices.append(self.pt.index_tp_theta_fld)
            names.append("theta_fld")
        if self.pt.has_source_theta_scf:
            indices.append(self.pt.index_tp_theta_scf)
            names.append("theta_scf")
        if self.pt.has_source_theta_dr:
            indices.append(self.pt.index_tp_theta_dr)
            names.append("theta_dr")
        if self.pt.has_source_theta_ur:
            indices.append(self.pt.index_tp_theta_ur)
            names.append("theta_ur")
        if self.pt.has_source_theta_idr:
            indices.append(self.pt.index_tp_theta_idr)
            names.append("theta_idr")
        if self.pt.has_source_theta_ncdm:
            for incdm in range(self.ba.N_ncdm):
              indices.append(self.pt.index_tp_theta_ncdm1+incdm)
              names.append("theta_ncdm[{}]".format(incdm))

        for index_type, name in zip(indices, names):
            tmparray = np.empty((k_size,tau_size))
            for index_k in range(k_size):
                for index_tau in range(tau_size):
                    tmparray[index_k][index_tau] = sources_ptr[index_md][index_ic*tp_size+index_type][index_tau*k_size + index_k];

            sources[name] = np.asarray(tmparray)

        return (sources, np.asarray(k_array), np.asarray(tau_array))

```

## cltt_terms.py

```python
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import necessary modules
from classy import Class
from math import pi
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt


# In[ ]:


#############################################
#
# Cosmological parameters and other CLASS parameters
#
common_settings = {# LambdaCDM parameters
                   'h':0.67810,
                   'omega_b':0.02238280,
                   'omega_cdm':0.1201075,
                   'A_s':2.100549e-09,
                   'n_s':0.9660499,
                   'tau_reio':0.05430842 ,
                   # output and precision parameters
                   'output':'tCl,pCl,lCl',
                   'lensing':'yes',
                   'l_max_scalars':5000}
#
M = Class()
#
###############
#    
# call CLASS for the total Cl's and then for each contribution
#
###############
#
M.set(common_settings)
M.compute()
cl_tot = M.raw_cl(3000)
cl_lensed = M.lensed_cl(3000)
M.empty()           # reset input
#
M.set(common_settings) # new input
M.set({'temperature contributions':'tsw'}) 
M.compute()
cl_tsw = M.raw_cl(3000) 
M.empty()
#
M.set(common_settings)
M.set({'temperature contributions':'eisw'})
M.compute()
cl_eisw = M.raw_cl(3000) 
M.empty()
#
M.set(common_settings)
M.set({'temperature contributions':'lisw'})
M.compute()
cl_lisw = M.raw_cl(3000) 
M.empty()
#
M.set(common_settings)
M.set({'temperature contributions':'dop'})
M.compute()
cl_dop = M.raw_cl(3000) 
M.empty()


# In[ ]:


#################
#
# start plotting
#
#################
#
plt.xlim([2,3000])
plt.xlabel(r"$\ell$")
plt.ylabel(r"$\ell (\ell+1) C_l^{TT} / 2 \pi \,\,\, [\times 10^{10}]$")
plt.grid()
#
ell = cl_tot['ell']
factor = 1.e10*ell*(ell+1.)/2./pi
plt.semilogx(ell,factor*cl_tsw['tt'],'c-',label=r'$\mathrm{T+SW}$')
plt.semilogx(ell,factor*cl_eisw['tt'],'r-',label=r'$\mathrm{early-ISW}$')
plt.semilogx(ell,factor*cl_lisw['tt'],'y-',label=r'$\mathrm{late-ISW}$')
plt.semilogx(ell,factor*cl_dop['tt'],'g-',label=r'$\mathrm{Doppler}$')
plt.semilogx(ell,factor*cl_tot['tt'],'r-',label=r'$\mathrm{total}$')
plt.semilogx(ell,factor*cl_lensed['tt'],'k-',label=r'$\mathrm{lensed}$')
#
plt.legend(loc='right',bbox_to_anchor=(1.4, 0.5))
plt.savefig('cltt_terms.pdf',bbox_inches='tight')


```

## cosmology_validation.py

```python
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





```

## default.ini

```
# *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
# *  CLASS input parameter file  *
# *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*

# This example of input file, intended for CLASS beginners, lists only
# the most common input parameters with small comments. Only lines
# containing an equal sign not preceded by a sharp sign "#" are considered by
# the code, any other line is considered as a comment.
#
# The normal syntax is:      parameter = value(s)


# -------------------------
# ----> General parameters:
# -------------------------

# REQUESTED OUTPUT FROM CLASS (Important!)
#  - 'tCl' for temperature Cls,
#  - 'pCl' for polarization (TE,BB,EE) Cls,
#  - 'lCl' for CMB lensing POTENTIAL Cl (Cl^psi-psi, required for lensed Cls),
#  - 'nCl' (or 'dCl') for density number count Cls,
#  - 'sCl' for galaxy lensing potential Cls (Cl^phi-phi),
#  - 'mPk' for total matter power spectrum P(k),
#  - 'dTk' for density transfer functions,
#  - 'vTk' for velocity transfer functions,
#  - 'Sd' for spectral distortions
#        --> deflection d: Cl^dd = l(l+1) C_l^phi-phi
#        --> convergence kappa and shear gamma: the share the same harmonic
#            power spectrum: Cl^gamma-gamma = 1/4 * [(l+2)!/(l-2)!] C_l^phi-phi
output = tCl,pCl,lCl,mPk
#output = tCl,pCl,lCl
#output = mPk,mTk
#output = Sd

lensing = yes                     # Should the Cls from above be lensed for CMB?
lcmb_rescale = 1                  # Amplitude rescale of lensing only
lcmb_tilt = 0                     # CMB l tilt of lensing
lcmb_pivot = 0.1                  # CMB l pivot of lensing

non_linear =                      # Select 'halofit' or 'HMCode' or leave blank

ic = ad                           # Select initial conditions
#(ad,bi,cdi,nid,nvi) -> (adiabatic,baryon,cdm,neutrino density,neutrino velocity)
modes = s                         # Modes of the perturbations
# (s,v,t) -> (scalar,vector,tensor)

#number_count_contributions =     # nCl contributions
#(density,lensing,rsd,gr) -> (density, lensing, rsd+doppler, all others)
#selection=gaussian               # nCl window function type
#selection_mean=1.0,1.25,2.0,3.5  # Mean redshifts of nCl window functions
#selection_width = 0.1            # Widths of nCl window functions
#selection_bias =                 # Biases of nCl window functions
#selection_magnification_bias =   # Biases of lensing of nCl
#non_diagonal=3                   # Number of non-diagonal terms

l_max_scalars = 2500              # lmax of CMB for scalar mode
#l_max_tensors = 500              # lmax of CMB for tensor mode
#l_max_lss = 300                  # lmax of nCl

P_k_max_h/Mpc = 1.                # Maximum k for P(k) in 1/Mpc
#P_k_max_1/Mpc = 0.7              # Maximum k for P(k) in h/Mpc
z_pk = 0                          # Redshifts of P(k,z)

# ----------------------------
# ----> Cosmological parameters:
# ----------------------------

h = 0.67810                       # Dimensionless reduced Hubble parameter (H_0 / (100km/s/Mpc))
#H0 = 67.810                      # Hubble parameter in km/s/Mpc
#100*theta_s = 1.041783           # Angular size of the sound horizon, exactly 100(ds_dec/da_dec)
                                  # with decoupling time given by maximum of visibility function
                                  # (different from theta_MC of CosmoMC and
                                  # slightly different from theta_* of CAMB)
T_cmb = 2.7255                    # CMB temperature

omega_b = 0.02238280              # Reduced baryon density (Omega*h^2)
#Omega_b =                        # Baryon density
omega_cdm = 0.1201075             # Reduced cold dark matter density (Omega*h^2)
#Omega_cdm =                      # CDM density
omega_dcdmdr = 0.0                # Reduced decaying dark matter density (Omega*h^2)
#Omega_dcdmdr =                   # DCDM density
#Gamma_dcdm = 0.0                 # Decay constant of DCDM in km/s/Mpc
Omega_k = 0.                      # Curvature density
Omega_fld = 0                     # Dark Energy as Fluid density
Omega_scf = 0                     # Dark Energy as Scalar field density

# Usually Omega_Lambda will be matched by the budget equation sum Omega_i = 1, no need to set it manually
#Omega_Lambda = 0.7               # Cosmological constant density


# If you have respectively 0,1,2, or 3 MASSIVE neutrinos and the default T_ncdm of 0.71611,
# designed to give M_tot/omega_nu of 93.14 eV, and if you want N_eff equal to 3.044,
# then you should pass for N_ur 3.044,2.0308,1.0176, or 0.00441
N_ur = 3.044                      # Effective number of MASSLESS neutrino species
#Omega_ur =                       # Reduced MASSLESS neutrino density (Omega*h^2)
#omega_ur =                       # MASSLESS neutrino density

N_ncdm =                          # Massive neutrino species
#m_ncdm = 0.06                    # Mass of the massive neutrinos
#omega_ncdm = 0.0006451439        # Reduced massive neutrino density (Omega*h^2)
#Omega_ncdm =                     # Massive neutrino density
#deg_ncdm =                       # Degeneracy of massive neutrinos


### For Omega_fld != 0
# Chevalier-Linder-Polarski => CLP
# Early Dark Energy         => EDE
#fluid_equation_of_state = CLP

#CLP case
#w0_fld = -0.9
#wa_fld = 0.
#cs2_fld = 1
#EDE case
#w0_fld = -0.9
#Omega_EDE = 0.
#cs2_fld = 1

# ----------------------------
# ----> Thermodynamics/Heating parameters:
# ----------------------------

# Infer YHe from BBN. Alternatively provide your own number here
YHe = BBN
# Recombination code : 'RECFAST' or 'HyRec'
recombination = HyRec

z_reio = 7.6711                    # Redshift of reionization
#tau_reio = 0.05430842            # Optical depth of reionization

reio_parametrization = reio_camb
reionization_exponent = 1.5
reionization_width = 0.5
helium_fullreio_redshift = 3.5
helium_fullreio_width = 0.5

### Energy injections
DM_annihilation_cross_section = 0.# Dark Matter annihilation cross section in [cm^3/s]
DM_annihilation_mass = 0.         # Dark Matter mass in [GeV]
DM_decay_fraction = 0.            # Dark Matter decay fraction
DM_decay_Gamma = 0.               # Dark Matter decay width

f_eff_type = on_the_spot          # Injection efficiency
chi_type = CK_2004                # Deposition function

# ----------------------------
# ----> Primordial parameters:
# ----------------------------

P_k_ini type = analytic_Pk        # Select primordial spectrum
#('analytic_Pk','inflation_V','inflation_H','inflation_V_end','two scales','external_Pk')
k_pivot = 0.05                    # Pivot scale for A_s,n_s
A_s = 2.100549e-09                # Amplitude of prim spectrum
#ln10^{10}A_s = 3.0980            # ln Amplitude of prim spectrum
# sigma8 = 0.848365               # Final density averaged over 8 Mpc
n_s = 0.9660499                   # Spectrum tilt
alpha_s = 0.                      # Spectrum running of tilt
#r = 1.                           # If tensors are activated
# See explanatory.ini for more information about all the different primordial spectra

# ---------------------------
# ----> Spectral distortions:
# ---------------------------

sd_branching_approx = exact       # Appriximation for the calculation of the branching ratios
sd_PCA_size = 2                   # Number of multipoles in PCA expansion
sd_detector_name = PIXIE          # Name of the detector
#sd_detector_nu_min = 30.         # Detector specifics
#sd_detector_nu_max = 1000.
#sd_detector_nu_delta = 15.
#sd_detector_bin_number = 65      # Alternative to 'sd_detector_nu_delta'
#sd_detector_delta_Ic = 5.e-26

#include_SZ_effect = no

# ----------------------------------
# ----> Output parameters:
# ----------------------------------

#root = output/default            # Root name of output files
overwrite_root = no               # Overwrite the output files?
write_background = no             # Write background parameter table
write_thermodynamics = no         # Write thermodynamics parameter table
#k_output_values = 1e-3,1e-2      # Write perturbations parameter table (at given k)
write_primordial = no             # Write primordial parameter table
write_parameters = yeap           # Write used/unused parameter files
write_warnings = yes              # Warn about forgotten/wrong inputs

#Verbosity
input_verbose = 1
background_verbose = 1
thermodynamics_verbose = 1
perturbations_verbose = 1
transfer_verbose = 1
primordial_verbose = 1
harmonic_verbose = 1
fourier_verbose = 1
lensing_verbose = 1
output_verbose = 1

```

## default_fast.ini

```
# *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
# *  CLASS input parameter file  *
# *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*

# This example of input file, intended for CLASS beginners, lists only
# the most common input parameters with small comments. Only lines
# containing an equal sign not preceded by a sharp sign "#" are considered by
# the code, any other line is considered as a comment.
#
# The normal syntax is:      parameter = value(s)


# -------------------------
# ----> General parameters:
# -------------------------

# REQUESTED OUTPUT FROM CLASS (Important!)
#  - 'tCl' for temperature Cls,
#  - 'pCl' for polarization (TE,BB,EE) Cls,
#  - 'lCl' for CMB lensing POTENTIAL Cl (Cl^psi-psi, required for lensed Cls),
#  - 'nCl' (or 'dCl') for density number count Cls,
#  - 'sCl' for galaxy lensing potential Cls (Cl^phi-phi),
#  - 'mPk' for total matter power spectrum P(k),
#  - 'dTk' for density transfer functions,
#  - 'vTk' for velocity transfer functions,
#  - 'Sd' for spectral distortions
#        --> deflection d: Cl^dd = l(l+1) C_l^phi-phi
#        --> convergence kappa and shear gamma: the share the same harmonic
#            power spectrum: Cl^gamma-gamma = 1/4 * [(l+2)!/(l-2)!] C_l^phi-phi
output = mPk, tCl
#output = tCl,pCl,lCl
#output = mPk,mTk
#output = Sd

#lensing = yes                     # Should the Cls from above be lensed for CMB?
#lcmb_rescale = 1                  # Amplitude rescale of lensing only
#lcmb_tilt = 0                     # CMB l tilt of lensing
#lcmb_pivot = 0.1                  # CMB l pivot of lensing

non_linear =                      # Select 'halofit' or 'HMCode' or leave blank

ic = ad                           # Select initial conditions
#(ad,bi,cdi,nid,nvi) -> (adiabatic,baryon,cdm,neutrino density,neutrino velocity)
modes = s                         # Modes of the perturbations
# (s,v,t) -> (scalar,vector,tensor)

#number_count_contributions =     # nCl contributions
#(density,lensing,rsd,gr) -> (density, lensing, rsd+doppler, all others)
#selection=gaussian               # nCl window function type
#selection_mean=1.0,1.25,2.0,3.5  # Mean redshifts of nCl window functions
#selection_width = 0.1            # Widths of nCl window functions
#selection_bias =                 # Biases of nCl window functions
#selection_magnification_bias =   # Biases of lensing of nCl
#non_diagonal=3                   # Number of non-diagonal terms

l_max_scalars = 2500              # lmax of CMB for scalar mode
#l_max_tensors = 500              # lmax of CMB for tensor mode
#l_max_lss = 300                  # lmax of nCl

P_k_max_h/Mpc = 50.                # Maximum k for P(k) in 1/Mpc
#P_k_max_1/Mpc = 0.7              # Maximum k for P(k) in h/Mpc
z_pk = 0,1,2,3,4                          # Redshifts of P(k,z)

# ----------------------------
# ----> Cosmological parameters:
# ----------------------------

h = 0.67810                       # Dimensionless reduced Hubble parameter (H_0 / (100km/s/Mpc))
#H0 = 67.810                      # Hubble parameter in km/s/Mpc
#100*theta_s = 1.041783           # Angular size of the sound horizon, exactly 100(ds_dec/da_dec)
                                  # with decoupling time given by maximum of visibility function
                                  # (different from theta_MC of CosmoMC and
                                  # slightly different from theta_* of CAMB)
T_cmb = 2.7255                    # CMB temperature

omega_b = 0.02238280              # Reduced baryon density (Omega*h^2)
#Omega_b =                        # Baryon density
omega_cdm = 0.1201075             # Reduced cold dark matter density (Omega*h^2)
#Omega_cdm =                      # CDM density
omega_dcdmdr = 0.0                # Reduced decaying dark matter density (Omega*h^2)
#Omega_dcdmdr =                   # DCDM density
#Gamma_dcdm = 0.0                 # Decay constant of DCDM in km/s/Mpc
Omega_k = 0.                      # Curvature density
Omega_fld = 0                     # Dark Energy as Fluid density
Omega_scf = 0                     # Dark Energy as Scalar field density

# Usually Omega_Lambda will be matched by the budget equation sum Omega_i = 1, no need to set it manually
#Omega_Lambda = 0.7               # Cosmological constant density


# If you have respectively 0,1,2, or 3 MASSIVE neutrinos and the default T_ncdm of 0.71611,
# designed to give M_tot/omega_nu of 93.14 eV, and if you want N_eff equal to 3.044,
# then you should pass for N_ur 3.044,2.0308,1.0176, or 0.00441
N_ur = 3.044                      # Effective number of MASSLESS neutrino species
#Omega_ur =                       # Reduced MASSLESS neutrino density (Omega*h^2)
#omega_ur =                       # MASSLESS neutrino density

N_ncdm =                          # Massive neutrino species
#m_ncdm = 0.06                    # Mass of the massive neutrinos
#omega_ncdm = 0.0006451439        # Reduced massive neutrino density (Omega*h^2)
#Omega_ncdm =                     # Massive neutrino density
#deg_ncdm =                       # Degeneracy of massive neutrinos


### For Omega_fld != 0
# Chevalier-Linder-Polarski => CLP
# Early Dark Energy         => EDE
#fluid_equation_of_state = CLP

#CLP case
#w0_fld = -0.9
#wa_fld = 0.
#cs2_fld = 1
#EDE case
#w0_fld = -0.9
#Omega_EDE = 0.
#cs2_fld = 1

# ----------------------------
# ----> Thermodynamics/Heating parameters:
# ----------------------------

# Infer YHe from BBN. Alternatively provide your own number here
YHe = BBN
# Recombination code : 'RECFAST' or 'HyRec'
recombination = HyRec

z_reio = 7.6711                    # Redshift of reionization
#tau_reio = 0.05430842            # Optical depth of reionization

reio_parametrization = reio_camb
reionization_exponent = 1.5
reionization_width = 0.5
helium_fullreio_redshift = 3.5
helium_fullreio_width = 0.5

### Energy injections
DM_annihilation_cross_section = 0.# Dark Matter annihilation cross section in [cm^3/s]
DM_annihilation_mass = 0.         # Dark Matter mass in [GeV]
DM_decay_fraction = 0.            # Dark Matter decay fraction
DM_decay_Gamma = 0.               # Dark Matter decay width

f_eff_type = on_the_spot          # Injection efficiency
chi_type = CK_2004                # Deposition function

# ----------------------------
# ----> Primordial parameters:
# ----------------------------

P_k_ini type = analytic_Pk        # Select primordial spectrum
#('analytic_Pk','inflation_V','inflation_H','inflation_V_end','two scales','external_Pk')
k_pivot = 0.05                    # Pivot scale for A_s,n_s
A_s = 2.100549e-09                # Amplitude of prim spectrum
#ln10^{10}A_s = 3.0980            # ln Amplitude of prim spectrum
# sigma8 = 0.848365               # Final density averaged over 8 Mpc
n_s = 0.9660499                   # Spectrum tilt
alpha_s = 0.                      # Spectrum running of tilt
#r = 1.                           # If tensors are activated
# See explanatory.ini for more information about all the different primordial spectra

# ---------------------------
# ----> Spectral distortions:
# ---------------------------

sd_branching_approx = exact       # Appriximation for the calculation of the branching ratios
sd_PCA_size = 2                   # Number of multipoles in PCA expansion
sd_detector_name = PIXIE          # Name of the detector
#sd_detector_nu_min = 30.         # Detector specifics
#sd_detector_nu_max = 1000.
#sd_detector_nu_delta = 15.
#sd_detector_bin_number = 65      # Alternative to 'sd_detector_nu_delta'
#sd_detector_delta_Ic = 5.e-26

#include_SZ_effect = no

# ----------------------------------
# ----> Output parameters:
# ----------------------------------

#root = output/default            # Root name of output files
overwrite_root = no               # Overwrite the output files?
write_background = no             # Write background parameter table
write_thermodynamics = no         # Write thermodynamics parameter table
#k_output_values = 1e-3,1e-2      # Write perturbations parameter table (at given k)
write_primordial = no             # Write primordial parameter table
write_parameters = yeap           # Write used/unused parameter files
write_warnings = yes              # Warn about forgotten/wrong inputs

#Verbosity
input_verbose = 1
background_verbose = 1
thermodynamics_verbose = 1
perturbations_verbose = 1
transfer_verbose = 1
primordial_verbose = 1
harmonic_verbose = 1
fourier_verbose = 1
lensing_verbose = 1
output_verbose = 1

```

## distances.py

```python
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import necessary modules
# uncomment to get plots displayed in notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from classy import Class


# In[ ]:


#Lambda CDM
LCDM = Class()
LCDM.set({'Omega_cdm':0.25,'Omega_b':0.05})
LCDM.compute()


# In[ ]:


#Einstein-de Sitter
CDM = Class()
CDM.set({'Omega_cdm':0.95,'Omega_b':0.05})
CDM.compute()

# Just to cross-check that Omega_Lambda is negligible 
# (but not exactly zero because we neglected radiation)
derived = CDM.get_current_derived_parameters(['Omega0_lambda'])
print (derived)
print ("Omega_Lambda =",derived['Omega0_lambda'])


# In[ ]:


#Get background quantities and recover their names:
baLCDM = LCDM.get_background()
baCDM = CDM.get_background()
baCDM.keys()


# In[ ]:


#Get H_0 in order to plot the distances in this unit
fLCDM = LCDM.Hubble(0)
fCDM = CDM.Hubble(0)


# In[ ]:


namelist = ['lum. dist.','comov. dist.','ang.diam.dist.']
colours = ['b','g','r']
for name in namelist:
    idx = namelist.index(name)
    plt.loglog(baLCDM['z'],fLCDM*baLCDM[name],colours[idx]+'-')
plt.legend(namelist,loc='upper left')
for name in namelist:
    idx = namelist.index(name)
    plt.loglog(baCDM['z'],fCDM*baCDM[name],colours[idx]+'--')
plt.xlim([0.07, 10])
plt.ylim([0.08, 20])

plt.xlabel(r"$z$")
plt.ylabel(r"$\mathrm{Distance}\times H_0$")
plt.tight_layout()
plt.savefig('distances.pdf')


```

## explanatory.ini

```
# *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*
# *  CLASS input parameter file  *
# *~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~*

# This example of input file, intended for CLASS beginners, lists all
# possibilities with detailed comments. You can use a more concise version, in
# which only the arguments in which you are interested would appear. Only lines
# containing an equal sign not preceded by a sharp sign "#" are considered by
# the code, any other line is considered as a comment.
#
# The normal syntax is:      parameter = value(s)
# where white spaces do not matter (they are removed automatically by the
# parser unless they are part of the parameter name).
# However,                   'parameter' = value(s)
# and                        "parameter" = value(s)
# are also accepted by the parser since v2.8.0
#
# Input files must have an extension ".ini".



# -------------------------
# ----> General parameters:
# -------------------------

# 1) List of output spectra requested:
#       - 'tCl' for temperature Cls,
#       - 'pCl' for polarization Cls,
#       - 'lCl' for CMB lensing potential Cls,
#       - 'nCl' (or 'dCl') for density number count Cls,
#       - 'sCl' for galaxy lensing potential Cls,
#       - 'mPk' for the matter power spectrum P(k) (then, depending on other options,
#               the code will return the linear and/or non-linear spectra,
#               for total matter and/or for clustering matter, and also possibly dewiggled),
#       - 'dTk' (or 'mTk') for density transfer functions for each species,
#       - 'vTk' for velocity transfer function for each species
#       - 'sd' for spectral distortions
#    Warning: both lCl and sCl compute the C_ls of the lensing potential,
#    C_l^phi-phi. If you are used to other codes, you may want to deal instead
#    with the deflection Cls or the shear/convergence Cls. The relations
#    between them are trivial:
#        --> deflection d: Cl^dd = l(l+1) C_l^phiphi
#        --> convergence kappa and shear gamma: the share the same harmonic
#            power spectrum: Cl^gamma-gamma = 1/4 * [(l+2)!/(l-2)!] C_l^phi-phi
#    By defaut, the code will try to compute the following cross-correlation
#    Cls (if available): temperature-polarisation, temperature-CMB lensing,
#    polarization-CMB lensing, CMB lensing-density, and density-lensing. Other
#    cross-correlations are not computed because they would slow down the
#    code considerably.
#
#    Can be left blank if you do not want to evolve cosmological perturbations
#    at all. (default: set to blank, no perturbation calculation)
output = tCl,pCl,lCl,mPk
#output = tCl,pCl,lCl
#output = mPk,mTk
#output = Sd

# 1.a) If you included 'tCl' in the list, you can take into account only some
#      of the terms contributing to the temperature spectrum:
#         - intrinsic temperature corrected by Sachs-Wolfe ('tsw' or 'TSW'),
#         - early integrated Sachs-Wolfe ('eisw' or 'EISW'),
#         - late integrated Sachs-Wolfe ('lisw' or 'LISW'),
#         - Doppler ('dop' or 'Dop'),
#         - polarisation contribution ('pol' or 'Pol').
#      Put below the list of terms to be included (defaut: if this field is not
#      passed, all terms will be included)
#temperature_contributions = tsw, eisw, lisw, dop, pol

# 1.a.1) If one of 'eisw' or 'lisw' is turned off, the code will read
#        'early/late isw redshift', the split value of redshift z at which the
#        isw is considered as late or early (if this field is absent or left
#        blank, by default, 'early/late isw redshift' is set to 50)
#early_late_isw_redshift =

# 1.b) If you included 'nCl' (or 'dCl') in the list, you can take into account
#      only some of the terms contributing to the obsevable number count
#      fluctuation spectrum:
#          - matter density ('density'),
#          - redshift-space and Doppler distortions ('rsd'),
#          - lensing ('lensing'),
#          - or gravitational potential terms ('gr').
#      Put below the list of terms to be included (defaut: if this field is not
#      passed, only 'dens' will be included)
#number_count_contributions = density, rsd, lensing, gr

# 1.c) If you included 'dTk' (or 'mTk') in the list, the code will give you by
#      default the transfer function of the scale-invariant Bardeen potentials
#      (for whatever gauge you are using). If you need the transfer function of
#      additional metric fluctuations, specific to the gauge you are using, set
#      the following flag to 'yes' (default: set to 'no')
#extra_metric_transfer_functions = yes


# 2) If you want to consider perturbed recombination, enter a word
#    containing the letter 'y' or 'Y'. CLASS will then compute the
#    perturbation in the ionization fraction x_e and the baryon
#    temperature, as in 0707.2727. The initial conformal time will be
#    small, therefore you should use the default integrator ndf15
#    (i.e. do not set 'evolver' to 0, otherwise the code will be
#    slower).  (default: no, neglect perturbed recombination)
#perturbed_recombination = yes

# 3) List of modes:
#         - 's' for scalars,
#         - 'v' for vectors,
#         - 't' for tensors).
#      More than one letter allowed, can be attached or separated by arbitrary
#      characters; letters can be small or capital. (default: set to 's')
modes = s
#modes = s,t

# 3.a) List of initial conditions for scalars:
#           - 'ad' for adiabatic,
#           - 'bi' for baryon isocurvature,
#           - 'cdi' for CDM isocurvature,
#           - 'nid' for neutrino density isocurvature,
#           - 'niv' for neutrino velocity isocurvature.
#        More than one of these allowed, can be attached or separated by arbitrary
#        characters; letters can be small or capital. (default: set to 'ad')
ic = ad
#ic = ad&bi&nid

# 3.b) Which perturbations should be included in tensor calculations?
#           - write 'exact' to include photons, ultra-relativistic species 'ur'
#             and all non-cold dark matter species 'ncdm';
#           - write 'massless' to approximate 'ncdm' as extra relativistic species
#             (good approximation if ncdm is still relativistic at the time of
#             recombination);
#           - write 'photons' to include only photons
#        (default: set to 'massless')
tensor_method =


# 4) Gauge
# 4.a) Gauge in which calculations are performed:
#         - 'sync' or 'synchronous' or 'Synchronous' for synchronous,
#         - 'new' or 'newtonian' or 'Newtonian' for Newtonian/longitudinal gauge
#      (default: set to synchronous)
gauge = synchronous

# 4.b) Do you want to output the N-body gauge quantities as well?
#      If you included 'dTk' or 'vTk' in the list of outputs, you may transform
#      your transfer functions into the Nbody gauge by setting the following
#      flag to 'yes'. This will also include the transfer function for the
#      metric perturbations H_T' (exact) and gamma (approximate) in the Nbody gauge.
#      See e.g. 1505.04756, and equations (A.2) and (A.5) in 1811.00904
#      for more precise definitions. These calculations are more stable with
#      'gauge=synchronous' (default). To compute H_T' and gamma
#      without converting the output to the Nbody gauge,
#      please use the flag 'extra metric transfer functions' instead.
#      Can be set to anything starting with 'y' or 'n'.
#      (default: set to 'no')
#nbody_gauge_transfer_functions = yes

# 4.c) Do you want the source functions for total non-relativistic matter, delta_m and theta_m, and baryon+cdm, delta_cb and theta_cb, to be outputed in the current gauge (the one selected in 4.a), instead of being automatically expressed as a gauge-invariant (GI) quantity, likeL: delta_m^GI=delta_m + 3 a H theta_m/k2, theta_m^GI=theta_m+alpha*k2 (default: no, that is, convert to GI)
#matter_source_in_current_gauge = no

# 4.d) Do you want the output table of perturbations (controlled by 'k_output_values') to be outputed in the current gauge, or to be always converted to the Newtonian gauge? (default: no, that is, convert to Newtonian)
#get_perturbations_in_current_gauge = no

# 5) Hubble parameter : either 'H0' in km/s/Mpc or 'h' (or 'theta_s_100'), where
#    the latter is the peak scale parameter defined exactly as 100(ds_dec/da_dec)
#    with a decoupling time given by maximum of visibility function (quite different
#    from theta_MC of CosmoMC and slightly different from theta_* of CAMB)
#    (default: 'h' set to 0.67810 such that 100*theta_s = 1.041783 like in Planck 2018)
h = 0.67810
#H0 = 67.810
#theta_s_100 = 1.041783


# 6) Primordial Helium fraction 'YHe', e.g. 0.25; if set to 'BBN' or 'bbn',
#    will be inferred from Big Bang Nucleosynthesis (default: set to 'BBN')
YHe = BBN


# 7) 'recombination' algorithm set to 'RECFAST' or 'HyRec'. 'HyRec' points at HyRec 2020. Its compute time is negligible compared to other CLASS modules. 'RECFAST' points at RecFastCLASS, an enhanced version of RecFast 1.5 with better integration shceme and less discontinuities. Recfast is still slightly faster than HyRec but less accurate. HyRec is better for most purposes. RecFast can still be useful for studying some particular modifications of standard recombination. Both schemes use the CLASS ODE integrators. (Default: HyRec')
recombination = HyRec

# 7.a) If recombination algorithm is set to 'RECFAST'
#      the photo-ionization coefficients beta(T) for normal Recfast depend on Tmat
#      This is an approximation (see e.g. arxiv:1605.03928 page 10, arxiv:1503.04827 page 2, right column)
#      With 'recfast_photoion_dependence' the photo-ionization coefficient beta(T) is set to depend on
#          - 'Tmat' uses beta(Tmat) depending on matter temperature
#                   (like in original RECFAST and in CLASS v2.x)
#          - 'Trad' uses beta(Trad) depending on radiation temperature
#                   (while this option is theoretically more motivated, the option 'Tmat' leads to
#                    results which agree better with HyRec and CosmoRec. This is probably due to the
#                    fudge factor for the Peebles coefficient being optimized for a Tmat dependence)
#      (default: set to 'Tmat')
recfast_photoion_dependence =


# 8) Parametrization of reionization: 'reio_parametrization' must be one of
#       - 'reio_none' (no reionization),
#       - 'reio_camb' (like CAMB: one tanh() step for hydrogen reionization one
#         for second helium reionization),
#       - 'reio_bins_tanh' (binned history x_e(z) with tanh()  interpolation
#         between input values),
#       - 'reio_half_tanh' (like 'reio_camb' excepted that we match the
#         function xe(z) from recombination with only half a tanh(z-z_reio)),
#       - 'reio_many_tanh' (arbitrary number of tanh-like steps with specified
#         ending values, a scheme usually more useful than 'reio_bins_tanh'),
#       - 'reio_inter' (linear interpolation between discrete values of xe(z)).
#    (default: set to 'reio_camb')
reio_parametrization = reio_camb

# 8.a) If 'reio_parametrization' set to 'reio_camb' or 'reio_half_tanh':
#      enter one of 'z_reio' or 'tau_reio' (default: 'z_reio' set to 7.6711 to
#      get tau_reio of 0.054308), plus 'reionization_exponent',
#      'reionization_width', 'helium_fullreio_redshift',
#      'helium_fullreio_width'. (default: set to 1.5, 0.5, 3.5, 0.5)
z_reio = 7.6711
#tau_reio = 0.05430842
reionization_exponent = 1.5
reionization_width = 0.5
helium_fullreio_redshift = 3.5
helium_fullreio_width = 0.5

# 8.b) If 'reio_parametrization' set to 'reio_bins_tanh':
#      enter number of bins and list of z_i and xe_i defining the free electron
#      density at the center of each bin. Also enter a dimensionless paramater
#      regulating the sharpness of the tanh() steps, independently of the bin
#      width; recommended sharpness is 0.3, smaller values will make steps too
#      sharp, larger values will make the step very progressive but with
#      discontinuity of x_e(z) derivative around z_i values. (default: set to
#      0, blank, blank, 0.3)
binned_reio_num = 3
binned_reio_z = 8,12,16
binned_reio_xe = 0.8,0.2,0.1
binned_reio_step_sharpness = 0.3

# 8.c) If 'reio_parametrization' set to 'reio_many_tanh':
#      enter number of jumps, list of jump redhsifts z_i (central value of each
#      tanh()), list of free electron density x_i after each jump, and common
#      width of all jumps. If you want to end up with all hydrogen reionized
#      but neglecting helium reionization, the first value of x_i in the list
#      should be 1. For each x_i you can also pass the flags -1 or -2. They
#      mean:
#         - -1: after hydrogen + first helium recombination (so the code will
#               substitute a value bigger than one based on Y_He);
#         - -2: after hydrogen + second helium recombination (the code will
#               substitute an even bigger value based on Y_He).
#      You can get results close to reio_camb by setting these parameters to
#      the value showed below (and adapting the second many_tanh_z to the usual
#      z_reio). (default: not set)
many_tanh_num = 2
many_tanh_z = 3.5,11.3
many_tanh_xe = -2,-1
many_tanh_width = 0.5

# 8.d) If 'reio_parametrization' set to 'reio_inter': enter the number of
#      points, the list of redshifts z_i, and the list of free electron
#      fraction values x_i. The code will do linear interpolation between them.
#      The first z_i should always be 0. Like above, for each x_i, you can also
#      pass the flags -1 or -2. They mean: for -1, after the hydrogen and the
#      first helium recombination (so the code will substitute a value bigger
#      than one based on Y_He); for -2, after the hydrogen and the second
#      helium recombination (the code will substitute an even bigger value
#      based on Y_He). The last value of x_i should always be zero, the code
#      will substitute it with the value that one would get in absence of
#      reionization, as computed by the recombination code. (default: not set)
reio_inter_num = 8
reio_inter_z =   0,  3,  4,   8,   9,  10,  11, 12
reio_inter_xe = -2, -2, -1,  -1, 0.9, 0.5, 0.1,  0


# 9) State whether you want the code to compute the simplest analytic
#    approximation to the photon damping scale (it will be added to the
#    thermodynamics output, and its value at recombination will be stored and
#    displayed in the standard output) (default: 'compute damping scale' set to
#    'no')
compute_damping_scale =


# 10) State whether you want to include a variation of fudamental constants. Can be set to 'none' or to 'instantaneous'. Smoother evolutions could be included by modifying the function "background_varconst_of_z" in source/background.c.
varying_fundamental_constants = none

# 10.a) If 'varying_fundamental_constants' is set to 'instantaneous', select the redshift of the transition 'varying_transition_redshift' (default: 50). At lower redshift, the value will be the currently observed value, while at higher redshift you can specify how large the value should be by giving the ratio of the value at high redshift with respect to the currently observed one. Provide the relative value of the fine structure constant 'varying_alpha' (default: 1), and the relative value of the effective electron mass 'varying_me' (default: 1). The treatment corresponds to that of 1705.03925.
varying_transition_redshift =
varying_alpha = 1.
varying_me = 1.

# 10.b) If 'varying_fundamental_constants' is not set to 'none' and 'YHe' is set to 'BBN', specify by how much the 'YHe' prediction from 'BBN' should be affected by the different value of the fine structure constant. The default value is motivated by 2001.01787. (default: 1)
bbn_alpha_sensitivity = 1.


# -------------------------
# ----> Species parameters:
# -------------------------

# 1) Photon density: either 'T_cmb' in K or 'Omega_g' or 'omega_g' (default:
#    'T_cmb' set to 2.7255)
T_cmb = 2.7255
#Omega_g =
#omega_g =


# 2) Baryon density: either 'Omega_b' or 'omega_b' (default: 'omega_b' set to
#    0.02238280)
omega_b = 0.02238280
#Omega_b =


# 3) Ultra-relativistic species / massless neutrino density: either
# 'N_ur' or 'Omega_ur' or 'omega_ur' (default: 'N_ur' set to 3.044;
# see 2008.01074 and 2012.02726. This value is more accurate than the
# previous default value of 3.046) (note: instead of 'N_ur' you can
# pass equivalently 'N_eff', although this syntax is deprecated) (one
# more remark: if you have respectively 1,2,3 massive neutrinos, if
# you stick to the default value T_ncdm equal to 0.71611, designed to
# give m/omega of 93.14 eV, and if you want to use N_ur to get N_eff
# equal to 3.044 in the early universe, then you should pass here
# respectively 2.0308,1.0176,0.00441)
N_ur = 3.044
#Omega_ur =
#omega_ur =

# 3.a) To simulate ultra-relativistic species with non-standard properties, you
#      can pass 'ceff2_ur' and 'cvis2_ur' (effective squared
#      sound speed and viscosity parameter, like in the Generalised Dark Matter
#      formalism of W. Hu) (default: both set to 1/3)
#ceff2_ur =
#cvis2_ur =


# 4) Density of cdm (cold dark matter): 'Omega_cdm' or 'omega_cdm', or,
#    density of total non-relativistic matter: 'Omega_m' or 'omega_m'.
#    If you pass 'Omega_m' or 'omega_m', the code will automatically fill
#    up the density of Cold Dark Matter such that all non-relativistic species
#    (including non-cold DM) sum up to your input value of Omega_m
#    (default: 'omega_cdm' set to 0.1201075)
omega_cdm = 0.1201075
#Omega_cdm =
#Omega_m =
#omega_m =


# 5) ncdm sector (i.e. any non-cold dark matter relics, including massive
#    neutrinos, warm dark matter, etc.):
# 5.a) 'N_ncdm' is the number of distinct species (default: set to 0)
N_ncdm =

# 5.b) 'use_ncdm_psd_files' is the list of N_ncdm numbers:
#           - 0 means 'phase-space distribution (psd) passed analytically
#             inside the code, in the mnodule background.c, inside the function
#             background_ncdm_distribution()',
#           - 1 means 'psd passed as a file with at list two columns: first for
#             q, second for f_0(q)', where q is p/T_ncdm
#      (default: only zeros)
#use_ncdm_psd_files = 0

# 5.b.1) If some of the previous values are equal to one, 'ncdm_psd_filenames' is
#        the list of names of psd files (as many as number of ones in previous
#        entry)
ncdm_psd_filenames = psd_FD_single.dat

# 5.c) 'ncdm_psd_parameters' is an optional list of double parameters to
#      describe the analytic distribution function or to modify a p.s.d. passed
#      as a file. It is made available in the routine
#      background_ncdm_distribution.
#ncdm_psd_parameters = 0.3 ,0.5, 0.05
#ncdm_psd_parameters = Nactive, sin^2_12 ,s23 ,s13

# 5.d) 'Omega_ncdm' or 'omega_ncdm' or 'm_ncdm' in eV (default: all set to
#      zero); with only one of these inputs, CLASS computes the correct value
#      of the mass; if both (Omega_ncdm, m_ncdm) or (omega_ncdm, m_ncdm) are
#      passed, CLASS will renormalise the psd in order to fulfill both
#      conditions. Passing zero in the list of m_ncdm's or Omeg_ncdm's means
#      that for this component, this coefficient is not imposed, and its value
#      is inferred from the other one.
m_ncdm = 0.06
#m_ncdm = 0.04, 0.04, 0.04
#Omega_ncdm =
#omega_ncdm =

# 5.e) 'T_ncdm' is the ncdm temperature in units of photon temperature
#      (default: set to 0.71611, which is slightly larger than the
#      instantaneous decoupling value (4/11)^(1/3); indeed, this default value
#      is fudged to give a ratio m/omega equal to 93.14 eV for active
#      neutrinos, as predicted by precise studies of active neutrino
#      decoupling, see hep-ph/0506164)
T_ncdm =

# 5.f) 'ksi_ncdm' is the ncdm chemical potential in units of its own
#      temperature (default: set to 0)
ksi_ncdm =

# 5.g) 'deg_ncdm' is the degeneracy parameter multiplying the psd: 1 stands for
#      'one family', i.e. one particle + anti-particle (default: set to 1.0)
deg_ncdm =

# 5.h) 'ncdm_quadrature_strategy' is the method used for the momentum sampling of
#      the ncdm distribution function.
#         - 0 is the automatic method,
#         - 1 is Gauss-Laguerre quadrature,
#         - 2 is the trapezoidal rule on [0,Infinity] using the transformation q->1/t-1.
#         - 3 is the trapezoidal rule on [0,q_max] where q_max is the next input.
#      (default: set to 0)
ncdm_quadrature_strategy =

# 5.h.1) 'ncdm_maximum_q' is the maximum q relevant only for Quadrature strategy 3.
#        (default: set to 15)
ncdm_maximum_q =

# 5.h.2) Number of momentum bins. (default: 150)
ncdm_N_momentum_bins =


# 6) Curvature: 'Omega_k' (default: 'Omega_k' set to 0)
Omega_k = 0.


# Begin of ADDITIONAL SPECIES --> Add your species here

# 7.1) Decaying CDM into Dark Radiation
# 7.1.a) The current fractional density of dcdm+dr (decaying cold dark matter
#      and its relativistic decay radiation): 'Omega_dcdmdr' or 'omega_dcdmdr'
#      (default: 'Omega_dcdmdr' set to 0)
Omega_dcdmdr = 0.0
#omega_dcdmdr = 0.0

# 7.1.b) The rescaled initial value for dcdm density (see 1407.2418 for
#      definitions). If you specify 7.a.1, 7.a.2 will be found automtically by a
#      shooting method, and vice versa. (default: 'Omega_dcdmdr' set to 0,
#      hence so is 'Omega_ini_dcdm')
#Omega_ini_dcdm =
#omega_ini_dcdm =

# 7.1.c) Decay constant of dcdm in km/s/Mpc, same unit as H0 above.
Gamma_dcdm = 0.0
tau_dcdm = 0.0


# 7.2) Multi-interacting Dark Matter (idm), implemented by N. Becker,
#    D.C. Hooper, and N. Schoeneberg. Described in (2010.04074)

# 7.2.1) Global parameters for the (multi-)interacting Dark Matter component

# 7.2.1.a) Amount of interacting Dark Matter
# Can be passed as either f_idm (fraction) or Omega_idm (relative abundance) (default : 0)
#Omega_idm = 0.
f_idm = 0.

# 7.2.1.b) Mass of interacting Dark Matter particle in eV (default : 1e9)
m_idm = 1e9

# 7.2.2) Dark Matter interacting with Dark Radiation (idm_dr) and
#     interacting Dark Radiation (idr), implemented by
#     M. Archidiacono, S. Bohr, and D.C. Hooper, following the ETHOS
#     framework (1512.05344).  Can also take as input the parameters
#     of the models of 1507.04351 (with non-abelian dark matter, dark
#     gluons...) which can be seen as a sub-class of ETHOS. See
#     1907.01496 for more details on both cases.

# 7.2.2.a) Amount of interacting dark radiation (idr)
#        - Can be parameterised through the temperature ratio 'xi_idr' (= T_idr/T_cmb)
#        - Can be parameterised through the number of extra relativistic relics 'N_idr' (or indifferently 'N_dg')
#    In all cases the parameter is dimensionless.
# (default : 0)
xi_idr =
#N_idr =

# 7.2.2.b) Statistical factor to differentiate between fermionic (= 7/8) and bosonic (= 1) dark radiation (default 7/8)
stat_f_idr = 0.875

# 7.2.2.c) Strength of the coupling between DM and DR:
#
#     Can be passed as 'a_idm_dr' or 'a_dark' in ETHOS parameterisation, in units of 1/Mpc.
#      Then in ETHOS notations: Gamma_DR-DM = - omega_DM a_dark ((1+z)/10^7)^nindex_dark
#                        while: Gamma_DM-DR = - 4/3 (rho_DR/rho_DM) omega_DM  a_dark ((1+z)/10^7)^nindex_dark
#                                           = - 4/3 omega_DR a_dark (1+z) ((1+z)/10^7)^nindex_dark
#        or in CLASS notations: dmu_idm_dr  = - Gamma_DR-DM = omega_idm_dr a_idm_dr ((1+z)/10^7)^nindex_idm_dr
#
#     Can be passed as 'Gamma_0_nadm' in NADM parameterisation, in units of 1/Mpc.
#      Then in ETHOS notations: Gamma_DR-DM = - 3/4 Omega_DM/Omega_DR Gamma_0_nadm
#                        while: Gamma_DM-DR = - (1+z) Gamma_0_nadm
#      or in CLASS notations:   dmu_idm_dr  = - Gamma_DR-DM = 3/4 Omega_idm_dr/Omega_idr Gamma_0_nadm
#
#    (default : 0)
a_idm_dr = 0.
#Gamma_0_nadm =

# 7.2.2.d) Only if ETHOS parametrization : Power of the temperature dependence of the co-moving idr - idm_dr interaction rate
#    Can be passed indifferently as 'nindex_idm_dr' (or 'nindex_dark'), in units of 1/Mpc.
#    (default : 4, unless Gamma_0_nadm has been passed, then default changes to 0)
nindex_idm_dr =

# 7.2.2.e) Only if ETHOS parametrization : Nature of the interacting dark radiation: 'free_streaming' or 'fluid'
#    (default = 'free_streaming', unless Gamma_0_nadm has been passed, then default changes to 'fluid')
idr_nature =

# 7.2.2.f) Strength of the dark radiation self interactions coupling,
#    can be passed as 'b_idr' or 'b_dark', in units of 1/Mpc.
#    In ETHOS notations: Gamma_DR-DR = (b_dark/a_dark) (Omega_DR/Omega_DM) Gamma_DR-DM
#    In CLASS notations: dmu_idr = - Gamma_DR-DR = (b_idr/a_idm_dr) (Omega_idr/Omega_idm_dr) dmu_idm_dr
# (default : 0)
b_idr =

# 7.2.2.g) idr - idm_dr interaction angular coefficient: 'alpha_idm_dr' (or indifferently 'alpha_dark')
#    Should be 3/4 if vector boson mediator; 3/2 if scalar boson mediator.
#    In full generality this coefficient may depend on l = 2, 3, 4...
#    The user can pass here a list of values with an arbitrary size. The first coefficients will be adjusted
#    accordingly. After that, the last value will be repeated.
#    For instance, if users passes 3, 2, 1, the code will take alpha_2=3, alpha_3=2, and all others equal to 1.
#    (default = all set to 1.5)
alpha_idm_dr = 1.5

# 7.2.2.h) idr self-interaction angular coefficient: 'beta_idr' (or indifferently 'beta_dark')
#    In full generality this coefficient may depend on l = 2, 3, 4...
#    The user can pass here a list of values with an arbitrary size. The first coefficients will be adjusted
#    accordingly. After that, the last value will be repeated.
#    For instance, if users passes 3, 2, 1, the code will take beta_2=3, beta_3=2, and all others equal to 1.
#    (default = all set to 1.5)
beta_idr = 1.5

# -> Precision parameters for idm_dr and idr can be found in precisions.h, with the tag idm_dr

# 7.2.3) Interacting Dark Matter with Baryons
# Implemented by D.C. Hooper, N. Schoeneberg, and N. Becker
# following 1311.2937, 1509.00029, 1803.09734, 1802.06788

# 7.2.3.a) Coupling strength of Dark Matter and baryons (in cm^2) (default : 0)
cross_idm_b = 0.
# 7.2.3.b) Temperature dependence of the DM - baryon interactions (between -4 and 4) (default : 0)
n_index_idm_b = 0

# 7.2.4) Dark Matter interactions with photons
# Implementd by N. Becker following the formalism of Stadler&Boehm (1802.06589)

# 7.2.4.a) Interaction coefficient or coupling strength between DM and photons
#  Can be passed as either u_idm_g (dimensionless interaction strength) or cross_idm_g (cross section in cm^2) (default : 0)
u_idm_g = 0
#cross_idm_g = 0

# 7.2.4.b) Temperature dependence of the DM - photon interactions (default : 0)
n_index_idm_g = 0

# End of ADDITIONAL SPECIES


# 8) Dark energy contributions.
#    At least one out of three conditions must be satisfied:
#          - 'Omega_Lambda' unspecified.
#          - 'Omega_fld' unspecified.
#          - 'Omega_scf' set to a negative value. [Will be refered to as
#             unspecified in the following text.]
#      The code will then use the first unspecified component to satisfy the
#      closure equation (sum_i Omega_i) equals (1 + Omega_k)
#      (default: 'Omega_fld' and 'Omega_scf' set to 0 and 'Omega_Lambda'
#      inferred by code)
Omega_fld = 0
Omega_scf = 0
# Omega_Lambda = 0.7

# 8.a) If Omega fluid is different from 0

# 8.a.1) The flag 'use_ppf' is 'yes' by default, to use the PPF approximation
#        (see 0808.3125 [astro-ph]) allowing perturbations to cross the
#        phantom divide. Set to 'no' to enforce true fluid equations for
#        perturbations. When the PPF approximation is used, you can choose the
#        ratio 'c_gamma_over_c_fld' (eq. (16) in 0808.3125). The default is 0.4
#        as recommended by that reference, and implicitely assumed in other
#        codes. (default: 'use_ppf' to yes, 'c_gamma_over_c_fld' to 0.4)
use_ppf = yes
c_gamma_over_c_fld = 0.4

# 8.a.2) Choose your equation of state between different models,
#         - 'CLP' for p/rho = w0_fld + wa_fld (1-a/a0)
#           (Chevalier-Linder-Polarski),
#         - 'EDE' for early Dark Energy
#      (default:'fluid_equation_of_state' set to 'CLP')
fluid_equation_of_state = CLP

# 8.a.2.1) Equation of state of the fluid in 'CLP' case and squared sound speed
#          'cs2_fld' of the fluid (this is the sound speed defined in the frame
#          comoving with the fluid, i.e. obeying to the most sensible physical
#          definition). Generalizing w(a) to a more complicated expressions would
#          be easy, for that, have a look into source/background.c at the
#          function background_w_fld(). (default: 'w0_fld' set to -1, 'wa_fld' to
#          0, 'cs2_fls' to 1)
#w0_fld = -0.9
#wa_fld = 0.
#cs2_fld = 1

# 8.a.2.2) Equation of state of the fluid in 'EDE' case and squared sound speed
#          'cs2_fld' of the fluid (this is the sound speed defined in the frame
#          comoving with the fluid, i.e. obeying to the most sensible physical
#          definition). Generalizing w(a) to a more complicated expressions would
#          be easy, for that, have a look into source/background.c at the
#          function background_w_fld(). (default: 'w0_fld' set to -1, 'Omega_EDE'
#          to 0, 'cs2_fls' to 1)
#w0_fld = -0.9
#Omega_EDE = 0.
#cs2_fld = 1

# 8.b) If Omega scalar field is different from 0

# 8.b.1) Scalar field (scf) potential parameters and initial conditions
#        (scf_parameters = [scf_lambda, scf_alpha, scf_A, scf_B, phi,
#        phi_prime]). V = ((\phi-B)^\alpha + A)exp(-lambda*phi), see
#        http://arxiv.org/abs/astro-ph/9908085. If 'attractor_ic_scf' is set to
#        'no', the last two entries are assumed to be the initial values of phi
#        in units of the reduced planck mass m_Pl and the conformal time
#        derivative of phi in units of [m_Pl/Mpc]. (Note however that CLASS
#        determines the initial scale factor dynamically and the results might
#        not be as expected in some models.)
scf_parameters = 10.0, 0.0, 0.0, 0.0, 100.0, 0.0

# 8.b.2) Scalar field (scf) initial conditions from attractor solution (assuming
#        pure exponential potential). (default: yes)
attractor_ic_scf = yes

# 8.b.3) Scalar field (scf) shooting parameter: If Omega_scf is set (can only be negative),
#        the following index (0,1,2,...) in the list scf_parameters will be used for shooting:
#        (See also the section about shooting in input.c)
#        Basically parameter number scf_tuning_index will be adjusted until
#        the correct Omega_scf is found to suffice the budget equation
scf_tuning_index = 0


# 8.b.4) Scalar field (scf) shooting parameter. With this, you can overwrite some parameter
#        of 8.b.1) depending on the index defined in 8.b.3)
scf_shooting_parameter =

# -----------------------------------------
# ----> Exotic energy injection parameters:
# -----------------------------------------

# 1) DM Annihilation

# 1.a) In order to model energy injection from DM annihilation, specify a
#      parameter 'annihilation_efficiency' corresponding to
#      <sigma*v> / m_cdm expressed here in units of m^3/s/J. Alternatively,
#      you can specify the annihilation cross section in cm^3/s and the DM
#      mass in GeV. The code will then evaluate 'annihilation_efficiency'.
#     (default: all set to zero)
DM_annihilation_efficiency = 0.
#DM_annihilation_cross_section = 0.
#DM_annihilation_mass = 0.
#DM_annihilation_fraction = 0.

# 1.a.1) You can model simple variations of the above quantity as a function of
#        redhsift. If 'annihilation_variation' is non-zero, the function F(z)
#        defined as (<sigma*v> / m_cdm)(z) (see 1.a) will be a parabola in log-log scale
#        between 'annihilation_zmin' and 'annihilation_zmax', with a curvature
#        given by 'annihilation_variation' (must be negative), and with a maximum
#        in 'annihilation_zmax'; it will be constant outside this range. To
#        take DM halos into account, specify the parameters 'annihilation_f_halo',
#        the amplitude of the halo contribution, and 'annihilation_z_halo',
#        the characteristic redshift of halos (default: no variation,
#        'annihilation_variation' and 'annihilation_f_halo' set to zero).
DM_annihilation_variation = 0.
DM_annihilation_z = 1000
DM_annihilation_zmax = 2500
DM_annihilation_zmin = 30
DM_annihilation_f_halo= 0
DM_annihilation_z_halo= 8


# 2) DM electromagnetic decay

# 2.a) Specify the dimensionless parameter 'decay_fraction' which is
#      equal to the fraction of cdm with electromagnetic decay
#      products (decaying into dark radiation is handled by
#      Omega_dcdm).  Note: Until class 2.7, this parameter was called
#      'decay'. Its name and its meaning have slightly changed to
#      avoid confusion when working with model in which the lifetime
#      of the dcdm can be small (this is allowed providing that the
#      'decay_fraction' parameter is small as well).  (default: set to
#      0)
DM_decay_fraction = 0.

# 2.b) Specify the decay width of the particle 'decay_Gamma' in 1/s.
#      (default: set to 0)
DM_decay_Gamma = 0.


# 3) PBH evaporation. In this case, check that in 5), 'f_eff_type' and
# 'f_eff' have their default values 'on_the_spot' and 1, because CLASS
# will automatically take into account the time-dependent efficiency
# of energy injection from evaporating BH, taking the spectrum of
# evaporated particles into account.

# 3.a) Specify a dimensionless parameter 'PBH_evaporation_fraction' which is equal
#      to the fraction of evaporating PBH. (default set to 0)
PBH_evaporation_fraction = 0.

# 3.b) Specify the mass of the evaporating PBH in g. (default set to 0)
PBH_evaporation_mass = 0.


# 4) PBH matter accretion

# 4.a) Specify a dimensionless parameter 'PBH_accreting_fraction' which is equal
#      to the fraction of accreting PBH. (default set to 0)
PBH_accretion_fraction = 0.

# 4.b) Specify the mass of the accreting PBH in Msun. (default set to 0)
PBH_accretion_mass = 0.

# 4.c) Specify the 'PBH_accretion_recipe' between 'spherical_accretion'
# (computed according to Ali-Haimoud and Kamionkowski 1612.05644), or
# 'disk_accretion' (computed according to Poulin et al. 1707.04206).
# (default set to 'disk_accretion')
PBH_accretion_recipe = disk_accretion

# 4.c.1) If you choose 'spherical_accretion', you might want to specify the
#        relative velocity between PBH and baryons in km/s.
#        If negative, the linear result is chosen by the code.
#        (default set to -1., standard value is the linear result extrapolated to PBH.)
PBH_accretion_relative_velocities = -1.

# 4.c.2) If you choose 'disk_accretion', you might want to specify the
#        factor 'PBH_accretion_ADAF_delta' which, determines the heating
#        of the electrons in the disk, influencing the emissivity.
#        Can be set to 0.5 (aggressive scenario), 0.1 or 1.e-3 (conservative).
#        (default set to 1.e-3)
#        Furthermore you can also specify the the eigenvalue of the accretion
#        rate. It rescales the perfect Bondi case. (see e.g. Ali-Haimoud
#        & Kamionkowski 2016) (default set to 0.1, standard value in the ADAF
#        scenario.)
PBH_accretion_ADAF_delta = 1.e-3
PBH_accretion_eigenvalue = 0.1

# 5) Define the so-called injection efficiency f_eff, i.e. the factor
#    determining how much of the heating is deposited at all,
#    regardless of the form. There are two options to define this
#    function: 'on_the_spot' or 'from_file' (default: set to 'on_the_spot').
#
#    - with 'on_the_spot', the injected energy is transformed into deposited energy
#      at the same redshift with efficiency f_eff. In this case the
#      user can pass explicitely the value of f_eff. (default: f_eff=1)
#
#    - for 'from_file' the code reads a precomputed function in an external file
#      with a path set by the user (default set to "external/heating/example_f_eff_file.dat")
f_eff_type = on_the_spot
#f_eff =
#f_eff_file = external/heating/example_f_eff_file.dat

# 6) Define the so-called deposition function chi, i.e the function which determines
#    the amount of energy effectively deposited into the different forms (heating,
#    ionization, Lyman alpha and low energy). There are several options
#        - by setting 'chi_type' to 'CK_2004', the approximation by Chen & Kamionkowski 2004 is employed.
#        - by setting 'chi_type' to 'PF_2005', the approximation by Padmanabhan & Finkbeiner 2005 is employed.
#        - by setting 'chi_type' to 'Galli_2013_file', the approximation by Galli et al. 2013 is employed.
#        - by setting 'chi_type' to 'Galli_2013_analytic', the approximation by Poulin is employed.
#          interpolating Galli et al. 2013 anyltically. Use this for consistency tests with
#          older versions of CLASS (2.x).
#        - by setting 'chi_type' to 'heat', the whole injected energy is going
#          to be deposited into heat.
#        - by setting 'chi_type' to 'from_x_file' or 'from_z_file', the user can
#          define own deposition functions with respect to the free electron
#          fraction x_e or to redshift, respectively.
#    (default set to 'CK_2004')
chi_type = CK_2004

# 6.a) If the option 'from_file' has been chosen, define the name of the file.
#      Two files 'example_chix_file.dat' and 'example_chiz_file.dat' are given
#      as example in external/heating. Note that 'example_chix_file.dat' has
#      been computed according to the approximations of Galli et al. 2013.
#      (default set to "external/heating/example_f_eff_file.dat")
#chi_file = external/heating/example_chiz_file.dat
#chi_file = external/heating/example_chix_file.dat



# -------------------------------
# ----> Non-linear parameters:
# -------------------------------

# 1) If you want an estimate of the non-linear P(k) and Cls:
#    Enter 'halofit' or 'Halofit' or 'HALOFIT' for Halofit
#    Enter 'hmcode' or 'Hmcode' or 'HMcode' or 'HMCODE' for HMcode;
#    otherwise leave blank (default: blank, linear P(k) and Cls)
non_linear =

# 1.a) if you chose Halofit:

# 1.a.1) if you have Omega0_fld != 0. (i.e. you
# set Omega_lambda=0.) & wa_fld != 0.,  then you might want to use the
# pk equal method of 0810.0190 and 1601.07230 by setting this flag to
# 'yes' (default: set to 'no')
pk_eq =

# 1.a.2) minimum value of k_max in 1/Mpc used internally inside
# Halofit to compute a few integrals. Should never be too small,
# otherwise these integrals would not converge. (default: 5 1/Mpc)
halofit_min_k_max =

# 1.b) if you chose HMcode:

# 1.b.1) choose the version among:
#    - '2015' or '2016' (just '15' or '16' also works) for HMcode 2016 by Mead et al. (arXiv 1602.02154)
#    - '2020' (just '20' also works) for HMcode 2020 by Mead et al. (arXiv 2009.01858)
#    - '2020_baryonic_feedback' for HMcode 2020 by Mead et al. (arXiv 2009.01858) with baryonic feedback
#    (defualt: 2020)
hmcode_version =

# 1.b.2) if you choose '2015' or '2016': baryonic feedback parameters 'eta_0' and 'c_min'
#    In HMcode 2016 you can specify a baryonic feedback model (otherwise only DM is used).
#    Each model depends on two parameters: the minimum concentration "c_min" from the
#    Bullock et al. 2001 mass-concentration relation and the halo bloating parameter "eta_0"
#    introduced in Mead et al. 2015. In Mead et al. 2015 the parameters c_min and eta_0 are fitted
#    to the Cosmic Emulator dark matter only simulation (Heitman et al. 2014) and the
#    hydrodynamical OWLS simulations (Schaye et al. 2010, van Daalen et al. 2011).
#    You can choose between the 5 models of Mead et al. 2015, Table 4:
#      Model           (eta_0, c_min) Explanation
#    - 'emu_dmonly'    (0.603, 3.13)  fits the only dark matter Cosmic Emulator simulation (default)
#    - 'owls_dmonly'   (0.64, 3.43)   fits the OWLS simulation of dark matter only
#    - 'owls_ref'      (0.68, 3.91)   fits the OWLS simulation that includes gas cooling, heating,
#                                     star formation and evolution, chemical enrichment and supernovae feedback
#    - 'owls_agn'      (0.76, 2.32)   fits the OWLS simulation that includes AGN feedback
#    - 'owls_dblim'    (0.70, 3.01)   fits the OWLS simulation that has extra supernova energy in wind velocity
#    Set 'feedback model' to one of these names,
#    or leave blank and pass manually the value of either 'eta_0' or 'c_min'
#    (the other one will then be fixed according to equation (30) in Mead et al. 2015),
#    or pass manually the two values of 'eta_0' and 'c_min'
#    (default: 'feedback model' set to 'emu_dmonly')
feedback model =
eta_0 =
c_min =

# 1.b.3) if you choose '2020_baryonic_feedback', single feedback model
# parameter 'log10T_heat_hmcode' (default: 7.8)
log10T_heat_hmcode =

# 1.b.4) minimum value of k_max in 1/Mpc used internally inside
# HMcode to compute a few integrals. Should never be too small,
# otherwise these integrals would not converge. (default: 5 1/Mpc)
hmcode_min_k_max =

# 1.b.5) if you chose HMcode, set the redshift value at which the Dark
# Energy correction is evaluated - this needs to be at early times,
# when dark Energy is totally subdominant (default: 10)
z_infinity =

# 1.b.6) if you chose HMcode, number of k points for the de-wiggling (default: 512)
nk_wiggle =

# 2) Control on the output of the nowiggle spectrum (assuming you required 'mPk')

# 2.a) do you want to enforce the calculation and output of an analytic
#      approximation to the nowiggle linear power spectrum, even when this is not
#      required by the chosen non-linear method? (default: no)
analytic_nowiggle =

# 2.b) do you want to enforce the calculation and output of the
#      nowiggle linear power spectrum, obtained by fuiltering/smoothing the
#      full spectrum, even when this is not required by the chosen
#      non-linear method? (default: no)
numerical_nowiggle =

# ----------------------------
# ----> Primordial parameters:
# ----------------------------

# 1) Primordial spectrum type
#       - 'analytic_Pk' for an analytic smooth function with amplitude, tilt,
#         running, etc.; analytic spectra with feature can also be added as
#         a new type;
#       - 'inflation_V' for a numerical computation of the inflationary
#         primordial spectrum, through a full integration of the perturbation
#         equations, given a parametrization of the potential V(phi) in the
#         observable window, like in astro-ph/0703625;
#       - 'inflation_H' for the same, but given a parametrization of the
#         potential H(phi) in the observable window, like in
#         astro-ph/0710.1630;
#       - 'inflation_V_end' for the same, but given a parametrization of the
#         potential V(phi) in the whole region between the observable part and
#         the end of inflation;
#       - 'two scales' allows to specify two amplitudes instead of one
#         amplitude and one tilt, like in the isocurvature mode analysis of the
#         Planck inflation paper (works also for adiabatic mode only; see
#         details below, item 2.c);
#       - 'external_Pk' allows for the primordial spectrum to be computed
#         externally by some piece of code, or to be read from a table, see
#         2.d).
#    (default: set to 'analytic_Pk')
Pk_ini_type = analytic_Pk

# 1.a) Pivot scale in Mpc-1 (default: set to 0.05)
k_pivot = 0.05

# 1.b) For type 'analytic_Pk':
# 1.b.1) For scalar perturbations
#        curvature power spectrum value at pivot scale ('A_s' or
#        'ln_A_s_1e10') OR one of 'sigma8' or 'S8' (found by iterations using a shooting
#        method). (default: set 'A_s' to 2.100549e-09)
A_s = 2.100549e-09
#ln_A_s_1e10 = 3.04478383
#sigma8 = 0.824398
#S8 = 0.837868

# 1.b.1.1) Adiabatic perturbations:
#          tilt at the same scale 'n_s', and tilt running 'alpha_s'
#          (default: set 'n_s' to 0.9660499, 'alpha_s' to 0)
n_s = 0.9660499
alpha_s = 0.

# 1.b.1.2) Isocurvature/entropy perturbations:
#          for each mode xx ('xx' being one of 'bi', 'cdi', 'nid', 'niv',
#          corresponding to baryon, cdm, neutrino density and neutrino velocity
#          entropy perturbations), enter the entropy-to-curvature ratio f_xx,
#          tilt n_xx and running alpha_xx, all defined at the pivot scale; e.g.
#          f_cdi of 0.5 means S_cdi/R equal to one half and (S_cdi/R)^2 to 0.25
#          (default: set each 'f_xx' to 1, 'n_xx' to 1, 'alpha_xx' to 0)
f_bi = 1.
n_bi = 1.5
f_cdi=1.
f_nid=1.
n_nid=2.
alpha_nid= 0.01
# etc.

# 1.b.1.3) Cross-correlation between different adiabatic/entropy mode:
#          for each pair (xx, yy) where 'xx' and 'yy' are one of 'ad', 'bi',
#          'cdi', 'nid', 'niv', enter the correlation c_xx_yy (parameter between
#          -1 and 1, standing for cosDelta, the cosine of the cross-correlation
#          angle), the tilt n_xx_yy of the function cosDelta(k), and its running
#          alpha_xx_yy, all defined at the pivot scale. So, for a pair of fully
#          correlated (resp. anti-correlated) modes, one should set (c_xx_yy,
#          n_xx_yy, alpha_xx_yy) to (1,0,0) (resp. (-1,0,0) (default: set each
#          'c_xx_yy' to 0, 'n_xx_yy' to 0, 'alpha_xx_yy' to 0)
c_ad_bi = 0.5
#n_ad_bi = 0.1
c_ad_cdi = -1.
c_bi_nid = 1.
#n_bi_nid = -0.2
#alpha_bi_nid = 0.002
# etc.

# 1.b.2) For tensor perturbations (if any):
#        tensor-to-scalar power spectrum ratio, tilt,
#        running at the pivot scale; if 'n_t' and/or 'alpha_t' is set to 'scc'
#        or 'SCC' isntead of a numerical value, it will be inferred from the
#        self-consistency condition of single field slow-roll inflation: for
#        n_t, -r/8*(2-r/8-n_s); for alpha_t, r/8(r/8+n_s-1) (default: set 'r'
#        to 1, 'n_t' to 'scc', 'alpha_t' to 'scc')
r = 1.
n_t = scc
alpha_t = scc

# 1.c) For type 'inflation_V'
# 1.c.1) Type of potential: 'polynomial' for a Taylor expansion of the
#        potential around phi_pivot. Other shapes can easily be defined in
#        primordial module.
potential = polynomial

# 1.c.2) For 'inflation_V' and 'polynomial': enter either the coefficients
#        'V_0', 'V_1', 'V_2', 'V_3', 'V_4' of the Taylor expansion (in units of
#        Planck mass to appropriate power), or their ratios 'R_0', 'R_1',
#        'R_2', 'R_3', 'R_4' corresponding to (128pi/3)*V_0^3/V_1^2,
#        V_1^2/V_0^2, V_2/V_0, V_1*V_3/V_0, V_1^2*V_4/V_0^3, or the
#        potential-slow-roll parameters 'PSR_0', 'PSR_1', 'PSR_2', 'PSR_3',
#        'PSR_4', equal respectively to R_0, epsilon_V=R_1/(16pi),
#        eta_V=R_2/(8pi), ksi_V=R_3/(8pi)^2, omega_V=R_4/(8pi)^3 (default:
#        'V_0' set to 1.25e-13, 'V_1' to 1.12e-14, 'V_2' to 6.95e-14, 'V_3'
#        and 'V_4' to zero).
V_0=1.e-13
V_1=-1.e-14
V_2=7.e-14
V_3=
V_4=
#R_0=2.18e-9
#R_1=0.1
#R_2=0.01
#R_3=
#R_4=
#PSR_0 = 2.18e-9
#PSR_1 = 0.001989
#PSR_2 = 0.0003979
#PSR_3 =
#PSR_4 =

# 1.d) For 'inflation_H':
#      enter either the coefficients 'H_0', 'H_1', 'H_2', 'H_3', 'H_4' of the
#      Taylor expansion (in units of Planck mass to appropriate power), or the
#      Hubble-slow-roll parameters 'HSR_0', 'HSR_1', 'HSR_2', 'HSR_3', 'HSR_4'
H_0=1.e-13
H_1=-1.e-14
H_2=7.e-14
H_3=
H_4=
#HSR_0 = 2.18e-9
#HSR_1 = 0.001989
#HSR_2 = 0.0003979
#HSR_3 =
#HSR_4 =

# 1.e) For type 'inflation_V_end':
# 1.e.1) Value of the field at the minimum of the potential after inflation, or
#        at a value in which you want to impose the end of inflation, in
#        hybrid-like models. By convention, the code expects inflation to take
#        place for values smaller than this value, with phi increasing with
#        time (using a reflection symmetry, it is always possible to be in that
#        case) (default: 'phi_end' set to 0)
phi_end =

# 1.e.2) Shape of the potential. Refers to functions pre-coded in the primordail
#        module, so far 'polynomial' and 'higgs_inflation'. (default:
#        'full_potential' set to 0)
full_potential = polynomial

# 1.e.3) Parameters of the potential. The meaning of each parameter is
#        explained in the function primrodial_inflation_potential() in
#        source/primordial.c
Vparam0 =
Vparam1 =
Vparam2 =
Vparam3 =
Vparam4 =

# 1.e.4) How much the scale factor a or the product (aH) increases between
#        Hubble crossing for the pivot scale (during inflation) and the end of
#        inflation. You can pass either: 'N_star' (standing for
#        log(a_end/a_pivot)) set to a number; or 'ln_aH_ratio' (standing for
#        log(aH_end/aH_pivot)) set to a number; (default: 'N_star' set to 60)
#ln_aH_ratio = 50
#N_star = 55

# 1.e.5) Should the inflation module do its nomral job of numerical integration
#        ('numerical') or use analytical slow-roll formulas to infer the
#        primordial spectrum from the potential ('analytical')? (default:
#        'inflation_behavior' set to 'numerical')
#inflation_behavior = numerical

# 1.f) For type 'two_scales' (currently this option works only for scalar modes,
#      and either for pure adiabatic modes or adiabatic + one type of
#      isocurvature):
# 1.f.1) Two wavenumbers 'k1' and 'k2' in 1/Mpc, at which primordial amplitude
#        parameters will be given. The value of 'k_pivot' will not be used in
#        input but quantities at k_pivot will still be calculated and stored in
#        the primordial structure (no default value: compulsory input if
#        'P_k_ini type' has been set to 'two_scales')
k1=0.002
k2=0.1

# 1.f.2) Two amplitudes 'P_{RR}^1', 'P_{RR}^2' for the adiabatic primordial
#        spectrum (no default value: compulsory input if 'P_k_ini type' has been
#        set to 'two_scales')
P_{RR}^1 = 2.3e-9
P_{RR}^2 = 2.3e-9

# 1.f.3) If one isocurvature mode has been turned on ('ic' set e.g. to 'ad,cdi'
#        or 'ad,nid', etc.), enter values of the isocurvature amplitude
#        'P_{II}^1', 'P_{II}^2', and cross-correlation amplitude 'P_{RI}^1',
#        '|P_{RI}^2|' (see Planck paper on inflation for details on
#        definitions)
P_{II}^1 = 1.e-11
P_{II}^2 = 1.e-11
P_{RI}^1 = -1.e-13
|P_{RI}^2| = 1.e-13

# 1.f.4) Set 'special iso' to 'axion' or 'curvaton' for two particular cases:
#        'axion' means uncorrelated, n_ad equal to n_iso, 'curvaton' means fully
#        anti-correlated with f_iso<0 (in the conventions of the Planck
#        inflation paper this would be called fully correlated), n_iso equal
#        to one; in these two cases, the last three of the four paramneters in
#        2.c.3 will be over-written give the input for 'P_{II}^1' (defaut:
#        'special_iso' left blanck, code assumes general case described by four
#        parameters of 2.c.3)
special_iso =

# 1.g) For type 'external_Pk' (see external documentation external_Pk/README.md
#      for more details):
# 1.g.1) Command generating the table. If the table is already generated, just
#        write "cat <table_file>". The table should have two columns (k, pk) if
#        tensors are not requested, or three columns (k, pks, pkt) if they are.
#command = python external/external_Pk/generate_Pk_example.py
#command = python external/external_Pk/generate_Pk_example_w_tensors.py
command = cat external/external_Pk/Pk_example.dat
#command = cat external/external_Pk/Pk_example_w_tensors.dat

# 1.g.2) If the table is not pregenerated, parameters to be passed to the
#        command, in the right order, starting from "custom1" and up to
#        "custom10". They must be real numbers.
custom1 = 0.05     # In the example command: k_pivot
custom2 = 2.215e-9 # In the example command: A_s
custom3 = 0.9624   # In the example command: n_s
custom4 = 2e-10    # In the example (with tensors) command: A_t
custom5 = -0.1     # In the example (with tensors) command: n_t
#custom6 = 0
#custom7 = 0
#custom8 = 0
#custom9 = 0
#custom10 = 0



# -------------------------
# ----> Spectra parameters:
# -------------------------

# 1) Maximum l for CLs:
#        - 'l_max_scalars' for CMB scalars (temperature, polarization, cmb
#          lensing potential),
#        - 'l_max_vectors' for CMB vectors
#        - 'l_max_tensors' for CMB tensors (temperature, polarization)
#        - 'l_max_lss'     for Large Scale Structure Cls (density, galaxy
#          lensing potential)
#    Reducing 'l_max_lss' with respect to l_max_scalars reduces the execution
#    time significantly (default: set 'l_max_scalars' to 2500, 'l_max_vectors'
#    to 500, 'l_max_tensors' to 500, 'l_max_lss' to 300)
l_max_scalars = 2500
#l_max_vectors = 500
l_max_tensors = 500
#l_max_lss = 300


# 2) Parameters for the the matter density number count (option 'nCl'
#    (or 'dCl')) or galaxy lensing potential (option 'sCl') Cls:
# 2.a) Enter here a description of the selection functions W(z) of each redshift
#      bin; selection can be set to 'gaussian', 'tophat' or 'dirac', then pass a
#      list of N mean redshifts in growing order separated by comas, 1 or N
#      widths separated by comas, 1 or N bias separated by a comma, and 1 or N
#      magnification bias separated by a comma. The width stands for one
#      standard deviation of the gaussian (in z space), or for the half-width of
#      the top-hat. Finally, non_diagonal sets the number of cross-correlation
#      spectra that you want to calculate: 0 means only auto-correlation, 1
#      means only adjacent bins, and number of bins minus one means all
#      correlations (default: set to 'gaussian',1,0.1,1.,0.,0)
#
#      NOTE: For good performances, the code uses the Limber approximation for
#            nCl. If you want high precision even with thin selection functions,
#            increase the default value of the precision parameters
#            l_switch_limber_for_nc_local_over_z and
#            l_switch_limber_for_nc_los_over_z; for instance, add them to the
#            input file with values 10000 and 2000, instead of the default 100
#            and 30.
selection=gaussian
selection_mean = 0.98,0.99,1.0,1.1,1.2
selection_width = 0.1
selection_bias =
selection_magnification_bias =
non_diagonal=4

# 2.b) It is possible to multiply the window function W(z) by a selection
#      function 'dNdz' (number of objects per redshift interval). Type the name
#      of the file containing the redshift in the first column and the number of
#      objects in the second column (do not call it 'analytic*'). Set to
#      'analytic' to use instead the analytic expression from arXiv:1004.4640
#      (this function can be tuned in the module transfer.c, in the subroutine
#      transfer_dNdz_analytic). Leave blank to use a uniform distribution
#      (default).
dNdz_selection =

# 2.c) It is possible to consider source number counts evolution. Type the name
#      of the file containing the redshift on the first column and the number
#      of objects on the second column (do not call it 'analytic*'). Set to
#      'analytic' to use instead the analytic expression from Eq. 48 of
#      arXiv:1105.5292. Leave blank to use constant comoving number densities
#      (default).
dNdz_evolution =


# 3) Power spectrum P(k)
# 3.a) Maximum k in P(k), 'P_k_max_h/Mpc' in units of h/Mpc or 'P_k_max_1/Mpc'
#      inunits of 1/Mpc. If scalar Cls are also requested, a minimum value is
#      automatically imposed (the same as in scalar Cls computation) (default:
#      set to 1 1/Mpc)
P_k_max_h/Mpc = 1.
#P_k_max_1/Mpc = 0.7

# 3.a.1) If you want to use a different value for k_max in the primordial and
#        perturbations structures, specify
#        'primordial_P_k_max_h/Mpc' in units of h/Mpc or
#        'primordial_P_k_max_1/Mpc' in units of 1/Mpc
#        to define the maximum value of k for primordial power spectrum. By doing
#        so, 'P_k_max_h/Mpc' will only apply to perturbations. If unspecified,
#        'primordial_P_k_max_h/Mpc' is assumed to be the same as 'P_k_max_h/Mpc'.
#primordial_P_k_max_h/Mpc =
#primordial_P_k_max_1/Mpc =

# 3.b) Value(s) 'z_pk' of redshift(s) for P(k,z) output file(s); can be ordered
#      arbitrarily, but must be separated by comas (default: set 'z_pk' to 0)
z_pk = 0
#z_pk = 0., 1.2, 3.5

# 3.c) If the code is interfaced with routines that need to interpolate P(k,z) at
#      various values of (k,z), enter 'z_max_pk', the maximum value of z at
#      which such interpolations are needed. (default: set to maximum value in
#      above 'z_pk' input)
#z_max_pk = 10.



# ----------------------------------
# ----> Lensing parameters:
# ----------------------------------

# 1) Relevant only if you ask for 'tCl, lCl' and/or 'pCl, lCl': if you want the
#    spectrum of lensed Cls. Can be anything starting with 'y' or 'n'.
#    (default: no lensed Cls)
lensing = yes


# 2) Should the lensing potential [phi+psi](k,tau) and the lensing spectrum Cl_phiphi be rescaled?
#    You can rescale [phi+psi](k,tau) at all times by an overall amplitude and tilt: lcmb_rescale*(k/lcmb_pivot)**lcmb_tilt
#    Or, more simply, you can pass the usual parameter 'A_l', and the potential will be rescaled by sqrt(A_L)
#    (matches standard definition in Calabrese et al. 0803.2309)
#    (default: no rescaling: A_L=lcmb_rescale=1)
A_L =
#lcmb_rescale = 1
#lcmb_tilt = 0
#lcmb_pivot = 0.1

# In general, do we want to use the full Limber scheme introduced in
# v3.2.2? With this full Limber scheme, the calculation of the CMB
# lensing potential spectrum C_l^phiphi for l > ppr->l_switch_limber
# is based on a new integration scheme. Compared to the previous
# scheme, which can be recovered by setting this parameter to 'no',
# the new scheme uses a larger k_max and a coarser k-grid (or q-grid)
# than the CMB transfer function. The new scheme is used by default,
# because the old one is inaccurate at large l due to the too small
# k_max. Set to anything starting with the letter 'y' or
# 'n'. (default: 'yes')
want_lcmb_full_limber = yes

# -------------------------------------
# ----> Distortions parameters:
# -------------------------------------

# 1) Which kind of appriximation would you like to use for the calculation of the
#    branching ratios?
#    To use approximation 1) 'branching approx'=sharp_sharp
#    To use approximation 2) 'branching approx'=sharp_soft
#    To use approximation 3) 'branching approx'=soft_soft
#    To use approximation 4) 'branching approx'=soft_soft_cons
#    To use approximation 5) 'branching approx'=exact
#
#    Approximation 3) violates energy conservation in the plasma, and is discouraged.
#    Please be aware, that the total energy injected will NOT include the residual distortion energy.
#    (default set to 'exact')
sd_branching_approx = exact

# 1.a) If the branching ratios are = exact, the user can specify additional parameters. For any other
#      branching ratio approximation, all of the following parameters are going to be ignored.
# 1.a.1) How many multipoles do you want to use for the residuals in the case of the PCA
#        analysis? The value can vary between 0 and 6. (default set to 2)
sd_PCA_size = 2

# 1.a.2) If PCA size is different from 0, you need to specify the chosen detector, by defining
#        setting "sd_detector_name" and defining any of 1.a.3.x).
#        In external/distortions, the file detectors_list.dat contains a list
#        of currently "known" detectors, i.e. detectors for which the PCA decomposition
#        has already been computed. If no detector name is specified, but the detector specifics
#        1.a.3.x) are, the name will be created automatically. If no name and no specifics are
#        given, PIXIE values are going to be assumed.
#
#        For instance, in the case of "sd_detector_name=PIXIE", the values are fixed respectively to 30 GHz,
#        1005 GHz, 15 GHz and 5 10^-26 W/(m^2 Hz sr), and all spectral shapes and branching ratios are
#        already precumputed.
#
#        It would be very helpful if, once computed the vectors for a new detector, the user would
#        send us the files containing spectral shapes and branching ratios (see e.g.
#        external/distortions for templates).
sd_detector_name = PIXIE

# 1.a.3) Provide the specifics of the detector (frequencies and sensitivities)
#        Either define a path to a full noise file or enter the specifics here
# 1.a.3.1) Give a path to the full noise file containing
#            - the frequency array in GHz and
#            - the detector noise in 10^-26 W/(m^2 Hz sr) for each frequency
#          Please supply the path relative to external/distortions
#sd_detector_file = FIRAS_nu_delta_I.dat

# 1.a.3.2) If you did not supply the full noise file, you need to set
#            - the minumum frequency in GHz
#            - the maximum frequency in GHz
#            - the bin width in GHz or alternatively the number of bins
#            - the detector noise in 10^-26 W/(m^2 Hz sr)
#          of the chosen detector.
#sd_detector_nu_min = 30.
#sd_detector_nu_max = 1000.
#sd_detector_nu_delta = 15.
#sd_detector_bin_number = 65
#sd_detector_delta_Ic = 5.


# 2) Only calculate non-LCDM contributions to heating?
#    Sometimes, for comparison, one might want to disable all LCDM contributions to SDs.
#    Can be set to anything starting with 'y' or 'n' (default: no)
sd_only_exotic = no


# 3) Include g distortions?
#    Can be set to anything starting with 'y' or 'n' (default: no)
sd_include_g_distortion = no


# 4) If you want to manually add a y and/or a mu parameter on top of the calculated values, specify
#    'sd_add_y' or 'sd_add_mu' or both (default set to 0 for both)
sd_add_y = 0.
sd_add_mu = 0.


# 5) Include SZ effect from reionization? Can be set to anything starting with 'y' or 'n'
#    (default: no)
include_SZ_effect = no

# 5.a) Specify the type of approximation you want to use for the SZ effect
#        - by setting 'sd_reio_type' to 'Nozawa_2005', the approximation by Nozawa et al. 2005 is employed.
#        - by setting 'sd_reio_type' to 'Chluba_2012', the approximation by Chluba et al. 2012 is employed.
#          Note that, for the moment, this appoximation is only valid for cluster temperatures lower
#          than few KeV.
#      (default set to 'Chluba_2012')
#sd_reio_type = Chluba_2012



# ----------------------------------
# ----> Output parameters:
# ----------------------------------

# 1) Output for external files
# 1.a) File name root 'root' for all output files (if Cl requested, written to
#      '<root>_cl.dat'; if P(k) requested, written to '<root>_pk.dat'; plus
#      similar files for scalars, tensors, pairs of initial conditions, etc.;
#      if file with input parameters requested, written to
#      '<root>_parameters.ini')
#      If no root is specified, the root will be set to 'output/<thisfilename>'
#      (default: output/<thisfilename>)
#root = output/test

# 1.a.1) If root is specified, do you want to keep overwriting the file,
#      or do you want to create files numbered as '<root>N_'.
#      Can be set to anything starting with 'y' or 'n' (default: no)
overwrite_root = no

# 1.b) Do you want headers at the beginning of each output file (giving
#      precisions on the output units/ format) ? Can be set to anything
#      starting with 'y' or 'n' (default: yes)
headers = yes

# 1.c) In all output files, do you want columns to be normalized and ordered
#      with the default CLASS definitions or with the CAMB definitions (often
#      idential to the CMBFAST one) ? Set 'format' to either 'class', 'CLASS',
#      'camb' or 'CAMB' (default: 'class')
format = class

# 1.d) Do you want to write a table of background quantitites in a file? This
#      will include H, densities, Omegas, various cosmological distances, sound
#      horizon, etc., as a function of conformal time, proper time, scale
#      factor. Can be set to anything starting with 'y' or 'no' (default: no)
write_background = no

# 1.e) Do you want to write a table of thermodynamics quantitites in a file?
#      Can be set to anything starting with 'y' or 'n'. (default: no)
write_thermodynamics = no

# 1.f) Do you want to write a table of perturbations to files for certain
#      wavenumbers k? Dimension of k is 1/Mpc. The actual wave numbers are
#      chosen such that they are as close as possible to the requested k-values. (default: none)
#k_output_values = 0.01, 0.1, 0.0001

# 1.g) Do you want to write the primordial scalar(/tensor) spectrum in a file,
#      with columns k [1/Mpc], P_s(k) [dimensionless], ( P_t(k)
#      [dimensionless])? Can be set to anything starting with 'y' or 'n'. (default: no)
write_primordial = no

# 1.h) Do you want to write the exotic energy injection function in a file,
#     with columns z [dimensionless], dE/dz_inj, dE/dz_dep [J/(m^3 s)]?
# 1.i) Do you want to write also the non-injected photon heating?
#     File created if 'write_exotic_injection' or
#     'write_noninjection' set to something containing the letter
#     'y' or 'Y', file written, otherwise not written (default: no)
write_exotic_injection = no
#write_noninjection = no

# 1.k) Do you want to write the spectral distortions in a file,
#     with columns x [dimensionless], DI(x) [dimensionless]?
#     File created if 'write_distortions' set to something containing the letter
#     'y' or 'Y', file written, otherwise not written (default: no)
write_distortions = no

# 1.l) Do you want to have all input/precision parameters which have been read
#      written in file '<root>parameters.ini', and those not written in file
#      '<root>unused_parameters' ? Can be set to anything starting with 'y'
#      or 'n'. (default: yes)
write_parameters = yes

# 1.m) Do you want a warning written in the standard output when an input
#      parameter or value could not be interpreted ? Can be set to anything starting
#      with 'y' or 'n' (default: no)
write_warnings = no

# 2) Amount of information sent to standard output: Increase integer values
#    to make each module more talkative (default: all set to 0)
input_verbose = 1
background_verbose = 1
thermodynamics_verbose = 1
perturbations_verbose = 1
transfer_verbose = 1
primordial_verbose = 1
harmonic_verbose = 1
fourier_verbose = 1
lensing_verbose = 1
distortions_verbose = 1
output_verbose = 1

```

## many_times.py

```python
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import necessary modules
# uncomment to get plots displayed in notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from classy import Class
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import math


# In[ ]:


#############################################
#
# User settings controlling the figure aspect
#
z_max_pk = 46000       # highest redshift involved
k_per_decade = 400     # number of k values, controls final resolution
k_min_tau0 = 40.       # this value controls the minimum k value in the figure (it is k_min * tau0)
P_k_max_inv_Mpc =1.0   # this value is directly the maximum k value in the figure in Mpc
tau_num_early = 2000   # number of conformal time values before recombination, controls final resolution
tau_num_late = 200     # number of conformal time values after recombination, controls final resolution
tau_ini = 10.          # first value of conformal time in Mpc
tau_label_Hubble = 20. # value of time at which we want to place the label on Hubble crossing
tau_label_ks = 40.     # value of time at which we want to place the label on sound horizon crossing
tau_label_kd = 230.    # value of time at which we want to place the label on damping scale crossing
#
# Cosmological parameters and other CLASS parameters
#
common_settings = {# which output? transfer functions only
                   'output':'mTk',
                   # LambdaCDM parameters
                   'h':0.67556,
                   'omega_b':0.022032,
                   'omega_cdm':0.12038,
                   'A_s':2.215e-9,
                   'n_s':0.9619,
                   'tau_reio':0.0925,
                   # Take fixed value for primordial Helium (instead of automatic BBN adjustment)
                   'YHe':0.246,
                   # other output and precision parameters
                   'z_max_pk':z_max_pk,
                   'k_per_decade_for_pk':k_per_decade,
                   'k_per_decade_for_bao':k_per_decade,
                   'k_min_tau0':k_min_tau0, # this value controls the minimum k value in the figure
                   'perturbations_sampling_stepsize':'0.05',
                   'P_k_max_1/Mpc':P_k_max_inv_Mpc,
                   'compute damping scale':'yes', # needed to output and plot Silk damping scale
                   'gauge':'newtonian',
                   'matter_source_in_current_gauge':'yes'}

###############
#   
# call CLASS 
#
###############
M = Class()
M.set(common_settings)
M.compute()
#
# define conformal time sampling array
#
times = M.get_current_derived_parameters(['tau_rec','conformal_age'])
tau_rec=times['tau_rec']
tau_0 = times['conformal_age']
tau1 = np.logspace(math.log10(tau_ini),math.log10(tau_rec),tau_num_early)
tau2 = np.logspace(math.log10(tau_rec),math.log10(tau_0),tau_num_late)[1:]
tau2[-1] *= 0.999 # this tiny shift avoids interpolation errors
tau = np.concatenate((tau1,tau2))
tau_num = len(tau)
#
# use table of background and thermodynamics quantitites to define some functions 
# returning some characteristic scales
# (of Hubble crossing, sound horizon crossing, etc.) at different time
#
background = M.get_background() # load background table
#print background.viewkeys()
thermodynamics = M.get_thermodynamics() # load thermodynamics table
#print thermodynamics.viewkeys()    
#
background_tau = background['conf. time [Mpc]'] # read conformal times in background table
background_z = background['z'] # read redshift
background_aH = 2.*math.pi*background['H [1/Mpc]']/(1.+background['z'])/M.h() # read 2pi * aH in [h/Mpc]
background_ks = 2.*math.pi/background['comov.snd.hrz.']/M.h() # read 2pi/(comoving sound horizon) in [h/Mpc]
background_rho_m_over_r =\
    (background['(.)rho_b']+background['(.)rho_cdm'])\
    /(background['(.)rho_g']+background['(.)rho_ur']) # read rho_r / rho_m (to find time of equality)
background_rho_l_over_m =\
    background['(.)rho_lambda']\
    /(background['(.)rho_b']+background['(.)rho_cdm']) # read rho_m / rho_lambda (to find time of equality)
thermodynamics_tau = thermodynamics['conf. time [Mpc]'] # read confromal times in thermodynamics table
thermodynamics_kd = 2.*math.pi/thermodynamics['r_d']/M.h() # read 2pi(comoving diffusion scale) in [h/Mpc]
#
# define a bunch of interpolation functions based on previous quantities
#
background_z_at_tau = interp1d(background_tau,background_z)
background_aH_at_tau = interp1d(background_tau,background_aH)
background_ks_at_tau = interp1d(background_tau,background_ks)
background_tau_at_mr = interp1d(background_rho_m_over_r,background_tau)
background_tau_at_lm = interp1d(background_rho_l_over_m,background_tau)
thermodynamics_kd_at_tau = interp1d(thermodynamics_tau, thermodynamics_kd)
#
# infer arrays of characteristic quantitites calculated at values of conformal time in tau array
#
aH = background_aH_at_tau(tau) 
ks = background_ks_at_tau(tau)
kd = thermodynamics_kd_at_tau(tau)
#
# infer times of R/M and M/Lambda equalities
#
tau_eq = background_tau_at_mr(1.)
tau_lambda = background_tau_at_lm(1.)
#
# check and inform user whether intiial arbitrary choice of z_max_pk was OK
max_z_needed = background_z_at_tau(tau[0])
if max_z_needed > z_max_pk:
    print ('you must increase the value of z_max_pk to at least ',max_z_needed)
    () + 1  # this strange line is just a trick to stop the script execution there
else:
    print ('in a next run with the same values of tau, you may decrease z_max_pk from ',z_max_pk,' to ',max_z_needed)
#
# get transfer functions at each time and build arrays Theta0(tau,k) and phi(tau,k)
#
for i in range(tau_num):
    one_time = M.get_transfer(background_z_at_tau(tau[i])) # transfer functions at each time tau
    if i ==0:   # if this is the first time in the loop: create the arrays (k, Theta0, phi)
        k = one_time['k (h/Mpc)']
        k_num = len(k)
        Theta0 = np.zeros((tau_num,k_num))
        phi = np.zeros((tau_num,k_num))
    Theta0[i,:] = 0.25*one_time['d_g'][:]
    phi[i,:] = one_time['phi'][:]
#
# find the global extra of Theta0(tau,k) and phi(tau,k), used to define color code later
#
Theta_amp = max(Theta0.max(),-Theta0.min()) 
phi_amp = max(phi.max(),-phi.min()) 
#
# reshaping of (k,tau) necessary to call the function 'pcolormesh'
#
K,T = np.meshgrid(k,tau)
#
# inform user of the size of the grids (related to the figure resolution)
#
print ('grid size:',len(k),len(tau),Theta0.shape)
#
#################
#
# start plotting
#
#################
#
fig = plt.figure(figsize=(18,8)) 
#
# plot Theta0(k,tau)
#
ax_Theta = fig.add_subplot(121)
print ('> Plotting Theta_0')
fig_Theta = ax_Theta.pcolormesh(K,T,Theta0,cmap='coolwarm',vmin=-Theta_amp,vmax=Theta_amp,shading='auto')
print ('> Done')
#
# plot lines (characteristic times and scales)
#
ax_Theta.axhline(y=tau_rec,color='k',linestyle='-')
ax_Theta.axhline(y=tau_eq,color='k',linestyle='-')
ax_Theta.axhline(y=tau_lambda,color='k',linestyle='-')
ax_Theta.plot(aH,tau,'r-',linewidth=2)
ax_Theta.plot(ks,tau,color='#FFFF33',linestyle='-',linewidth=2)
ax_Theta.plot(kd,tau,'b-',linewidth=2)
#
# dealing with labels
#
ax_Theta.set_title(r'$\Theta_0$')
ax_Theta.text(1.5*k[0],0.9*tau_rec,r'$\mathrm{rec.}$')
ax_Theta.text(1.5*k[0],0.9*tau_eq,r'$\mathrm{R/M} \,\, \mathrm{eq.}$')
ax_Theta.text(1.5*k[0],0.9*tau_lambda,r'$\mathrm{M/L} \,\, \mathrm{eq.}$')
ax_Theta.annotate(r'$\mathrm{Hubble} \,\, \mathrm{cross.}$',
                  xy=(background_aH_at_tau(tau_label_Hubble),tau_label_Hubble),
                  xytext=(0.1*background_aH_at_tau(tau_label_Hubble),0.8*tau_label_Hubble),
                  arrowprops=dict(facecolor='black', shrink=0.05, width=1, headlength=5, headwidth=5))
ax_Theta.annotate(r'$\mathrm{sound} \,\, \mathrm{horizon} \,\, \mathrm{cross.}$',
                  xy=(background_ks_at_tau(tau_label_ks),tau_label_ks),
                  xytext=(0.07*background_aH_at_tau(tau_label_ks),0.8*tau_label_ks),
                  arrowprops=dict(facecolor='black', shrink=0.05, width=1, headlength=5, headwidth=5))
ax_Theta.annotate(r'$\mathrm{damping} \,\, \mathrm{scale} \,\, \mathrm{cross.}$',
                  xy=(thermodynamics_kd_at_tau(tau_label_kd),tau_label_kd),
                  xytext=(0.2*thermodynamics_kd_at_tau(tau_label_kd),2.0*tau_label_kd),
                  arrowprops=dict(facecolor='black', shrink=0.05, width=1, headlength=5, headwidth=5))
#
# dealing with axes
#
ax_Theta.set_xlim(k[0],k[-1])
ax_Theta.set_xscale('log')
ax_Theta.set_yscale('log')
ax_Theta.set_xlabel(r'$k  \,\,\, \mathrm{[h/Mpc]}$')
ax_Theta.set_ylabel(r'$\tau   \,\,\, \mathrm{[Mpc]}$')
ax_Theta.invert_yaxis()
#
# color legend
#
fig.colorbar(fig_Theta)
#
# plot phi(k,tau)
#
ax_phi = fig.add_subplot(122)
ax_phi.set_xlim(k[0],k[-1])
#ax_phi.pcolor(K,T,phi,cmap='coolwarm')
print ('> Plotting phi')
fig_phi = ax_phi.pcolormesh(K,T,phi,cmap='coolwarm',vmin=-0.,vmax=phi_amp,shading='auto')
print ('> Done')
#
# plot lines (characteristic times and scales)
#
ax_phi.axhline(y=tau_rec,color='k',linestyle='-')
ax_phi.axhline(y=tau_eq,color='k',linestyle='-')
ax_phi.axhline(y=tau_lambda,color='k',linestyle='-')
ax_phi.plot(aH,tau,'r-',linewidth=2)
ax_phi.plot(ks,tau,color='#FFFF33',linestyle='-',linewidth=2)
#
# dealing with labels
#
ax_phi.set_title(r'$\phi$')
ax_phi.text(1.5*k[0],0.9*tau_rec,r'$\mathrm{rec.}$')
ax_phi.text(1.5*k[0],0.9*tau_eq,r'$\mathrm{R/M} \,\, \mathrm{eq.}$')
ax_phi.text(1.5*k[0],0.9*tau_lambda,r'$\mathrm{M/L} \,\, \mathrm{eq.}$')
ax_phi.annotate(r'$\mathrm{Hubble} \,\, \mathrm{cross.}$',
                  xy=(background_aH_at_tau(tau_label_Hubble),tau_label_Hubble),
                  xytext=(0.1*background_aH_at_tau(tau_label_Hubble),0.8*tau_label_Hubble),
                  arrowprops=dict(facecolor='black', shrink=0.05, width=1, headlength=5, headwidth=5))
ax_phi.annotate(r'$\mathrm{sound} \,\, \mathrm{horizon} \,\, \mathrm{cross.}$',
                  xy=(background_ks_at_tau(tau_label_ks),tau_label_ks),
                  xytext=(0.07*background_aH_at_tau(tau_label_ks),0.8*tau_label_ks),
                  arrowprops=dict(facecolor='black', shrink=0.05, width=1, headlength=5, headwidth=5))
#
# dealing with axes
#
ax_phi.set_xscale('log')
ax_phi.set_yscale('log')
ax_phi.set_xlabel(r'$k \,\,\, \mathrm{[h/Mpc]}$')
ax_phi.set_ylabel(r'$\tau \,\,\, \mathrm{[Mpc]}$')
ax_phi.invert_yaxis()
#
# color legend
#
fig.colorbar(fig_phi)
#
# produce the PDF
#
#plt.show()
plt.savefig('many_times.png',dpi=300)



```

## neutrinohierarchy.py

```python
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import necessary modules
# uncomment to get plots displayed in notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from classy import Class
from scipy.optimize import fsolve


# In[ ]:


# a function returning the three masses given the Delta m^2, the total mass, and the hierarchy (e.g. 'IN' or 'IH')
# taken from a piece of MontePython written by Thejs Brinckmann
def get_masses(delta_m_squared_atm, delta_m_squared_sol, sum_masses, hierarchy):
    # any string containing letter 'n' will be considered as refering to normal hierarchy
    if 'n' in hierarchy.lower():
        # Normal hierarchy massive neutrinos. Calculates the individual
        # neutrino masses from M_tot_NH and deletes M_tot_NH
        #delta_m_squared_atm=2.45e-3
        #delta_m_squared_sol=7.50e-5
        m1_func = lambda m1, M_tot, d_m_sq_atm, d_m_sq_sol: M_tot**2. + 0.5*d_m_sq_sol - d_m_sq_atm + m1**2. - 2.*M_tot*m1 - 2.*M_tot*(d_m_sq_sol+m1**2.)**0.5 + 2.*m1*(d_m_sq_sol+m1**2.)**0.5
        m1,opt_output,success,output_message = fsolve(m1_func,sum_masses/3.,(sum_masses,delta_m_squared_atm,delta_m_squared_sol),full_output=True)
        m1 = m1[0]
        m2 = (delta_m_squared_sol + m1**2.)**0.5
        m3 = (delta_m_squared_atm + 0.5*(m2**2. + m1**2.))**0.5
        return m1,m2,m3
    else:
        # Inverted hierarchy massive neutrinos. Calculates the individual
        # neutrino masses from M_tot_IH and deletes M_tot_IH
        #delta_m_squared_atm=-2.45e-3
        #delta_m_squared_sol=7.50e-5
        delta_m_squared_atm = -delta_m_squared_atm
        m1_func = lambda m1, M_tot, d_m_sq_atm, d_m_sq_sol: M_tot**2. + 0.5*d_m_sq_sol - d_m_sq_atm + m1**2. - 2.*M_tot*m1 - 2.*M_tot*(d_m_sq_sol+m1**2.)**0.5 + 2.*m1*(d_m_sq_sol+m1**2.)**0.5
        m1,opt_output,success,output_message = fsolve(m1_func,sum_masses/3.,(sum_masses,delta_m_squared_atm,delta_m_squared_sol),full_output=True)
        m1 = m1[0]
        m2 = (delta_m_squared_sol + m1**2.)**0.5
        m3 = (delta_m_squared_atm + 0.5*(m2**2. + m1**2.))**0.5
        return m1,m2,m3


# In[ ]:


# test of this function, returning the 3 masses for total mass of 0.1eV
m1,m2,m3 = get_masses(2.45e-3,7.50e-5,0.1,'NH')
print ('NH:',m1,m2,m3,m1+m2+m3)
m1,m2,m3 = get_masses(2.45e-3,7.50e-5,0.1,'IH')
print ('IH:',m1,m2,m3,m1+m2+m3)


# In[ ]:


# The goal of this cell is to compute the ratio of P(k) for NH and IH with the same total mass
commonsettings = {'N_ur':0,
                  'N_ncdm':3,
                  'output':'mPk',
                  'P_k_max_1/Mpc':3.0,
                  # The next line should be uncommented for higher precision (but significantly slower running)
                  'ncdm_fluid_approximation':3,
                  # You may uncomment this line to get more info on the ncdm sector from Class:
                  'background_verbose':1
                 }

# array of k values in 1/Mpc
kvec = np.logspace(-4,np.log10(3),100)
# array for storing legend
legarray = []

# loop over total mass values
for sum_masses in [0.1, 0.115, 0.13]:
    # normal hierarchy
    [m1, m2, m3] = get_masses(2.45e-3,7.50e-5, sum_masses, 'NH')
    NH = Class()
    NH.set(commonsettings)
    NH.set({'m_ncdm':str(m1)+','+str(m2)+','+str(m3)})
    NH.compute()
    # inverted hierarchy
    [m1, m2, m3] = get_masses(2.45e-3,7.50e-5, sum_masses, 'IH')
    IH = Class()
    IH.set(commonsettings)
    IH.set({'m_ncdm':str(m1)+','+str(m2)+','+str(m3)})
    IH.compute()
    pkNH = []
    pkIH = []
    for k in kvec:
        pkNH.append(NH.pk(k,0.))
        pkIH.append(IH.pk(k,0.))
    NH.struct_cleanup()
    IH.struct_cleanup()
    # extract h value to convert k from 1/Mpc to h/Mpc
    h = NH.h()
    plt.semilogx(kvec/h,1-np.array(pkNH)/np.array(pkIH))
    legarray.append(r'$\Sigma m_i = '+str(sum_masses)+'$eV')
plt.axhline(0,color='k')
plt.xlim(kvec[0]/h,kvec[-1]/h)
plt.xlabel(r'$k [h \mathrm{Mpc}^{-1}]$')
plt.ylabel(r'$1-P(k)^\mathrm{NH}/P(k)^\mathrm{IH}$')
plt.legend(legarray)    
plt.savefig('neutrinohierarchy.pdf')


```

## one_k.py

```python
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import necessary modules
# uncomment to get plots displayed in notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from classy import Class
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import math


# In[ ]:


#############################################
#
# value of k that we want to follow in [1/Mpc]
#
k = 0.5  # 1/Mpc
#
# Cosmological parameters and other CLASS parameters
#
common_settings = {# we need to set the output field to something although
                   # the really releveant outpout here will be set with 'k_output_values'
                   'output':'mPk',
                   # value of k we want to polot in [1/Mpc]
                   'k_output_values':k,
                   # LambdaCDM parameters
                   'h':0.67810,
                   'omega_b':0.02238280,
                   'omega_cdm':0.1201075,
                   'A_s':2.100549e-09 ,
                   'n_s':0.9660499,
                   'tau_reio':0.05430842,
                   # Take fixed value for primordial Helium (instead of automatic BBN adjustment)
                   'YHe':0.2454,
                   # other options and settings
                   'compute damping scale':'yes', # needed to output the time of damping scale crossing
                   'gauge':'newtonian'}  
##############
#    
# call CLASS
#
M = Class()
M.set(common_settings)
M.compute()
#
# load perturbations
#
all_k = M.get_perturbations()  # this potentially constains scalars/tensors and all k values
print (all_k['scalar'][0].keys())
#    
one_k = all_k['scalar'][0]     # this contains only the scalar perturbations for the requested k values
#    
tau = one_k['tau [Mpc]']
Theta0 = 0.25*one_k['delta_g']
phi = one_k['phi']
psi = one_k['psi']
theta_b = one_k['theta_b']
a = one_k['a']
# compute related quantitites    
R = 3./4.*M.Omega_b()/M.Omega_g()*a    # R = 3/4 * (rho_b/rho_gamma)
zero_point = -(1.+R)*psi               # zero point of oscillations: -(1.+R)*psi
#
# get Theta0 oscillation amplitude (for vertical scale of plot)
#
Theta0_amp = max(Theta0.max(),-Theta0.min())
#
# get the time of decoupling
#
quantities = M.get_current_derived_parameters(['tau_rec'])
# print times.viewkeys()
tau_rec = quantities['tau_rec']
#
# use table of background quantitites to find the time of
# Hubble crossing (k / (aH)= 2 pi), sound horizon crossing (k * rs = 2pi)
#
background = M.get_background() # load background table
#print background.viewkeys()
#
background_tau = background['conf. time [Mpc]'] # read confromal times in background table
background_z = background['z'] # read redshift
background_k_over_aH = k/background['H [1/Mpc]']*(1.+background['z']) # read k/aH = k(1+z)/H
background_k_rs = k * background['comov.snd.hrz.'] # read k * rs
background_rho_m_over_r =\
    (background['(.)rho_b']+background['(.)rho_cdm'])\
    /(background['(.)rho_g']+background['(.)rho_ur']) # read rho_r / rho_m (to find time of equality)
#
# define interpolation functions; we want the value of tau when the argument is equal to 2pi (or 1 for equality)
#
tau_at_k_over_aH = interp1d(background_k_over_aH,background_tau)
tau_at_k_rs = interp1d(background_k_rs,background_tau)
tau_at_rho_m_over_r = interp1d(background_rho_m_over_r,background_tau)
#
# finally get these times
#
tau_Hubble = tau_at_k_over_aH(2.*math.pi)
tau_s = tau_at_k_rs(2.*math.pi)
tau_eq = tau_at_rho_m_over_r(1.)
#
#################
#
# start plotting
#
#################
#    
plt.xlim([tau[0],tau_rec*1.3])
plt.ylim([-1.3*Theta0_amp,1.3*Theta0_amp])
plt.xlabel(r'$\tau \,\,\, \mathrm{[Mpc]}$')
plt.title(r'$\mathrm{Transfer} (\tau,k) \,\,\, \mathrm{for} \,\,\, k=%g \,\,\, [1/\mathrm{Mpc}]$'%k)
plt.grid()
#
plt.axvline(x=tau_Hubble,color='r')
plt.axvline(x=tau_s,color='y')
plt.axvline(x=tau_eq,color='k')
plt.axvline(x=tau_rec,color='k')
#
plt.annotate(r'Hubble cross.',
                xy=(tau_Hubble,1.08*Theta0_amp),
                xytext=(0.15*tau_Hubble,1.18*Theta0_amp),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headlength=5, headwidth=5))
plt.annotate(r'sound hor. cross.',
                 xy=(tau_s,-1.0*Theta0_amp),
                 xytext=(1.5*tau_s,-1.2*Theta0_amp),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headlength=5, headwidth=5))
plt.annotate(r'eq.',
                 xy=(tau_eq,1.08*Theta0_amp),
                 xytext=(0.45*tau_eq,1.18*Theta0_amp),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headlength=5, headwidth=5))
plt.annotate(r'rec.',
                 xy=(tau_rec,1.08*Theta0_amp),
                 xytext=(0.45*tau_rec,1.18*Theta0_amp),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headlength=5, headwidth=5))
#
# Possibility to add functions one by one, saving between each (for slides)
#
plt.semilogx(tau,psi,'y-',label=r'$\psi$')
#plt.legend(loc='right',bbox_to_anchor=(1.4, 0.5))
#plt.savefig('one_k_1.pdf',bbox_inches='tight')
#
plt.semilogx(tau,phi,'r-',label=r'$\phi$')
#plt.legend(loc='right',bbox_to_anchor=(1.4, 0.5))
#plt.savefig('one_k_2.pdf',bbox_inches='tight')
#
plt.semilogx(tau,zero_point,'k:',label=r'$-(1+R)\psi$')
#plt.legend(loc='right',bbox_to_anchor=(1.4, 0.5))
#plt.savefig('one_k_3.pdf',bbox_inches='tight')
#
plt.semilogx(tau,Theta0,'b-',linewidth=2,label=r'$\Theta_0$')
#plt.legend(loc='right',bbox_to_anchor=(1.4, 0.5))
#plt.savefig('one_k_4.pdf',bbox_inches='tight')
#
plt.semilogx(tau,Theta0+psi,'c-',linewidth=2,label=r'$\Theta_0+\psi$')
#plt.legend(loc='right',bbox_to_anchor=(1.4, 0.5))
#plt.savefig('one_k_5.pdf',bbox_inches='tight')
#
plt.semilogx(tau,theta_b,'g-',label=r'$\theta_b$')
plt.legend(loc='right',bbox_to_anchor=(1.4, 0.5))
plt.savefig('one_k.pdf',bbox_inches='tight')
#


```

## one_time.py

```python
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import necessary modules
from classy import Class
from math import pi
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt


# In[ ]:


#####################################################
#
# Cosmological parameters and other CLASS parameters
#
#####################################################
common_settings = {# LambdaCDM parameters
                   'h':0.67810,
                   'omega_b':0.02238280,
                   'omega_cdm':0.12038,
                   'A_s':2.100549e-09,
                   'n_s': 0.9660499,
                   'tau_reio':0.05430842,
                   # output and precision parameters
                   'output':'tCl,mTk,vTk',
                   'l_max_scalars':5000,
                   'P_k_max_1/Mpc':10.0,
                   'gauge':'newtonian'
                   }


# In[ ]:


###############
#
# call CLASS a first time just to compute z_rec (will compute transfer functions at default: z=0)
#
###############
M = Class()
M.set(common_settings)
M.compute()
derived = M.get_current_derived_parameters(['z_rec','tau_rec','conformal_age'])
print (derived.keys())
z_rec = derived['z_rec']
z_rec = int(1000.*z_rec)/1000. # round down at 4 digits after coma
print ('z_rec=',z_rec)
#
# In the last figure the x-axis will show l/(tau_0-tau_rec), so we need (tau_0-tau_rec) in units of [Mpc/h]
#
tau_0_minus_tau_rec_hMpc = (derived['conformal_age']-derived['tau_rec'])*M.h()


# In[ ]:


################
#
# call CLASS again for the perturbations (will compute transfer functions at input value z_rec)
#
################
M.empty() # reset input parameters to default, before passing a new parameter set
M.set(common_settings)
M.set({'z_pk':z_rec})
M.compute()
#
# save the total Cl's (we will plot them in the last step)
#
cl_tot = M.raw_cl(5000)
#
#
# load transfer functions at recombination
#
one_time = M.get_transfer(z_rec)
print (one_time.keys())
k = one_time['k (h/Mpc)']
Theta0 = 0.25*one_time['d_g']
phi = one_time['phi']
psi = one_time['psi']
theta_b = one_time['t_b']
# compute related quantitites
R = 3./4.*M.Omega_b()/M.Omega_g()/(1+z_rec)  # R = 3/4 * (rho_b/rho_gamma) at z_rec
zero_point = -(1.+R)*psi                     # zero point of oscillations: -(1.+R)*psi
Theta0_amp = max(Theta0.max(),-Theta0.min()) # Theta0 oscillation amplitude (for vertical scale of plot)
print ('At z_rec: R=',R,', Theta0_amp=',Theta0_amp)


# In[ ]:


# use table of background quantitites to find the wavenumbers corresponding to
# Hubble crossing (k = 2 pi a H), sound horizon crossing (k = 2pi / rs)
#
background = M.get_background() # load background table
print (background.keys())
#
background_tau = background['conf. time [Mpc]'] # read confromal times in background table
background_z = background['z'] # read redshift
background_kh = 2.*pi*background['H [1/Mpc]']/(1.+background['z'])/M.h() # read kh = 2pi aH = 2pi H/(1+z) converted to [h/Mpc]
background_ks = 2.*pi/background['comov.snd.hrz.']/M.h() # read ks = 2pi/rs converted to [h/Mpc]
#
# define interpolation functions; we want the value of tau when the argument is equal to 2pi
#
from scipy.interpolate import interp1d
kh_at_tau = interp1d(background_tau,background_kh)
ks_at_tau = interp1d(background_tau,background_ks)
#
# finally get these scales
#
tau_rec = derived['tau_rec']
kh = kh_at_tau(tau_rec)
ks = ks_at_tau(tau_rec)
print ('at tau_rec=',tau_rec,', kh=',kh,', ks=',ks)


# In[ ]:


#####################
#
# call CLASS with TSW (intrinsic temperature + Sachs-Wolfe) and save
#
#####################
M.empty()           # clean input
M.set(common_settings) # new input
M.set({'temperature contributions':'tsw'})
M.compute()
cl_TSW = M.raw_cl(5000)


# In[ ]:


######################
#
# call CLASS with early ISW and save
#
######################
M.empty()
M.set(common_settings)
M.set({'temperature contributions':'eisw'})
M.compute()
cl_eISW = M.raw_cl(5000)


# In[ ]:


######################
#
# call CLASS with late ISW and save
#
######################
M.empty()
M.set(common_settings)
M.set({'temperature contributions':'lisw'})
M.compute()
cl_lISW = M.raw_cl(5000)


# In[ ]:


######################
#
# call CLASS with Doppler and save
#
######################
M.empty()
M.set(common_settings)
M.set({'temperature contributions':'dop'})
M.compute()
cl_Doppler = M.raw_cl(5000)


# In[ ]:


#################
#
# start plotting
#
#################
#
fig, (ax_Tk, ax_Tk2, ax_Cl) = plt.subplots(3,sharex=True,figsize=(8,12))
fig.subplots_adjust(hspace=0)
##################
#
# first figure with transfer functions
#
##################
ax_Tk.set_xlim([3.e-4,0.5])
ax_Tk.set_ylim([-1.1*Theta0_amp,1.1*Theta0_amp])
ax_Tk.tick_params(axis='x',which='both',bottom='off',top='on',labelbottom='off',labeltop='on')
ax_Tk.set_xlabel(r'$\mathrm{k} \,\,\,  \mathrm{[h/Mpc]}$')
ax_Tk.set_ylabel(r'$\mathrm{Transfer}(\tau_\mathrm{dec},k)$')
ax_Tk.xaxis.set_label_position('top')
ax_Tk.grid()
#
ax_Tk.axvline(x=kh,color='r')
ax_Tk.axvline(x=ks,color='y')
#
ax_Tk.annotate(r'Hubble cross.',
                xy=(kh,0.8*Theta0_amp),
                xytext=(0.15*kh,0.9*Theta0_amp),
                arrowprops=dict(facecolor='black', shrink=0.05, width=1, headlength=5, headwidth=5))
ax_Tk.annotate(r'sound hor. cross.',
                 xy=(ks,0.8*Theta0_amp),
                 xytext=(1.3*ks,0.9*Theta0_amp),
                 arrowprops=dict(facecolor='black', shrink=0.05, width=1, headlength=5, headwidth=5))
#
ax_Tk.semilogx(k,psi,'y-',label=r'$\psi$')
ax_Tk.semilogx(k,phi,'r-',label=r'$\phi$')
ax_Tk.semilogx(k,zero_point,'k:',label=r'$-(1+R)\psi$')
ax_Tk.semilogx(k,Theta0,'b-',label=r'$\Theta_0$')
ax_Tk.semilogx(k,(Theta0+psi),'c',label=r'$\Theta_0+\psi$')
ax_Tk.semilogx(k,theta_b,'g-',label=r'$\theta_b$')
#
ax_Tk.legend(loc='right',bbox_to_anchor=(1.4, 0.5))
#######################
#
# second figure with transfer functions squared
#
#######################
ax_Tk2.set_xlim([3.e-4,0.5])
ax_Tk2.tick_params(axis='x',which='both',bottom='off',top='off',labelbottom='off',labeltop='off')
ax_Tk2.set_ylabel(r'$\mathrm{Transfer}(\tau_\mathrm{dec},k)^2$')
ax_Tk2.grid()
#
ax_Tk2.semilogx(k,(Theta0+psi)**2,'c',label=r'$(\Theta_0+\psi)^2$')
#
ax_Tk2.legend(loc='right',bbox_to_anchor=(1.4, 0.5))
########################
#
# third figure with all contributions to Cls
#
# We already computed each contribution (TSW, earlyISW, lateISW, Doppler, total)
# Note that there is another contribution from polarisation. We don't plot it because it is
# too small to be seen, however it is included by default in the total.
#
# After each step we will save the figure (to get intermediate figures that can be used in slides)
#
#########################
# presentation settings
ax_Cl.set_xlim([3.e-4,0.5])
ax_Cl.set_ylim([0.,8.])
ax_Cl.set_xlabel(r'$\ell/(\tau_0-\tau_{rec}) \,\,\, \mathrm{[h/Mpc]}$')
ax_Cl.set_ylabel(r'$\ell (\ell+1) C_l^{TT} / 2 \pi \,\,\, [\times 10^{10}]$')
ax_Cl.tick_params(axis='x',which='both',bottom='on',top='off',labelbottom='on',labeltop='off')
ax_Cl.grid()
#
# plot and save with TSW
#
ax_Cl.semilogx(cl_TSW['ell']/tau_0_minus_tau_rec_hMpc,1.e10*cl_TSW['ell']*(cl_TSW['ell']+1.)*cl_TSW['tt']/2./pi,'c-',label=r'$\mathrm{T+SW}$')
#
ax_Cl.legend(loc='right',bbox_to_anchor=(1.4, 0.5))
fig.savefig('one_time_with_cl_1.pdf',bbox_inches='tight')
#
# plot and save with additionally early ISW and late ISW
#
ax_Cl.semilogx(cl_eISW['ell']/tau_0_minus_tau_rec_hMpc,1.e10*cl_eISW['ell']*(cl_eISW['ell']+1.)*cl_eISW['tt']/2./pi,'r-',label=r'$\mathrm{early} \,\, \mathrm{ISW}$')
ax_Cl.semilogx(cl_lISW['ell']/tau_0_minus_tau_rec_hMpc,1.e10*cl_lISW['ell']*(cl_lISW['ell']+1.)*cl_lISW['tt']/2./pi,'y-',label=r'$\mathrm{late} \,\, \mathrm{ISW}$')
#
ax_Cl.legend(loc='right',bbox_to_anchor=(1.4, 0.5))
fig.savefig('one_time_with_cl_2.pdf',bbox_inches='tight')
#
# plot and save with additionally Doppler
#
ax_Cl.semilogx(cl_Doppler['ell']/tau_0_minus_tau_rec_hMpc,1.e10*cl_Doppler['ell']*(cl_Doppler['ell']+1.)*cl_Doppler['tt']/2./pi,'g-',label=r'$\mathrm{Doppler}$')
#
ax_Cl.legend(loc='right',bbox_to_anchor=(1.4, 0.5))
fig.savefig('one_time_with_cl_3.pdf',bbox_inches='tight')
#
# plot and save with additionally total Cls
#
ax_Cl.semilogx(cl_tot['ell']/tau_0_minus_tau_rec_hMpc,1.e10*cl_tot['ell']*(cl_tot['ell']+1.)*cl_tot['tt']/2./pi,'k-',label=r'$\mathrm{Total}$')
#
ax_Cl.legend(loc='right',bbox_to_anchor=(1.4, 0.5))
fig.savefig('one_time_with_cl_tot.pdf',bbox_inches='tight')


```

## test_hmcode.py

```python
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import classy module
from classy import Class
# uncomment to get plots displayed in notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from math import pi
import numpy as np


# In[ ]:


# create instance of the class "Class"
LambdaCDM = Class()
# pass input parameters
LambdaCDM.set({'omega_b':0.0223828,'omega_cdm':0.1201075,'h':0.67810,'A_s':2.100549e-09,'n_s':0.9660499,'tau_reio':0.05430842})
LambdaCDM.set({'output':'tCl,pCl,lCl,mPk','analytic_nowiggle':'yes','numerical_nowiggle':'yes','lensing':'yes'})
LambdaCDM.set({'P_k_max_1/Mpc':3.0,'z_max_pk':1.1})
LambdaCDM.set({'non_linear':'HMcode','hmcode_version':'2020'})
# run class
LambdaCDM.compute()


# In[ ]:


kk = np.logspace(-4,np.log10(3),1000) # k in h/Mpc
h = LambdaCDM.h() # get reduced Hubble for conversions to 1/Mpc
Pk = [] # P(k) in (Mpc/h)**3
Pk_lin = [] # P(k) in (Mpc/h)**3
Pk_nw = []
Pk_an_nw = []


# In[ ]:


# get P(k) at redhsift z
z=0
for k in kk:
    Pk_lin.append(LambdaCDM.pk_lin(k*h,z)*h**3) # function .pk(k,z)
    Pk_nw.append(LambdaCDM.pk_numerical_nw(k*h,z)*h**3) # function .pk(k,z)
    Pk_an_nw.append(LambdaCDM.pk_analytic_nw(k*h)*h**3) # function .pk(k,z)
    Pk.append(LambdaCDM.pk(k*h,z)*h**3) # function .pk(k,z)


# In[ ]:


# plot P(k)
#plt.figure(1)
plt.xscale('log');plt.yscale('log')
#plt.xlim(kk[0],kk[-1])
plt.xlim(1.e-3,0.5)
plt.ylim(200,3e4)
plt.xlabel(r'$k \,\,\,\, [h/\mathrm{Mpc}]$')
plt.ylabel(r'$P(k) \,\,\,\, [\mathrm{Mpc}/h]^3$')
plt.plot(kk,Pk_nw,'k-')
plt.plot(kk,Pk_an_nw,'r-')
plt.plot(kk,Pk_lin,'g-')
plt.savefig('test_hmcode.pdf',bbox_inches='tight')


```

## thermo.py

```python
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import necessary modules
# uncomment to get plots displayed in notebook#%matplotlib inline
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from classy import Class
from scipy.optimize import fsolve
from scipy.interpolate import interp1d
import math


# In[ ]:


common_settings = {'output' : 'tCl',
                   # LambdaCDM parameters
                   'h':0.6781,
                   'omega_b':0.02238280,
                   'omega_cdm':0.1201075,
                   'A_s':2.100549e-09,
                   'n_s':0.9660499,
                   'tau_reio':0.05430842,
                   'thermodynamics_verbose':1
                   }  
##############
#    
# call CLASS
#
###############
M = Class()
M.set(common_settings)
M.compute()
derived = M.get_current_derived_parameters(['tau_rec','conformal_age','conf_time_reio'])
thermo = M.get_thermodynamics()
print (thermo.keys())


# In[ ]:


tau = thermo['conf. time [Mpc]']
g = thermo['g [Mpc^-1]']
# to make the reionisation peak visible, rescale g by 100 for late times
g[:500] *= 100
#################
#
# start plotting
#
#################
#    
plt.xlim([1.e2,derived['conformal_age']])
plt.xlabel(r'$\tau \,\,\, \mathrm{[Mpc]}$')
plt.ylabel(r'$\mathrm{visibility} \,\,\, g \,\,\, [\mathrm{Mpc}^{-1}]$')
plt.axvline(x=derived['tau_rec'],color='k')
plt.axvline(x=derived['conf_time_reio'],color='k')
#
plt.semilogx(tau,g,'r',label=r'$\psi$')
plt.savefig('thermo.pdf',bbox_inches='tight')


```

## varying_neff.py

```python
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import necessary modules
# uncomment to get plots displayed in notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from classy import Class
from scipy.optimize import fsolve
import math


# In[ ]:


############################################
#
# Varying parameter (others fixed to default)
#
var_name = 'N_ur'
var_array = np.linspace(3.044,5.044,5)
var_num = len(var_array)
var_legend = r'$N_\mathrm{eff}$'
var_figname = 'neff'
#
# Constraints to be matched
#
# As explained in the "Neutrino cosmology" book, CUP, Lesgourgues et al., section 5.3, the goal is to vary
# - omega_cdm by a factor alpha = (1 + coeff*Neff)/(1 + coeff*3.046)
# - h by a factor sqrt*(alpha)
# in order to keep a fixed z_equality(R/M) and z_equality(M/Lambda)
#
omega_b = 0.0223828
omega_cdm_standard = 0.1201075
h_standard = 0.67810
#
# coefficient such that omega_r = omega_gamma (1 + coeff*Neff),
# i.e. such that omega_ur = omega_gamma * coeff * Neff:
# coeff = omega_ur/omega_gamma/Neff_standard 
# We could extract omega_ur and omega_gamma on-the-fly within th script, 
# but for simplicity we did a preliminary interactive run with background_verbose=2
# and we copied the values given in the budget output.
#
coeff = 1.70961e-05/2.47298e-05/3.044
print ("coeff=",coeff)
#
#############################################
#
# Fixed settings
#
common_settings = {# fixed LambdaCDM parameters
                   'omega_b':omega_b,
                   'A_s':2.100549e-09,
                   'n_s':0.9660499,
                   'tau_reio':0.05430842,
                   # output and precision parameters
                   'output':'tCl,pCl,lCl,mPk',
                   'lensing':'yes',
                   'P_k_max_1/Mpc':3.0}
#
##############################################
#
# loop over varying parameter values
#
M = {}
#
for i, N_ur in enumerate(var_array):
    #
    # rescale omega_cdm and h
    #
    alpha = (1.+coeff*N_ur)/(1.+coeff*3.044)
    omega_cdm = (omega_b + omega_cdm_standard)*alpha - omega_b
    h = h_standard*math.sqrt(alpha)
    print (' * Compute with %s=%e, %s=%e, %s=%e'%('N_ur',N_ur,'omega_cdm',omega_cdm,'h',h))
    #
    # call CLASS
    #
    M[i] = Class()
    M[i].set(common_settings)
    M[i].set(better_precision_settings)
    M[i].set({'N_ur':N_ur})
    M[i].set({'omega_cdm':omega_cdm})
    M[i].set({'h':h})
    M[i].compute()


# In[ ]:


#############################################
#
# extract spectra and plot them
#
#############################################
kvec = np.logspace(-4,np.log10(3),1000) # array of kvec in h/Mpc
twopi = 2.*math.pi
#
# Create figures
#
fig_Pk, ax_Pk = plt.subplots()
fig_TT, ax_TT = plt.subplots()
#
# loop over varying parameter values
#
ll = {}
clM = {}
clTT = {}
pkM = {}
legarray = []

for i, N_ur in enumerate(var_array):
    #
    alpha = (1.+coeff*N_ur)/(1.+coeff*3.044)
    h = 0.67810*math.sqrt(alpha) # this is h
    #
    # deal with colors and legends
    #
    if i == 0:
        var_color = 'k'
        var_alpha = 1.
    else:
        var_color = plt.cm.Reds(0.8*i/(var_num-1))
    #
    # get Cls
    #
    clM[i] = M[i].lensed_cl(2500)
    ll[i] = clM[i]['ell'][2:]
    clTT[i] = clM[i]['tt'][2:]
    #
    # store P(k) for common k values
    #
    pkM[i] = []
    # The function .pk(k,z) wants k in 1/Mpc so we must convert kvec for each case with the right h 
    khvec = kvec*h # This is k in 1/Mpc
    for kh in khvec:
        pkM[i].append(M[i].pk(kh,0.)*h**3) 
    #    
    # plot P(k)
    #
    if i == 0:
        ax_Pk.semilogx(kvec,np.array(pkM[i])/np.array(pkM[0]),
                       color=var_color,#alpha=var_alpha,
                       linestyle='-')
    else:
        ax_Pk.semilogx(kvec,np.array(pkM[i])/np.array(pkM[0]),
                       color=var_color,#alpha=var_alpha,
                       linestyle='-',
                      label=r'$\Delta N_\mathrm{eff}=%g$'%(N_ur-3.044))
    #
    # plot C_l^TT
    #
    if i == 0:
        ax_TT.semilogx(ll[i],clTT[i]/clTT[0],
                       color=var_color,alpha=var_alpha,linestyle='-')
    else:    
        ax_TT.semilogx(ll[i],clTT[i]/clTT[0],
                       color=var_color,alpha=var_alpha,linestyle='-',
                      label=r'$\Delta N_\mathrm{eff}=%g$'%(N_ur-3.044))
#
# output of P(k) figure
#
ax_Pk.set_xlim([1.e-3,3.])
ax_Pk.set_ylim([0.98,1.20])
ax_Pk.set_xlabel(r'$k \,\,\,\, [h^{-1}\mathrm{Mpc}]$')
ax_Pk.set_ylabel(r'$P(k)/P(k)[N_\mathrm{eff}=3.046]$')
ax_Pk.legend(loc='upper left')
fig_Pk.tight_layout()
fig_Pk.savefig('ratio-%s-Pk.pdf' % var_figname)
#
# output of C_l^TT figure
#      
ax_TT.set_xlim([2,2500])
ax_TT.set_ylim([0.850,1.005])
ax_TT.set_xlabel(r'$\mathrm{Multipole} \,\,\,\,  \ell$')
ax_TT.set_ylabel(r'$C_\ell^\mathrm{TT}/C_\ell^\mathrm{TT}(N_\mathrm{eff}=3.046)$')
ax_TT.legend(loc='lower left')
fig_TT.tight_layout()
fig_TT.savefig('ratio-%s-cltt.pdf' % var_figname)


```

## varying_pann.py

```python
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import necessary modules
# uncomment to get plots displayed in notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from classy import Class
from scipy.optimize import fsolve
from math import pi


# In[ ]:


############################################
#
# Varying parameter (others fixed to default)
#
# With the input suntax of class <= 2.9 we used: annihilation = 1.e-5 m^3/s/Kg
# With the new syntax this is equivalent to DM_annihilation_efficiency = 1.11e-22 m^3/s/J
# (the ratio is a factor (c/[1 m/s])**2 = 9.e16)
#
var_name = 'DM_annihilation_efficiency'
var_array = np.linspace(0,1.11e-22,5)
var_num = len(var_array)
var_legend = r'$p_\mathrm{ann}$'
var_figname = 'pann'
#
#############################################
#
# Fixed settings
#
common_settings = {# LambdaCDM parameters
                   'h':0.67556,
                   'omega_b':0.022032,
                   'omega_cdm':0.12038,
                   'A_s':2.215e-9,
                   'n_s':0.9619,
                   'tau_reio':0.0925,
                   # output and precision parameters
                   'output':'tCl,pCl,lCl,mPk',
                   'lensing':'yes',
                   'P_k_max_1/Mpc':3.0,
                   'l_switch_limber':9
                   }
#
# arrays for output
#
kvec = np.logspace(-4,np.log10(3),1000)
legarray = []
twopi = 2.*pi
#
# Create figures
#
fig_Pk, ax_Pk = plt.subplots()
fig_TT, ax_TT = plt.subplots()
fig_EE, ax_EE = plt.subplots()
fig_PP, ax_PP = plt.subplots()
#
M = Class()
#
# loop over varying parameter values
#
for i,var in enumerate(var_array):
    #
    print (' * Compute with %s=%e'%(var_name,var))
    #
    # deal with colors and legends
    #
    if i == 0:
        var_color = 'k'
        var_alpha = 1.
        legarray.append(r'ref. $\Lambda CDM$')
    else:
        var_color = 'r'
        var_alpha = 1.*i/(var_num-1.)
    if i == var_num-1:
        legarray.append(var_legend)  
    #    
    # call CLASS
    #
    M.set(common_settings)
    M.set({var_name:var})
    M.compute()
    #
    # get Cls
    #
    clM = M.lensed_cl(2500)
    ll = clM['ell'][2:]
    clTT = clM['tt'][2:]
    clEE = clM['ee'][2:]
    clPP = clM['pp'][2:]
    #
    # get P(k) for common k values
    #
    pkM = []
    for k in kvec:
        pkM.append(M.pk(k,0.))
    #    
    # plot P(k)
    #
    ax_Pk.loglog(kvec,np.array(pkM),color=var_color,alpha=var_alpha,linestyle='-')
    #
    # plot C_l^TT
    #
    ax_TT.semilogx(ll,clTT*ll*(ll+1)/twopi,color=var_color,alpha=var_alpha,linestyle='-')
    #
    # plot Cl EE 
    #
    ax_EE.loglog(ll,clEE*ll*(ll+1)/twopi,color=var_color,alpha=var_alpha,linestyle='-')
    #
    # plot Cl phiphi
    #
    ax_PP.loglog(ll,clPP*ll*(ll+1)*ll*(ll+1)/twopi,color=var_color,alpha=var_alpha,linestyle='-')
    #
    # reset CLASS
    #
    M.empty()    
#
# output of P(k) figure
#
ax_Pk.set_xlim([1.e-4,3.])
ax_Pk.set_xlabel(r'$k \,\,\,\, [h/\mathrm{Mpc}]$')
ax_Pk.set_ylabel(r'$P(k) \,\,\,\, [\mathrm{Mpc}/h]^3$')
ax_Pk.legend(legarray)
fig_Pk.tight_layout()
fig_Pk.savefig('varying_%s_Pk.pdf' % var_figname)
#
# output of C_l^TT figure
#      
ax_TT.set_xlim([2,2500])
ax_TT.set_xlabel(r'$\ell$')
ax_TT.set_ylabel(r'$[\ell(\ell+1)/2\pi]  C_\ell^\mathrm{TT}$')
ax_TT.legend(legarray)
fig_TT.tight_layout()
fig_TT.savefig('varying_%s_cltt.pdf' % var_figname)
#
# output of C_l^EE figure
#    
ax_EE.set_xlim([2,2500])
ax_EE.set_xlabel(r'$\ell$')
ax_EE.set_ylabel(r'$[\ell(\ell+1)/2\pi]  C_\ell^\mathrm{EE}$')
ax_EE.legend(legarray)
fig_EE.tight_layout()
fig_EE.savefig('varying_%s_clee.pdf' % var_figname)
#
# output of C_l^pp figure
#   
ax_PP.set_xlim([10,2500])
ax_PP.set_xlabel(r'$\ell$')
ax_PP.set_ylabel(r'$[\ell^2(\ell+1)^2/2\pi]  C_\ell^\mathrm{\phi \phi}$')
ax_PP.legend(legarray)
fig_PP.tight_layout()
fig_PP.savefig('varying_%s_clpp.pdf' % var_figname)


```

## warmup.py

```python
#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import classy module
from classy import Class


# In[ ]:


# create instance of the class "Class"
LambdaCDM = Class()
# pass input parameters
LambdaCDM.set({'omega_b':0.0223828,'omega_cdm':0.1201075,'h':0.67810,'A_s':2.100549e-09,'n_s':0.9660499,'tau_reio':0.05430842})
LambdaCDM.set({'output':'tCl,pCl,lCl,mPk','lensing':'yes','P_k_max_1/Mpc':3.0})
# run class
LambdaCDM.compute()


# In[ ]:


# get all C_l output
cls = LambdaCDM.lensed_cl(2500)
# To check the format of cls
cls.keys()


# In[ ]:


ll = cls['ell'][2:]
clTT = cls['tt'][2:]
clEE = cls['ee'][2:]
clPP = cls['pp'][2:]


# In[ ]:


# uncomment to get plots displayed in notebook
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
from math import pi


# In[ ]:


# plot C_l^TT
plt.figure(1)
plt.xscale('log');plt.yscale('linear');plt.xlim(2,2500)
plt.xlabel(r'$\ell$')
plt.ylabel(r'$[\ell(\ell+1)/2\pi]  C_\ell^\mathrm{TT}$')
plt.plot(ll,clTT*ll*(ll+1)/2./pi,'r-')
plt.savefig('warmup_cltt.pdf')


# In[ ]:


# get P(k) at redhsift z=0
import numpy as np
kk = np.logspace(-4,np.log10(3),1000) # k in h/Mpc
Pk = [] # P(k) in (Mpc/h)**3
h = LambdaCDM.h() # get reduced Hubble for conversions to 1/Mpc
for k in kk:
    Pk.append(LambdaCDM.pk(k*h,0.)*h**3) # function .pk(k,z)


# In[ ]:


# plot P(k)
plt.figure(2)
plt.xscale('log');plt.yscale('log');plt.xlim(kk[0],kk[-1])
plt.xlabel(r'$k \,\,\,\, [h/\mathrm{Mpc}]$')
plt.ylabel(r'$P(k) \,\,\,\, [\mathrm{Mpc}/h]^3$')
plt.plot(kk,Pk,'b-')
plt.savefig('warmup_pk.pdf')


```

