# Configuration file for the Sphinx documentation builder.
#
# to build:
#
# sphinx-apidoc -t docs/_templates/ -o docs/source .
# rm docs/source/.rst docs/source/tenet.rst docs/source/modules.rst
# sphinx-build -b html docs/ docs/_build/

import sys
import os

def git_version():
    """ Return git revision and date that these docs were made from. """
    from subprocess import Popen, PIPE

    pipe = Popen('git log -n 1', stdout=PIPE, shell=True)
    info = pipe.stdout.read().decode('utf-8').split("\n")

    dateLine = 3 if 'Merge:' in info[1] else 2

    rev = info[0].split(" ")[1][0:7]
    date = info[dateLine].split("Date:")[1].strip()
    date = " ".join(date.split(" ")[:-1])  # chop off timezone detail
    version = rev + " (" + date + ")"

    return version

# -- Project information -----------------------------------------------------

project = 'tenet'
copyright = '2022, Dylan Nelson'
author = 'Dylan Nelson'

version = git_version()
release = git_version()

# -- custom directive to dynamically generate .rst based on custom parsing ---

from os.path import basename
from docutils.parsers.rst import Directive
from docutils import nodes, statemachine
from io import StringIO

class ExecDirective(Directive):
    """Execute the specified python code and insert the output into the document"""
    has_content = True

    def run(self):
        oldStdout, sys.stdout = sys.stdout, StringIO()

        tab_width = self.options.get('tab-width', self.state.document.settings.tab_width)
        source = self.state_machine.input_lines.source(self.lineno - self.state_machine.input_offset - 1)

        try:
            exec('\n'.join(self.content))
            text = sys.stdout.getvalue()
            lines = statemachine.string2lines(text, tab_width, convert_whitespace=True)
            self.state_machine.insert_input(lines, source)
            return []
        except Exception:
            return [nodes.error(None, nodes.paragraph(text = "Unable to execute python code at %s:%d:" % (basename(source), self.lineno)), nodes.paragraph(text = str(sys.exc_info()[1])))]
        finally:
            sys.stdout = oldStdout


# -- General configuration ---------------------------------------------------

extensions = ['sphinx.ext.mathjax',
              'sphinx.ext.autodoc',
              'sphinx.ext.napoleon',
              'sphinx.ext.intersphinx']

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates'] # we use this for apidoc -t templates instead

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

napoleon_google_docstring = True
napoleon_numpy_docstring = False

autodoc_member_order = 'bysource'

add_module_names = False

def setup(app):
    app.add_css_file('style.css')
    app.add_directive('exec', ExecDirective)

# -- Intersphinx -------------------------------------------------------------

intersphinx_mapping = {
    'sphinx': ('https://www.sphinx-doc.org/en/master/', None),
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'h5py': ('https://docs.h5py.org/en/latest/', None),
}

# -- Options for HTML output -------------------------------------------------

html_theme = 'sphinx_rtd_theme'
html_show_sphinx = False
html_show_copyright = False
html_title = 'documentation'
html_logo = '_static/logo_sm.png'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# -- create custom rst files --

def math_latex(label):
    """ Convert $-based latex into rest :math: syntax. """
    label_math = ''
    state = 0

    for s in label:
        if s == '$':
            if state % 2 == 0:
                label_math += " :math:`"
            else:
                label_math += "` "
            state += 1
        else:
            label_math += s

    return label_math.strip()

def quant_rowtext(custom_fields, custom_fields_aliases, key, cat=False):
    """ Write a given quantity out, given its key and the registry dict to use. """
    # get properties
    func = custom_fields[key]
    f_name = key.replace('_','\_') # avoid rest linking
    f_link = ":py:func:`~%s.%s`" % (func.__module__, func.__name__)
    f_desc = func.__doc__.replace('\n','').strip()

    f_aliases = ''
    if key in custom_fields_aliases:
        f_aliases = ', '.join(custom_fields_aliases[key])

    # label
    f_label = getattr(func, 'label', '')

    if callable(f_label):
        # we don't know the actual format of the expected field, try a few
        field = key
        if key in [' flux',' lum']:
            field = '\{line_name\}' + key # prefixed single parameter case (e.g. cloudy_flux_)
        elif not cat:
            field = key + 'X' # single parameter cases

        for i in range(3):
            try:
                f_label = f_label('','',field) # sim,pt are blank
                break # quit on success
            except Exception as e:
                if i == 0:
                    field = key + 'X_Y' # double parameter cases (e.g. ionmassratio_)
                if i == 1:
                    field = '\{ion\} \{num\}' + key # prefixed double parameter cases (e.g. cloudy_mass_)

        f_name = field.replace('_','\_') # include in display name

    f_label = math_latex(f_label.replace('[pt]',''))

    # units
    f_units = getattr(func, 'units', '')

    if callable(f_units):
        f_units = f_units('','',key) # sim,pt are blank

    f_units = math_latex(f_units)
    if f_units == '': f_units = '--' # dimensionless

    # return strings
    return f_name, f_label, f_units, f_aliases, f_desc

with open('quants_custom.rst','w') as f:
    from tenet.load.snapshot import custom_fields, custom_multi_fields
    from tenet.load.snapshot import custom_fields_aliases

    # write header for custom table
    f.write('.. csv-table::\n')
    f.write('    :header: "Field Name", "Label", "Units", "Aliases", "Description"\n')
    f.write('    :widths: 10, 25, 15, 20, 30\n')
    f.write('\n')

    for key in custom_fields_aliases.keys():
        f_name, f_label, f_units, f_aliases, f_desc = quant_rowtext(custom_fields,custom_fields_aliases,key)
        s = '    "%s", "%s", "%s", "%s", "%s"\n' % (f_name,f_label,f_units,f_aliases,f_desc)
        f.write(s)

    # write header for multi table
    f.write('\n\n')
    f.write('The following \'wildcard\' fields match any field request whose name contains the pattern.\n\n')

    f.write('.. csv-table::\n')
    f.write('    :header: "Field Name", "Label", "Units", "Description"\n')
    f.write('    :widths: 10, 25, 15, 50\n')
    f.write('\n')

    for key in custom_multi_fields.keys():
        f_name, f_label, f_units, f_aliases, f_desc = quant_rowtext(custom_fields,custom_fields_aliases,key)
        s = '    "%s", "%s", "%s", "%s"\n' % (f_name,f_label,f_units,f_desc)
        f.write(s)

    f.write('\n')

with open('quants_cat_custom.rst','w') as f:
    from tenet.load.groupcat import custom_cat_fields, custom_cat_multi_fields
    from tenet.load.groupcat import custom_cat_fields_aliases

    # write header for custom (catalog field) table
    f.write('.. csv-table::\n')
    f.write('    :header: "Field Name", "Label", "Units", "Aliases", "Description"\n')
    f.write('    :widths: 10, 25, 15, 20, 30\n')
    f.write('\n')

    for key in custom_cat_fields_aliases.keys():
        f_name, f_label, f_units, f_aliases, f_desc = quant_rowtext(custom_cat_fields,custom_cat_fields_aliases,key,cat=True)
        s = '    "%s", "%s", "%s", "%s", "%s"\n' % (f_name,f_label,f_units,f_aliases,f_desc)
        f.write(s)

    f.write('\n')
