# Configuration file for the Sphinx documentation builder.
#
# to build:
#
# sphinx-apidoc -t docs/_templates/ -o docs/source .
# rm docs/source/.rst docs/source/test.rst
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
copyright = '2021, Dylan Nelson'
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
