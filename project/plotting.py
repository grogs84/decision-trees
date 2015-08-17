# Copyright 2013 Philip N. Klein
"""
This file contains a simple plotting interface, which uses a browser with SVG to 
present a plot of points represented as either complex numbers or 2-vectors.

"""

import webbrowser
from numbers import Number

import tempfile
import os
import atexit

_browser = None

def plot(L, scale=4, dot_size = 3, browser=None):
    """ plot takes a list of points, optionally a scale (relative to a 200x200 frame),
        optionally a dot size (diameter) in pixels, and optionally a browser name.
        It produces an html file with SVG representing the given plot,
        and opens the file in a web browser. It returns nothing.
    """
    scalar = 200./scale
    origin = (210, 210)
    hpath = create_temp('.html')
    with open(hpath, 'w') as h:
        h.writelines(
            ['<!DOCTYPE html>\n'
            ,'<head>\n'
            ,'<title>plot</title>\n'
            ,'</head>\n'
            ,'<body>\n'
            ,'<svg height="420" width=420 xmlns="http://www.w3.org/2000/svg">\n'
            ,'<line x1="0" y1="210" x2="420" y2="210"'
            ,'style="stroke:rgb(0,0,0);stroke-width:2"/>\n'
            ,'<line x1="210" y1="0" x2="210" y2="420"'
            ,'style="stroke:rgb(0,0,0);stroke-width:2"/>\n'])
        for pt in L:
            if isinstance(pt, Number):
                x,y = pt.real, pt.imag
            else:
                if isinstance(pt, tuple) or isinstance(pt, list):
                    x,y = pt
                else:
                    raise ValueError
            h.writelines(['<circle cx="%d" cy="%d" r="%d" fill="red"/>\n'
                          % (origin[0]+scalar*x,origin[1]-scalar*y,dot_size)])
        h.writelines(['</svg>\n</body>\n</html>'])
    if browser is None:
        browser = _browser
    webbrowser.get(browser).open('file://%s' % hpath)

def plot_line(L, scale=10, dot_size = 3, browser=None):
    """ plot takes a list of points, optionally a scale (relative to a 200x200 frame),
        optionally a dot size (diameter) in pixels, and optionally a browser name.
        It produces an html file with SVG representing the given plot,
        and opens the file in a web browser. It returns nothing.
    """
    scalar = 200./scale
    # origin = (210, 210)
    origin = 210
    hpath = create_temp('.html')
    with open(hpath, 'w') as h:
        h.writelines(
            ['<!DOCTYPE html>\n'
            ,'<head>\n'
            ,'<title>plot</title>\n'
            ,'</head>\n'
            ,'<body>\n'
            ,'<svg height="420" width=420 xmlns="http://www.w3.org/2000/svg">\n'
            ,'<line x1="0" y1="210" x2="420" y2="210"'
            ,'style="stroke:rgb(0,0,0);stroke-width:2"/>\n'
            ,'<line x1="210" y1="0" x2="210" y2="420"'
            ,'style="stroke:rgb(0,0,0);stroke-width:2"/>\n'])
        for line in L:
            pt1, pt2 = line
            x1, y1 = pt1
            x2, y2 = pt2
            h.writelines(['<line x1="%d" y1="%d" x2="%d" y2="%d" style="stroke:rgb(0,0,0);stroke-width:2"/>\n'
                          % (origin+x1*scalar,origin-y1*scalar,origin+x2*scalar,origin-y2*scalar)])
        h.writelines(['</svg>\n</body>\n</html>'])
    if browser is None:
        browser = _browser
    webbrowser.get(browser).open('file://%s' % hpath)

def setbrowser(browser=None):
    """ Registers the given browser and saves it as the module default.
        This is used to control which browser is used to display the plot.
        The argument should be a value that can be passed to webbrowser.get()
        to obtain a browser.  If no argument is given, the default is reset
        to the system default.

        webbrowser provides some predefined browser names, including:
        'firefox'
        'opera'

        If the browser string contains '%s', it is interpreted as a literal
        browser command line.  The URL will be substituted for '%s' in the command.
        For example:
        'google-chrome %s'
        'cmd "start iexplore.exe %s"'

        See the webbrowser documentation for more detailed information.

        Note: Safari does not reliably work with the webbrowser module,
        so we recommend using a different browser.
    """
    global _browser
    if browser is None:
        _browser = None  # Use system default
    else:
        webbrowser.register(browser, None, webbrowser.get(browser))
        _browser = browser

def getbrowser():
    """ Returns the module's default browser """
    return _browser

# Create a temporary file that will be removed at exit
# Returns a path to the file
def create_temp(suffix='', prefix='tmp', dir=None):
    _f, path = tempfile.mkstemp(suffix, prefix, dir)
    os.close(_f)
    remove_at_exit(path)
    return path

# Register a file to be removed at exit
def remove_at_exit(path):
    atexit.register(os.remove, path)




class Graph:
    def __init__(self):
        self.points = {}
        self.lines = {}

    def add_point(self, pt, name=""):
        if name == "":
            name = "pt" + str(len(self.points))
        self.points[name] = pt

    def add_line(self, pts, name=""):
        if name == "":
            name = "line" + str(len(self.lines))
        self.lines[name] = pts

    def show(self, browser=None):
        scalar = 600./10
        # origin = (210, 210)
        dot_size = 3
        origin = 210
        hpath = create_temp('.html')
        with open(hpath, 'w') as h:
            h.writelines(
                ['<!DOCTYPE html>\n'
                ,'<head>\n'
                ,'<title>plot</title>\n'
                ,'</head>\n'
                ,'<body>\n'
                ,'<svg height="420" width=420 xmlns="http://www.w3.org/2000/svg">\n'
                ,'<line x1="0" y1="210" x2="420" y2="210"'
                ,'style="stroke:rgb(0,0,0);stroke-width:2"/>\n'
                ,'<line x1="210" y1="0" x2="210" y2="420"'
                ,'style="stroke:rgb(0,0,0);stroke-width:2"/>\n'])
            for line in self.lines:
                pt1, pt2 = self.lines[line]
                x1, y1 = pt1
                x2, y2 = pt2
                h.writelines(['<line x1="%d" y1="%d" x2="%d" y2="%d" style="stroke:rgb(0,0,0);stroke-width:2"/>\n'
                              % (origin+x1*scalar,origin-y1*scalar,origin+x2*scalar,origin-y2*scalar)])
            for pts in self.points:
                pt = self.points[pts]
                x,y = pt
                h.writelines(['<circle cx="%d" cy="%d" r="%d" fill="red"/>\n'
                            % (origin+scalar*x,origin-scalar*y,dot_size)])
            h.writelines(['</svg>\n</body>\n</html>'])
        if browser is None:
            browser = _browser
        webbrowser.get(browser).open('file://%s' % hpath)



