#https://github.com/tartley/colorama/blob/master/demos/fixpath.py
import sys
from os.path import normpath, dirname, join
sys.path.insert(0, normpath(join(dirname(__file__), '..')))