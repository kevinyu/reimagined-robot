import os
import sys

if os.path.exists(sys.argv[1]):
    sys.path.append(os.path.dirname(os.path.expanduser(sys.argv[1])))
    from opt import *
else:
    from config_template import *

