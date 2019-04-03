import os
import numpy as np
import sys
#run slam.py from here
#store video var here 
vid = sys.argv[1]
os.system('F=25 ./slam.py videos/'+ str(vid))
 
