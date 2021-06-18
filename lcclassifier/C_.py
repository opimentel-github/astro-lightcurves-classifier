import numpy as np
import lchandler.C_ as C_lchandler

###################################################################################################################################################

EPS = 1e-5

### loss
REC_LOSS_EPS = 1
REC_LOSS_K = 1e-3 # 10 1e-3 0
MSE_K = 5000 # 0 1000 5000
XENTROPY_K = 1

MAX_DAY = 100.
DEFAULT_DAYS_N = 100
DEFAULT_DAYS_N_AN = 50 # 5 50 100
DEFAULT_MIN_DAY = 2.

### plots
DEFAULT_FIGSIZE_BOX = (10,10)
DEFAULT_FIGSIZE_REC = (10,3)
PLOT_FIGZISE_CM = (7,5)
PLOT_FIGZISE_RECT = (15,7)
FONTSIZE = 14
