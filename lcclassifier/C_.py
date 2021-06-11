import numpy as np
import lchandler.C_ as C_lchandler

###################################################################################################################################################

EPS = 1e-5

### LOSS
REC_LOSS_EPS = 1
REC_LOSS_K = 0
MSE_K = 1e4 # 0 1e3 1e4
XENTROPY_K = 1

MAX_DAY = 100.
DEFAULT_DAYS_N = 100
DEFAULT_DAYS_N_AN = 50 # 5 50 100
DEFAULT_MIN_DAY = 2.

### PLOTS
P_PLOT = 10
DEFAULT_FIGSIZE_BOX = (10,10)
DEFAULT_FIGSIZE_REC = (10,3)
PLOT_FIGZISE_CM = (7,5)
PLOT_FIGZISE_RECT = (15,7)
FONTSIZE = 14

'''
FNAME_REPLACE_DICT = { # formating
	'mdl:':'',
	#'mdl':'model',
	'inD':'$N_{in}$',
	'teD':'$N_{TE}$',
	'rnnL:':'$L_{RNN}$:',
	'rnnU:':'$U_{RNN}$:',
}

METRIC_FEXT_DICT = {
	'mse':'exprec',
	'ase':'exprec',
	'*f1score*':'expmet',
}
'''