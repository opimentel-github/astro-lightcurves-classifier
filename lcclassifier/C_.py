import numpy as np
import lchandler.C_ as C_lchandler

###################################################################################################################################################

EPS = 1e-10

### JOBLIB
JOBLIB_BACKEND = 'threading' # loky threading
N_JOBS = 6 # The number of jobs to use for the computation. If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all, which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.
CHUNK_SIZE = N_JOBS*1

### SYNTHETIC
OBSE_STD_SCALE = 1/3
CPDS_P = 5./100. # curve points down sampling probability
HOURS_NOISE_AMP = 3.

EFFECTIVE_BETA_EPS = 0.000001 # same weight -> 0.01 0.001 0.0001 0.00001 -> 1/freq
XENTROPY_K = 1e2
MSE_K = 1e-2

DEFAULT_DAYS_N = 50
DEFAULT_MIN_DAY = 2.

### PLOTS
DEFAULT_FIGSIZE_BOX = (10,10)
DEFAULT_FIGSIZE_REC = (10,3)
PLOT_FIGZISE_CM = (7,5)
PLOT_FIGZISE_RECT = (13,7)
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