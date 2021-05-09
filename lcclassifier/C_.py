import numpy as np
import lchandler.C_ as C_lchandler

###################################################################################################################################################

EPS = 1e-5

### JOBLIB
import os
JOBLIB_BACKEND = 'loky' # loky multiprocessing threading
N_JOBS = -1 # The number of jobs to use for the computation. If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all, which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.
CHUNK_SIZE = os.cpu_count() if N_JOBS<0 else N_JOBS

### SYNTHETIC
OBSE_STD_SCALE = 1/2 # ***
CPDS_P = 10./100. # ***
HOURS_NOISE_AMP = 8. # ***

### important
REC_LOSS_EPS = .1 # *** 0.01 is not working as good
XENTROPY_K = 1
MSE_K = 5e2

MAX_DAY = 100.
DEFAULT_DAYS_N = 100
DEFAULT_DAYS_N_AN = 50
DEFAULT_MIN_DAY = 2.

### PLOTS
P_PLOT = 10
DEFAULT_FIGSIZE_BOX = (10,10)
DEFAULT_FIGSIZE_REC = (10,3)
PLOT_FIGZISE_CM = (7,5)
PLOT_FIGZISE_RECT = (15,7)
FONTSIZE = 14


### ARCH
DECODER_MLP_LAYERS = 1
NUM_HEADS = 6 # 4

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