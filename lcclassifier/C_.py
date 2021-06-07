import numpy as np
import lchandler.C_ as C_lchandler

###################################################################################################################################################

EPS = 1e-5

### JOBLIB
import os
JOBLIB_BACKEND = 'threading' # loky multiprocessing threading
N_JOBS = -1 # The number of jobs to use for the computation. If -1 all CPUs are used. If 1 is given, no parallel computing code is used at all, which is useful for debugging. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used. Thus for n_jobs = -2, all CPUs but one are used.
CHUNK_SIZE = os.cpu_count() if N_JOBS<0 else N_JOBS

### SYNTHETIC
OBSE_STD_SCALE = 1/2 # ***

### important
REC_LOSS_EPS = 1
REC_LOSS_K = .2
MSE_K = 1e4
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


### ARCH
DECODER_MLP_LAYERS = 0
DECODER_EMB_K = 0.5 # 0.5 0.25
DECODER_LAYERS = 1 # 1 None
NUM_HEADS = 4 # 4 8

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