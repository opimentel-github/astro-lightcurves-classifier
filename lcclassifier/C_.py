import synthsne.C_ as C_synth

###################################################################################################################################################

EPS = 1e-10

HOURS_NOISE_AMP = C_synth.HOURS_NOISE_AMP
OBSE_STD_SCALE = C_synth.OBSE_STD_SCALE
CPDS_P = C_synth.CPDS_P

DEFAULT_DAYS_N = 80
DEFAULT_MIN_DAY = 1
DEFAULT_FIGSIZE_BOX = (10,10)
DEFAULT_FIGSIZE_REC = (10,3)
EMAIL = 'oscarlo.pimentel@gmail.com'

PLOT_FIGZISE_CM = (7,5)
PLOT_FIGZISE_RECT = (13,7)
FONTSIZE = 14

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