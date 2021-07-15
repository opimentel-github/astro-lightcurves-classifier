import numpy as np
import lchandler.C_ as C_lchandler

###################################################################################################################################################

EPS = 1e-5
ALPHABET = 'abcdefghi'
RANDOM_STATE = 0

### loss
REC_LOSS_EPS = 1
REC_LOSS_K = 1e-3 # 10 1e-3 0
MSE_K = 5000 # 0 1000 5000
XENTROPY_K = 1

MAX_DAY = 100.
DEFAULT_DAYS_N = 100
DEFAULT_DAYS_N_AN = 50 # 5 50 100
DEFAULT_MIN_DAY = 2.

### DICTS
CLASSES_STYLES = C_lchandler.CLASSES_STYLES
COLOR_DICT = C_lchandler.COLOR_DICT