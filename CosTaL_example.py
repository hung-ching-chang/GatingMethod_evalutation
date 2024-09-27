import pandas as pd
import numpy as np
from collections import Counter

# fresh negative, layer2
dt = pd.read_csv('data.csv', sep=',', header = 0)

import CosTaL as ct
clusters = ct.clustering(dt, method =  'tani', pp_method = 'mass', nbr_num = 30)

