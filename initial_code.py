# %%
# %%
from sklearn import svm
import numpy as np

from auxil.testfunc import Hosaki2d, Branin2d, HARTMANN4D, HARTMANN6D, Dette8d
from auxil.initialization import Initialization
#from core.asopt import MyGPR, ActiveSampling
#from auxil.samplesvm import Sampling_based_SVM
from auxil.awsauxil import send_datatos3
#from plotting.plots import progress_plot, required_sample, plot_svm_boundary, plot_heatmap_uncertainty

# %%
# Benchmark Function Test Main script
case = 'benchmark' # case can be 'benchmark' or 'simulation'
dim = 2 # number of features for benchmark function
sample_method = 'sobol'
# Set condition for feasible region  
# This is only needed for benchmark function (case = 'benchmark')
#condition = lambda x: HARTMANN4D(x) <= -1.0
condition = lambda x: Hosaki2d(x) <= -0.5

# Initial samples
num_samples = 2**4    # number of initial samples
Initializer = Initialization(dim, num_samples, case = case, condition = condition) # Class for initial sampling

# Check data feasibility
# Raise ValueError if only one classification (e.g., only 1 and no -1) is included in the initial sample
# Sampling again if the error is raised
X = Initializer.sample(sample_method)

        # for _X in X:
        #     if self.case == 'benchmark':
        #         y.append(check_class(_X, case = self.case, condition=self.condition))
        #     else:
        #         y.append(check_class(_X, case = self.case))

np.save('X_initial_array.npy', X)



# %%

send_datatos3('X_initial_array.npy', 'testbucketjoonjae', 'initial_input/X_initial_array.npy')


# %%
