r"""
.. module:: BBN

:Synopsis: BBB class 
:Author: David Camarena
"""

from cobaya.likelihood import Likelihood
from cobaya.typing import InputDict, Sequence
import numpy as np

class BBN(Likelihood):
    # Data type for aggregated chi2 (case sensitive)
    type = "BBN"

    # variables from yaml
    means: Sequence | np.ndarray
    covariance: Sequence | np.ndarray
    quantities: list

    def initialize(self):
        self.means = np.array(self.means)
        self.cov = np.array(self.covariance)
        self.inv_cov = np.linalg.inv(self.cov)

        if len(self.means) != len(self.quantities):
            raise ValueError("Length of means must match quantities")

        if self.cov.shape != (len(self.quantities), len(self.quantities)):
            raise ValueError("Covariance shape inconsistent with quantities")

    def get_requirements(self):
        return {q: None for q in self.quantities}

    def logp(self, **params_values):
        # for q in self.quantities:
            # print(q,self.provider.get_param(q))
        x = np.array([self.provider.get_param(q) for q in self.quantities])
        diff = x - self.means
        chi2 = diff @ self.inv_cov @ diff
        return -0.5 * chi2