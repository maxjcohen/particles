from matplotlib import pyplot as plt
import numpy as np

import particles
from particles import distributions as dists
from particles import state_space_models as ssm
from particles.collectors import Moments


class StochVol(ssm.StateSpaceModel):
    def PX0(self):
        return dists.Normal(
            loc=self.mu, scale=self.sigma / np.sqrt(1.0 - self.rho ** 2)
        )

    def PX(self, t, xp):
        return dists.Normal(loc=self.mu + self.rho * (xp - self.mu), scale=self.sigma)

    def PY(self, t, xp, x):
        return dists.Normal(loc=0.0, scale=np.exp(x))


my_model = StochVol(mu=-1.0, rho=0.9, sigma=0.1)
true_states, data = my_model.simulate(100)

plt.style.use("ggplot")
plt.plot(data)
plt.show()
