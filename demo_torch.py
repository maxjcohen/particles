from matplotlib import pyplot as plt
import numpy as np
import torch.distributions as dists

import particles
from particles import state_space_models as ssm
from particles.collectors import Moments
from particles.foreign_distributions import torch_distribution


class StochVol(ssm.StateSpaceModel):
    @torch_distribution
    def PX0(self):
        return dists.Normal(
            loc=self.mu, scale=self.sigma / np.sqrt(1.0 - self.rho ** 2)
        )

    @torch_distribution
    def PX(self, t, xp):
        return dists.Normal(loc=self.mu + self.rho * (xp - self.mu), scale=self.sigma)

    @torch_distribution
    def PY(self, t, xp, x):
        return dists.Normal(loc=0.0, scale=np.exp(x))


my_model = StochVol(mu=-1.0, rho=0.9, sigma=0.1)
true_states, data = my_model.simulate(100)

plt.style.use("ggplot")
plt.plot(data)

fk_model = ssm.Bootstrap(ssm=my_model, data=data)  # we use the Bootstrap filter
pf = particles.SMC(fk=fk_model, N=100, resampling='stratified',
                   collect=[Moments()], store_history=True)  # the algorithm
pf.run()  # actual computation

# plot
plt.figure()
plt.plot([yt**2 for yt in data], label='data-squared')
plt.plot([m['mean'] for m in pf.summaries.moments], label='filtered volatility')
plt.legend()
plt.show()
