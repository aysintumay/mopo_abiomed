import numpy as np

class StaticFns:
    @staticmethod
    def termination_fn(obs, act, next_obs):
        #no termination policy
        done = np.array([False]*obs.shape[0])
        return done