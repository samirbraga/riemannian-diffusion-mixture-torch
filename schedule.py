
from abc import ABC, abstractmethod
import torch
import numpy as np

class BetaSchedule(ABC):
    @abstractmethod
    def beta_t(self, t):
        pass

    @abstractmethod
    def reverse(self):
        pass
        
class LinearBetaSchedule(BetaSchedule):
    def __init__(
        self,
        tf: float = 1,
        t0: float = 0,
        beta_0: float = 0.2,
        beta_f: float = 0.001,
    ):
        self.tf = tf
        self.t0 = t0
        self.beta_0 = beta_0
        self.beta_f = beta_f
        self._beta = beta_f - beta_0
        self._t = tf - t0

    def beta_t(self, t):
        # This robust version prevents broadcasting errors (it was happening before :( )
        return self.beta_0 + (t - self.t0) / self._t * self._beta

    def reverse(self):
        return LinearBetaSchedule(
            tf=self.tf, t0=self.t0, beta_f=self.beta_0, beta_0=self.beta_f
        )