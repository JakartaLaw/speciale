import numpy as np
import pickle
import pandas as pd

from environments.interface import InterfaceEnvironment

# Necessary Data
with open("..//data//fertility.pkl", "rb") as f:
    fertility_data = pickle.load(f)

with open("..//data//men_wage_path.pkl", "rb") as f:
    men_wage_path_data = pickle.load(f)

with open("..//data//men_hours_empirical", "rb") as f:
    men_hours_data = pickle.load(f)

men_salary_path = np.array(men_hours_data * men_wage_path_data * 46)

# calculating scales
Q_mean, Q_scale = (60 + 18) * 0.5, (60 - 18) * 0.5
K_mean, K_scale = (0 + 5) * 0.5, (5 - 0) * 0.5
G_mean, G_scale = (0 + 5) * 0.5, (5 - 0) * 0.5
Z_mean, Z_scale = (-200 + 200), (200 - (-200)) * 0.5

beta_K_mean, beta_K_scale = (-5 + 5) * 0.5, (5 - (-5)) * 0.5
beta_L_mean, beta_L_scale = (-5 + 5) * 0.5, (5 - (-5)) * 0.5


def scale_states(Q, G, K, Z, beta_K, beta_L):
    Q = (Q - Q_mean) / Q_scale
    G = (G - G_mean) / G_scale
    K = (K - K_mean) / K_scale
    Z = (Z - Z_mean) / Z_scale

    beta_K = (beta_K - beta_K_mean) / beta_K_scale
    beta_L = (beta_L - beta_L_mean) / beta_L_scale
    return np.array([Q, G, K, Z, beta_K, beta_L])


ACTIONS = [0, 25, 37, 45]


def translate_action_model1(a):
    return ACTIONS[a]


# values for reward scaler is found by tuning parameters so r_scaled in [-1, 1]
def reward_scaler_model1(r, beta_K, beta_L):
    return (r - 13.5 - (beta_K * 0.9) - (beta_L * 8.64)) / 3


class EnvironmentModel1(InterfaceEnvironment):

    """
    Ordering of items
    states: Q, M, K, W
    shocks: epsilon, rho, psi
    """

    DEFAULT_Q = 18
    DEFAULT_G = 2.0
    DEFAULT_K = 0
    DEFAULT_Z = 0.0

    def __init__(
        self, sigma_epsilon, S_min, eta_G, eta_G_sq, alpha, delta, beta_K, beta_L
    ):

        # parameters
        self.sigma_epsilon = sigma_epsilon
        self.S_min = S_min
        self.eta_G = eta_G
        self.eta_G_sq = eta_G_sq
        self.alpha = alpha
        self.delta = delta

        # The parameters that need to be tuned!
        self.beta_K = beta_K
        self.beta_L = beta_L

        # states
        self.Q = self.DEFAULT_Q
        self.G = self.DEFAULT_G
        self.K = self.DEFAULT_K
        self.Z = self.DEFAULT_Z

    def __repr__(self):
        return f"(Q: {self.Q}, G: {self.G}, K: {self.K}, Z: {self.Z})"

    @property
    def states(self):
        return scale_states(self.Q, self.G, self.K, self.Z, self.beta_K, self.beta_L)

    def reset(self, states=None, parameters=None):
        """Expect states given as: (Q, G, K, Z) """
        if states is not None:
            Q, G, K, Z = states[0], states[1], states[2], states[3]
            self.Q = Q
            self.G = G
            self.K = K
            self.Z = Z
        else:
            self.Q = self.DEFAULT_Q
            self.G = self.DEFAULT_G
            self.K = self.DEFAULT_K
            self.Z = self.DEFAULT_Z

        if parameters is not None:
            for key, val in parameters.items():
                setattr(self, key, val)

        return self.states

    def step(self, action, shocks=None, parameters=None):
        """
        shocks:
            (epsilon, psi) <- that order
        """
        if shocks is None:
            shocks = self.draw_shocks()
        epsilon, psi = shocks

        if parameters is not None:
            for key, val in parameters.items():
                setattr(self, key, val)
        # remember action: hours (H)

        ### transition
        self.calc_Q()
        self.calc_Z(epsilon)
        self.calc_K(psi)

        ### model dynamic
        L = self.calc_L(action)

        # wage/salary process
        log_S_tilde = self.calc_log_S_tilde()
        S = self.calc_S(log_S_tilde)
        W = self.calc_W(S, action)

        # husband income
        M = self.calc_M()

        # household income
        Y = self.calc_Y(W, M)

        utility = self.calc_U(L, Y)

        self.calc_G(action)

        # this might be changed
        done = self.calc_stops()

        _info = f"Y: {Y}, L: {L}, W: {W}, S: {S}, M: {M}"
        if done is True:
            return self.states, utility, True, _info

        return self.states, utility, False, _info

    # model dynamic
    def calc_log_S_tilde(self):
        return self.alpha + self.eta_G * self.G + self.eta_G_sq * self.G ** 2

    def calc_U(self, L, Y):
        u = (
            self.beta_K * np.log(self.K + 1)
            + self.beta_L * np.log(L + 1)
            + np.log(Y + 1)
        )
        if np.isnan(u):
            raise Exception(f"K: {self.K}, L: {L}, Y: {Y}")
        return u

    def calc_W(self, S, H):
        return S * H * 46

    def calc_M(self):
        # use data (non parametric)
        # return 450000
        try:
            return men_salary_path[self.Q]
        except:
            return men_salary_path[int(self.Q)]

    def calc_Y(self, W, M):
        return W + M

    def calc_L(self, hours):
        return 46 * (7 * 24 - hours)

    def calc_stops(self):
        # stops the model (returns done flag)
        if self.Q > 60:
            return True
        return False

    def calc_K(self, psi):
        if self.K < 5:
            self.K = self.K + psi

    def calc_S(self, log_S_tilde):
        _S = np.exp(log_S_tilde) + self.Z
        return max(self.S_min, _S)
        # return 190

    def calc_Q(self):
        self.Q = self.Q + 1

    def calc_G(self, H):
        self.G = self.G * (1 - self.delta) + H / 37

    def calc_Z(self, epsilon):
        self.Z = self.Z + epsilon

    # def shocks
    def draw_shocks(self):
        return (self.draw_epsilon(), self.draw_psi())

    def draw_epsilon(self):
        return np.random.normal(0, self.sigma_epsilon)

    def get_p_psi(self):

        try:
            return fertility_data[self.Q]
        except:
            return fertility_data[int(self.Q)]

    def draw_psi(self):
        p_psi = self.get_p_psi()
        return np.random.binomial(1, p_psi)

    # helpers
    @property
    def observation_space(self):
        return self.states

    @property
    def action_space(self):
        return 4
