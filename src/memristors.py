import numpy as np

class MMSMemristor:
    def __init__(self, R_on=500, R_off=1500, U_on=0.27, U_off=0.27, tau=1e-4, T=298.5, x_init=0):
        self.R_on = R_on
        self.R_off = R_off
        self.U_on = U_on
        self.U_off = U_off
        self.tau = tau
        self.T = T
        self.q = 1.602176634e-19
        self.k = 1.380649e-23
        self.x = np.clip(x_init, 0, 1)
        self._update_memristance()

    def _update_memristance(self):
        self.W = self.x / self.R_on + (1.0 - self.x) / self.R_off
        self.M = 1.0 / self.W

    def update_state(self, u, dt):
        alpha = dt / self.tau
        beta = self.q / (self.k * self.T)        

        P_on = alpha / (1 + np.exp(-beta * (u - self.U_on)))
        P_off = alpha * (1 - 1 / (1 + np.exp(-beta * (u + self.U_off))))
        
        N_on = P_on * (1 - self.x)
        N_off = P_off * self.x
        
        self.x += N_on - N_off
        self.x = np.clip(self.x, 0, 1)

        self._update_memristance()

    def get_current(self, u):
        return u * self.W


class YacopcicMemristor:
    def __init__(self, A_p=4000, A_n=4000, U_p=0.5, U_n=0.5, alpha_p=1, alpha_n=5, x_p=0.3, x_n=0.3, a1=0.17, a2=0.17, b=0.05, x_init=0.11, x_on=0):
        self.A_p = A_p
        self.A_n = A_n
        self.U_p = U_p
        self.U_n = U_n
        self.alpha_p = alpha_p
        self.alpha_n = alpha_n
        self.x_p = x_p
        self.x_n = x_n
        self.a1 = a1
        self.a2 = a2
        self.b = b
        self.x_on = x_on
        self.x = np.clip(x_init, self.x_on, 1)
        self._update_memristance()

    def g(self, u):
        if u > self.U_p:
            return self.A_p * (np.exp(u) - np.exp(self.U_p))
        elif u < -self.U_n:
            return -self.A_n * (np.exp(-u) - np.exp(self.U_n))
        else:
            return 0

    def f_p(self, x):
        if x >= self.x_p:
            wp = (self.x_p - x) / (1 - self.x_p) + 1
            return np.exp(-self.alpha_p * (x - self.x_p)) * wp
        else:
            return 1

    #def f_n(self, x):
    #    if x <= (1 - self.x_n):
    #        wn = x / (1 - self.x_n)
    #        return np.exp(self.alpha_n * (x + self.x_n - 1)) * wn
    #    else:
    #        return 1

    def f_n(self, x):
        if x <= (1 - self.x_n):
            return np.exp(self.alpha_n * (x + self.x_n - 1)) * (x - self.x_on) / (self.x_n - self.x_on)
        else:
            return 1
    
    def f(self, x, u):
        return self.f_p(x) if (u >= 0) else self.f_n(x)
    
    def _update_memristance(self):
        self.W = self.a1 * self.x * np.sinh(self.b)
        self.M = 1 / self.W

    def update_state(self, u, dt):
        dx = self.g(u) * self.f(self.x, u)
        self.x += dx * dt
        self.x = np.clip(self.x, self.x_on, 1)
        self._update_memristance()

    def get_current(self, u):
        if u >= 0:
            return self.a1 * self.x * np.sinh(self.b * u)
        else:
            return self.a2 * self.x * np.sinh(self.b * u)


class VTEAMMemristor:
    def __init__(self, k_off=5e-4, k_on=-10, alpha_off=3, alpha_on=1, w_off=3e-9, w_on=0, w_init=0, a_off=0.8, a_on=0.2, w_c=0.12, u_off=0.5, u_on=-0.5, R_on=100, R_off=2.5e3):
        self.k_off = k_off
        self.k_on = k_on
        self.alpha_off = alpha_off
        self.alpha_on = alpha_on
        self.w_on = w_on
        self.w_off = w_off
        self.w = np.clip(w_init, self.w_on, self.w_off)
        self.a_off = a_off
        self.a_on = a_on
        self.w_c = w_c
        self.u_off = u_off
        self.u_on = u_on
        self.R_on = R_on
        self.R_off = R_off
        self._update_memristance()

    def f_off(self, w):
        return np.exp(-np.exp((w - self.a_off) / self.w_c))

    def f_on(self, w):
        return np.exp(-np.exp(-(w - self.a_on) / self.w_c))
    
    def _update_memristance(self):
        _lambda = np.log(self.R_off / self.R_on)
        self.M = self.R_on * np.exp((_lambda / (self.w_off - self.w_on)) * (self.w - self.w_on))
        self.W = 1 / self.M
    
    def update_state(self, u, dt):
        dw_dt = 0
        
        if (0 < self.u_off < u):
            dw_dt = self.k_off * (u / self.u_off - 1.0)**self.alpha_off * self.f_off(self.w)
        elif (u < self.u_on < 0):
            dw_dt = self.k_on * (u / self.u_on - 1.0)**self.alpha_on * self.f_on(self.w)

        self.w += dw_dt * dt
        self.w = np.clip(self.w, self.w_on, self.w_off)
        self._update_memristance()

    def get_current(self, u):
        _lambda = np.log(self.R_off / self.R_on)
        exponent_term = (-_lambda / (self.w_off - self.w_on)) * (self.w - self.w_on)
        return u / self.R_on * np.exp(exponent_term)


class BiolekMemristor:
    def __init__(self, mu_v=1e-9, D=1e-8, R_ON=100, R_OFF=16e3, w_init=None, p=1):
        self.mu_v = mu_v
        self.D = D
        self.R_ON = R_ON
        self.R_OFF = R_OFF
        self.w = np.clip(w_init, 0, self.D) if w_init is not None else D / 2
        self.p = p
        self._update_memristance()

    def _update_memristance(self):
        self.M = self.R_ON * (self.w / self.D) + self.R_OFF * (1 - (self.w / self.D))
        self.W = 1.0 / self.M

    @staticmethod
    def step_function(i):
        return 1.0 if i >= 0 else 0.0

    def window_function(self, x, i):
        return 1.0 - (x - self.step_function(-i))**(2 * self.p)

    def update_state(self, V, dt):
        i = self.get_current(V)
        f_w = self.window_function(self.w / self.D, i)
        dw_dt = self.mu_v * (self.R_ON / self.D) * i * f_w
        self.w += dw_dt * dt
        self.w = np.clip(self.w, 0, self.D)
        self._update_memristance()

    def get_current(self, V):
        return V / self.M


class JoglekarMemristor:
    def __init__(self, mu_v=1e-9, D=1e-8, R_ON=100, R_OFF=16e3, w_init=None, p=1):
        super().__init__()
        self.mu_v = mu_v
        self.D = D
        self.R_ON = R_ON
        self.R_OFF = R_OFF
        self.w = np.clip(w_init, 0, self.D) if w_init is not None else D / 2
        self.p = p
        self._update_memristance()

    def _update_memristance(self):
        self.M = self.R_ON * (self.w / self.D) + self.R_OFF * (1 - (self.w / self.D))
        self.W = 1.0 / self.M

    def window_function(self, x):
        return 1 - (2 * x - 1) ** (2 * self.p)

    def update_state(self, V, dt):
        i = self.get_current(V)
        f_w = self.window_function(self.w / self.D)
        dw_dt = self.mu_v * (self.R_ON / self.D) * i * f_w
        self.w += dw_dt * dt
        self.w = np.clip(self.w, 0, self.D)
        self._update_memristance()

    def get_current(self, V):
        return V / self.M


class LinearIonDriftMemristor:
    def __init__(self, mu_v=1e-9, D=1e-8, R_ON=100, R_OFF=16e3, w_init=None):
        super().__init__()
        self.mu_v = mu_v
        self.D = D
        self.R_ON = R_ON
        self.R_OFF = R_OFF
        self.w = np.clip(w_init, 0, self.D) if w_init is not None else D / 2
        self._update_memristance()

    def _update_memristance(self):
        self.M = self.R_ON * (self.w / self.D) + self.R_OFF * (1 - (self.w / self.D))
        self.W = 1.0 / self.M

    def update_state(self, V, dt):
        i = self.get_current(V)
        dw_dt = (self.mu_v * self.R_ON / self.D) * i
        self.w += dw_dt * dt
        self.w = np.clip(self.w, 0, self.D)
        self._update_memristance()

    def get_current(self, V):
        return V / self.M


class LinearUpdates:
    def __init__(self, R_ON=100, R_OFF=16000, W_init=1e3):
        self.G_ON = 1.0 / R_ON
        self.G_OFF = 1.0 / R_OFF
        self.W = W_init
        self._update_memristance(0, 0)

    def _update_memristance(self, V, dt):
        self.W += V * dt
        self.W = np.clip(self.W, self.G_OFF, self.G_ON)
        self.M = 1.0 / self.W

    def update_state(self, V, dt):
        self._update_memristance(V, dt)

    def get_current(self, V):
        return V / self.M
