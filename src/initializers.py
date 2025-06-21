#######################################################################
#                               imports                               #
#######################################################################

import numpy as np
from src.memristors import *

#######################################################################
#                    weight initialization methods                    #
#######################################################################

class Initializers:
    def __init__(self, init_type, params):
        self.init_type = init_type
        self.params = params

    def get_bounds(self):
        L = self.params["g_min"]
        U = self.params["g_max"]
        L = self.params.get("L", L)
        U = self.params.get("U", U)
        return L, U

    def clip_conductances(self, w):
        if self.params["g_min"]:
            w = np.clip(w, self.params["g_min"], None)
        if self.params["g_max"]:
            w = np.clip(w, None, self.params["g_max"])
        return w

    def initialize_weights(self, shape):
        if self.init_type == 'random_uniform':
            match self.params["memristor"]:
                case "linear":
                    return self.random_uniform_linear(shape)
                case "linear_ion_drift":
                    return self.random_uniform_linear_ion_drift(shape)
                case "joglekar":
                    return self.random_uniform_joglekar(shape)
                case "biolek":
                    return self.random_uniform_biolek(shape)
                case "vteam":
                    return self.random_uniform_vteam(shape)
                case "yacopcic":
                    return self.random_uniform_yacopcic(shape)
                case "mms":
                    return self.random_uniform_mms(shape)

    def random_uniform_mms(self, shape):
        R_OFF = 1 / self.params["g_min"]
        R_ON = 1 / self.params["g_max"]
        U_OFF = 1 / self.params["U_off"]
        U_ON = 1 / self.params["U_on"]

        L, U = self.get_bounds()
        G_values = np.random.uniform(L, U, size=shape)
        
        memristors = np.empty(shape, dtype=object)
        for i in range(shape[0]):
            for j in range(shape[1]):
                G = G_values[i, j]
                x_init = (G * R_ON * R_OFF - R_ON) / (R_OFF - R_ON)
                memristors[i, j] = MMSMemristor(R_on=R_ON, R_off=R_OFF, U_off=U_OFF, U_on=U_ON, x_init=x_init)

        return memristors


    def random_uniform_yacopcic(self, shape):
        R_ON = 1 / self.params["g_max"]
        R_OFF = 1 / self.params["g_min"]
        b = self.params["b"]
        A_p = self.params["A_p"]
        A_n = self.params["A_n"]

        x_on = R_ON / R_OFF
        a = 1 / (R_ON * np.sinh(b))

        L, U = self.get_bounds()
        G_values = np.random.uniform(L, U, size=shape)
        
        memristors = np.empty(shape, dtype=object)
        for i in range(shape[0]):
            for j in range(shape[1]):
                G = G_values[i, j]
                M = 1.0 / G
                x_init = 1.0 / (M * a * np.sinh(b))
                memristors[i, j] = YacopcicMemristor(x_init=x_init, a1=a, a2=a, b=b, x_on=x_on, A_n=A_n, A_p=A_p)

        return memristors


    def random_uniform_vteam(self, shape):
        R_ON = 1 / self.params["g_max"]
        R_OFF = 1 / self.params["g_min"]
        w_on = self.params["w_on"]
        w_off = self.params["w_off"]

        L, U = self.get_bounds()
        G_values = np.random.uniform(L, U, size=shape)
        
        memristors = np.empty(shape, dtype=object)
        for i in range(shape[0]):
            for j in range(shape[1]):
                G = G_values[i, j]
                M = 1.0 / G
                w_init = w_on + w_off - w_on / np.log(R_OFF / R_ON) * np.log(M / R_ON)
                memristors[i, j] = VTEAMMemristor(R_on=R_ON, R_off=R_OFF, w_init=w_init, w_on=w_on, w_off=w_off)

        return memristors
    

    def random_uniform_biolek(self, shape):
        R_ON = 1 / self.params["g_max"]
        R_OFF = 1 / self.params["g_min"]
        D = self.params["D"]
        p = self.params["p"]

        L, U = self.get_bounds()
        G_values = np.random.uniform(L, U, size=shape)
        
        memristors = np.empty(shape, dtype=object)
        for i in range(shape[0]):
            for j in range(shape[1]):
                G = G_values[i, j]
                M = 1.0 / G 
                w_init = D * (M - R_OFF) / (R_ON - R_OFF)
                memristors[i, j] = BiolekMemristor(R_ON=R_ON, R_OFF=R_OFF, D=D, w_init=w_init, p=p)

        return memristors


    def random_uniform_joglekar(self, shape):
        R_ON = 1 / self.params["g_max"]
        R_OFF = 1 / self.params["g_min"]
        D = self.params["D"]
        p = self.params["p"]

        L, U = self.get_bounds()
        G_values = np.random.uniform(L, U, size=shape)
        
        memristors = np.empty(shape, dtype=object)
        for i in range(shape[0]):
            for j in range(shape[1]):
                G = G_values[i, j]
                M = 1.0 / G 
                w_init = D * (M - R_OFF) / (R_ON - R_OFF)
                memristors[i, j] = JoglekarMemristor(R_ON=R_ON, R_OFF=R_OFF, D=D, w_init=w_init, p=p)

        return memristors
    

    def random_uniform_linear_ion_drift(self, shape):
        R_ON = 1 / self.params["g_max"]
        R_OFF = 1 / self.params["g_min"]
        D = self.params["D"]

        L, U = self.get_bounds()
        G_values = np.random.uniform(L, U, size=shape)
        
        memristors = np.empty(shape, dtype=object)
        for i in range(shape[0]):
            for j in range(shape[1]):
                G = G_values[i, j]
                M = 1.0 / G 
                w_init = D * (M - R_OFF) / (R_ON - R_OFF)
                memristors[i, j] = LinearIonDriftMemristor(R_ON=R_ON, R_OFF=R_OFF, D=D, w_init=w_init)

        return memristors
    

    def random_uniform_linear(self, shape):
        R_ON = 1 / self.params["g_max"]
        R_OFF = 1 / self.params["g_min"]

        L, U = self.get_bounds()
        G_values = np.random.uniform(L, U, size=shape)

        memristors = np.empty(shape, dtype=object)
        for i in range(shape[0]):
            for j in range(shape[1]):
                G = G_values[i, j]
                memristors[i, j] = LinearUpdates(R_ON=R_ON, R_OFF=R_OFF, W_init=G)

        return memristors
