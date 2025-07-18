#######################################################################
#                               imports                               #
#######################################################################

import numpy as np
import pickle

from src.memristors import VTEAMMemristor

#######################################################################
#                          optimizer methods                          #
#######################################################################

class SGD():
    def __init__(self, model, beta=0.001):
        self.model = model
        self.beta = beta

    def step(self):
        for layer in self.model.computation_graph:
            if layer.trainable:
                if layer.layer_type == 'dense_layer':
                    update = layer.voltage_drops / self.model.batch_size
                    new_weights = layer.w - layer.lr / self.beta * update
                    layer.w = layer.initializer.clip_conductances(new_weights)


class SGDMomentum():
    def __init__(self, model, beta=0.001, momentum=0.9):
        self.model = model
        self.beta = beta
        self.momentum = momentum
        self.c, self.v = {}, {}
        for layer in self.model.computation_graph:
            if layer.trainable:
                if layer.layer_type == 'dense_layer':
                    self.v[layer.name] = np.zeros(layer.shape)
                elif layer.layer_type in ['diode_layer', 'bias_voltage_layer']:
                    self.c[layer.name] = np.zeros(layer.shape)

    def step(self):
        for layer in self.model.computation_graph:
            if layer.trainable:
                if layer.layer_type == 'dense_layer':
                    update = layer.voltage_drops / self.model.batch_size
                    self.v[layer.name] = self.momentum * self.v[layer.name] + (1 - self.momentum) * update
                    new_weights = layer.w - layer.lr / self.beta * self.v[layer.name]
                    layer.w = layer.initializer.clip_conductances(new_weights)


class Adam():
    def __init__(self, model, beta=0.001, gamma_v=0.5, gamma_s=0.5, eps=1e-8, V_const=1.0, dt_scaling=5e3,
                 modulation_mode="PAM", frequency=None):
        self.model = model
        self.beta = beta
        self.gamma_v = gamma_v
        self.gamma_s = gamma_s
        self.eps = eps
        self.V_const = V_const
        self.dt_scaling = dt_scaling
        self.modulation_mode = modulation_mode
        self.frequency = frequency
        self.k = 0
        self.c, self.v, self.s= {}, {}, {}
        for layer in self.model.computation_graph:
            if layer.trainable:
                if layer.layer_type == 'dense_layer':
                    self.v[layer.name] = np.zeros(layer.shape)
                    self.s[layer.name] = np.zeros(layer.shape)
                elif layer.layer_type in ['diode_layer', 'bias_voltage_layer']:
                    self.c[layer.name] = np.zeros(layer.shape)
              
    def step(self):
        self.k += 1
        for layer in self.model.computation_graph:
            if layer.trainable:
                if layer.layer_type == 'dense_layer':
                    update = layer.voltage_drops / self.model.batch_size

                    self.v[layer.name] = self.gamma_v * self.v[layer.name] + (1 - self.gamma_v) * update
                    self.v[layer.name] = self.v[layer.name] / (1 - self.gamma_v ** self.k)

                    self.s[layer.name] = self.gamma_s * self.s[layer.name] + (1 - self.gamma_s) * (update ** 2)
                    self.s[layer.name] = self.s[layer.name] / (1 - self.gamma_s ** self.k)

                    dt_update = self.dt_scaling * layer.lr * self.v[layer.name] / (np.sqrt(self.s[layer.name]) + self.eps)

                    for i in range(layer.w.shape[0]):
                        for j in range(layer.w.shape[1]):

                            if self.modulation_mode == "PWM":
                                V_applied = -self.V_const if dt_update[i, j] >= 0 else self.V_const
                                dt_pulse = abs(dt_update[i, j]) / (abs(V_applied))

                            elif self.modulation_mode == "PAM":
                                dt_pulse = 1 / self.frequency
                                V_applied = -dt_update[i, j] / dt_pulse
                            
                            if isinstance(layer.w[i, j], VTEAMMemristor):
                                V_applied = -V_applied

                            layer.w[i, j].update_state(V_applied, dt_pulse)

                elif layer.layer_type in ['diode_layer', 'bias_voltage_layer']:
                    layer.bias_voltage -=  1e5 * layer.voltage_drops

    def save_state(self, filepath):
        optimizer_state = {}
        for layer in self.model.computation_graph:
            if layer.layer_type == 'dense_layer':
                optimizer_state[layer.name] = { 'weights' : layer.w,
                                                'velocity' : self.v,
                                                'acceleration' : self.s
                                                }
        with open(filepath, 'wb') as handle:
            pickle.dump(optimizer_state, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_state(self, filepath):
        with open(filepath, 'rb') as handle:
            optimizer_state = pickle.load(handle)

        for layer in self.model.computation_graph:
            if layer.layer_type == 'dense_layer':
                layer.w = optimizer_state[layer.name]['weights']
                layer.v = optimizer_state[layer.name]['velocity']
                layer.s = optimizer_state[layer.name]['acceleration']
