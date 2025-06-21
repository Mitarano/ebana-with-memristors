import json
from src.datasets import *
from src.initializers import *

class ModelConfig:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        self.dataset_params = self.config.get("dataset", {})
        self.model_params = self.config.get("model", {})

    def initialize_dataset(self):
        dataset_params = self.dataset_params.copy()
        dataset_type = dataset_params.pop("type", None)

        if dataset_type == "XOR":
            dataset = XORDataset(**dataset_params)
        elif dataset_type == "Iris":
            dataset = IrisDataset(**dataset_params)
        elif dataset_type == "BreastCancer":
            dataset = BreastCancerDataset(**dataset_params)
        elif dataset_type == "FullAdder":
            dataset = FullAdderDataset(**dataset_params)
        elif dataset_type == "Digits":
            dataset = DigitsDataset(**dataset_params)
        else:
            raise ValueError(f"Unknown dataset type: {dataset_type}")

        return dataset

    def initialize_model_params(self, dataset):
        model_params = self.model_params
        input_units = dataset.X.shape[-1]
        output_units = dataset.Y.shape[-1]
        hidden_1_units = model_params.get("hidden_1_units", 2)

        bias_p = np.array([model_params.get("bias_p", 1)])
        bias_n = np.array([model_params.get("bias_n", -1)])
        down_diode_bias = np.zeros(shape=(hidden_1_units,))
        up_diode_bias = np.zeros(shape=(hidden_1_units,))

        weight_initializer = Initializers(
            init_type=model_params["weight_initializer"]["init_type"],
            params=model_params["weight_initializer"]["params"]
        )

        diode_params = model_params["diode_params"]
        amp_param = model_params["amp_params"]

        beta = model_params["training"]["beta"]
        lr = model_params["training"]["lr"]
        dt_scaling = model_params["training"]["dt_scaling"]
        epochs = model_params["training"]["epochs"]
        modulation_mode = model_params["training"]["modulation_mode"]
        V_const   = model_params["training"].get("V_const", None)
        frequency = model_params["training"].get("frequency", None)

        load_weights_path = model_params.get("load_weights_path", None)

        return {
            "input_units": input_units,
            "output_units": output_units,
            "hidden_1_units": hidden_1_units,
            "bias_p": bias_p,
            "bias_n": bias_n,
            "down_diode_bias": down_diode_bias,
            "up_diode_bias": up_diode_bias,
            "weight_initializer": weight_initializer,
            "diode_params": diode_params,
            "amp_params": amp_param,
            "beta": beta,
            "lr": lr,
            "dt_scaling": dt_scaling,
            "epochs": epochs,
            "modulation_mode": modulation_mode,
            "V_const": V_const,
            "frequency": frequency,
            "load_weights_path": load_weights_path,
        }