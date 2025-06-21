import numpy as np
from sklearn.datasets import load_iris, load_digits, load_breast_cancer, fetch_california_housing
from sklearn import preprocessing
from sklearn.preprocessing import OneHotEncoder
from src.batchgenerator import BatchGenerator
from src.utils import generate_voltage_values
import abc

class BaseDataset(abc.ABC):
    def __init__(self, scale=0.5, split_size=0.7, shift=0, output_shift=0, output_midpoint=0.5, batch_size=1, shuffle=False):
        self.scale = scale
        self.split_size = split_size
        self.shift = shift
        self.output_shift = output_shift
        self.output_midpoint = output_midpoint
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.X, self.Y = self._load_data()
        self.X_train, self.Y_train, self.X_test, self.Y_test = self._split_data(self.X, self.Y)
        self.train_dataset = self._construct_dataset(self.X_train, self.Y_train)
        self.test_dataset = self._construct_dataset(self.X_test, self.Y_test)
        self._create_dataloaders()

    @abc.abstractmethod
    def _load_data(self):
        pass

    def _split_data(self, X, Y):
        if self.split_size == 0:
            return X, Y, X, Y
        
        p = np.random.permutation(len(Y))
        X_shuffled, Y_shuffled = X[p], Y[p]

        n_train = int(self.split_size * len(Y_shuffled))

        X_train, X_test = X_shuffled[:n_train], X_shuffled[n_train:]
        Y_train, Y_test = Y_shuffled[:n_train], Y_shuffled[n_train:]

        return X_train, Y_train, X_test, Y_test
    
    def _construct_dataset(self, X, Y):
        return {
            'inputs': {'xp': X + self.shift, 'xn': -X + self.shift},
            'outputs': {'out': Y + self.output_shift}
        }
    
    def _create_dataloaders(self):
        self.train_dataloader = BatchGenerator(self.train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self.test_dataloader = BatchGenerator(self.test_dataset, batch_size=len(self.test_dataset), shuffle=self.shuffle)


class XORDataset(BaseDataset):
    def __init__(self, scale=0.5, split_size=0, shift=0, output_shift=-0.5, output_midpoint=0.0, batch_size=4, shuffle=False):
        super().__init__(scale=scale, split_size=split_size, shift=shift, output_shift=output_shift, output_midpoint=output_midpoint, batch_size=batch_size, shuffle=shuffle)

    def _load_data(self):
        X = np.array(generate_voltage_values(values=[-self.scale, self.scale], count=2))
        Y = np.array([[0],[1],[1],[0]])
        return X, Y


class FullAdderDataset(BaseDataset):
    def __init__(self, scale=0.5, split_size=0, shift=0, output_shift=-0.5, output_midpoint=0.0, batch_size=8, shuffle=False):
        super().__init__(scale=scale, split_size=split_size, shift=shift, output_shift=output_shift, output_midpoint=output_midpoint, batch_size=batch_size, shuffle=shuffle)

    def _load_data(self):
        X = np.array(generate_voltage_values(values=[-self.scale, self.scale], count=3))
        result = np.array([[0,0],[1,0],[1,0],[0,1],[1,0],[0,1],[0,1],[1,1]])
        return X, result


class IrisDataset(BaseDataset):
    def __init__(self, scale=0.5, split_size=0.7, shift=0, output_shift=0, output_midpoint=0.5, batch_size=105, shuffle=True):
        super().__init__(scale=scale, split_size=split_size, shift=shift, output_shift=output_shift, output_midpoint=output_midpoint, batch_size=batch_size, shuffle=shuffle)

    def _load_data(self):
        iris = load_iris()
        X = iris["data"]
        Y = iris["target"]

        X = np.round(self.scale * preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(X), 2)
        Y = OneHotEncoder(sparse_output=False).fit_transform(Y.reshape(-1, 1))

        return X, Y


class BreastCancerDataset(BaseDataset):
    def __init__(self, scale=0.5, split_size=0.7, shift=0, output_shift=0, output_midpoint=0.5, batch_size=398, shuffle=True):
        super().__init__(scale=scale, split_size=split_size, shift=shift, output_shift=output_shift, output_midpoint=output_midpoint, batch_size=batch_size, shuffle=shuffle)

    def _load_data(self):
        breast_cancer = load_breast_cancer()
        X = breast_cancer["data"]
        Y = breast_cancer["target"]

        X = np.round(self.scale * preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(X), 2)
        Y = OneHotEncoder(sparse_output=False).fit_transform(Y.reshape(-1, 1))

        return X, Y


class CaliforniaHousingDataset(BaseDataset):
    def __init__(self, scale=0.5, split_size=0.7, shift=0, output_shift=0, output_midpoint=0.5, batch_size=105, shuffle=True):
        super().__init__(scale=scale, split_size=split_size, shift=shift, output_shift=output_shift, output_midpoint=output_midpoint, batch_size=batch_size, shuffle=shuffle)

    def _load_data(self):
        california_housing = fetch_california_housing()
        X = california_housing["data"]
        Y = california_housing["target"]

        X = np.round(self.scale * preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(X), 2)
        Y = OneHotEncoder(sparse_output=False).fit_transform(Y.reshape(-1, 1))

        return X, Y


class DigitsDataset(BaseDataset):
    def __init__(self, scale=0.5, split_size=0.779, shift=0, output_shift=-0.5, output_midpoint=0.0, batch_size=200, shuffle=True):
        super().__init__(scale=scale, split_size=split_size, shift=shift, output_shift=output_shift, output_midpoint=output_midpoint, batch_size=batch_size, shuffle=shuffle)

    def _load_data(self):
        digits = load_digits()
        X = digits["data"]
        Y = digits["target"]

        X = np.round(self.scale * preprocessing.MinMaxScaler(feature_range=(-1, 1)).fit_transform(X), 3)
        Y = OneHotEncoder(sparse_output=False).fit_transform(np.array(Y).reshape(-1, 1))

        return X, Y