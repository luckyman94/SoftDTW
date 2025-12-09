import numpy as np
import os
from config import *

class Dataset:
    """
    Utility class for loading and handling time series from the UCR Archive.

    Parameters
    name : str (default="ECG200")
        Name of the UCR dataset to load. 
        
    Attributes

    X_train, y_train : np.ndarray
        Training time series and labels.
    X_test, y_test : np.ndarray
        Test time series and labels.
    """
    def __init__(self, name="ECG200"):
        self.dirname = DIRNAME
        self.name = name

    def _parse_file(self, filename):
        """
        Parse a UCR data file.

        Each line: label, v1, v2, ..., vT

        Parameters

        filename : str
            Path to a TRAIN or TEST file.

        Returns

        X : list of np.ndarray
            List of time series reshaped as (T, 1).
        y : np.ndarray
            Corresponding labels.
        """
        X, y = [], []
        with open(filename, "r") as f:
            for line in f:
                values = line.strip().split(",")
                label = int(values[0])
                series = np.array(values[1:], dtype=np.float64).reshape(-1, 1)
                y.append(label)
                X.append(series)
        return X, np.array(y)

    def load_dataset(self):
        """
        Load TRAIN and TEST splits of the selected UCR dataset.

        Returns
        
        X_train, y_train, X_test, y_test : np.ndarray
            Loaded data splits.

        Raises
        
        FileNotFoundError
            If the dataset folder or files are missing.
        """
        print(f"Loading UCR dataset: {self.name}")

        folder = os.path.join(self.dirname, self.name)
        train_path = os.path.join(folder, f"{self.name}_TRAIN")
        test_path  = os.path.join(folder, f"{self.name}_TEST")

        try:
            X_train, y_train = self._parse_file(train_path)
            X_test, y_test = self._parse_file(test_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Dataset '{self.name}' not found in '{self.dirname}'. "
                "Please download the UCR archive and place it accordingly."
            )
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_test, y_test = np.array(X_test), np.array(y_test)
        self.X_train, self.y_train = X_train, y_train
        self.X_test,  self.y_test  = X_test, y_test

        return X_train, y_train, X_test, y_test


    def unique_classes(self, split="train"):
        """
        Return all unique labels in a split.

        Parameters
        
        split : {"train", "test"}
            Dataset split to inspect.

        Returns
        
        np.ndarray
            Sorted list of unique class labels.
        """
        y = self.y_train if split == "train" else self.y_test
        return np.unique(y)


    def get(self, split="train", idx=0):
        """
        Retrieve one sample and its label.

        Parameters
        split : {"train", "test"}
        idx : int
            Index of the sample.

        Returns
        X : np.ndarray
            Time series sample.
        y : int
            Corresponding label.
        """
        if split == "train":
            return self.X_train[idx], self.y_train[idx]
        else:
            return self.X_test[idx], self.y_test[idx]


    def indices_of_class(self, cls, split="train"):
        """
        Return indices of all samples belonging to a given class.

        Parameters
        cls : int
            Target class.
        split : {"train", "test"}

        Returns
        np.ndarray
            Array of indices.
        """
        y = self.y_train if split == "train" else self.y_test
        return np.where(y == cls)[0]


    def sample_by_class(self, cls, n=1, split="train", seed=None):
        """
        Sample n series belonging to a specific class.

        Parameters
        cls : int
            Class to sample.
        n : int
            Number of series to draw.
        seed : int or None
            Random seed.

        Returns
        X_samples : np.ndarray
            Sampled series.
        cls : int
            The class (returned for convenience).
        """
        rng = np.random.default_rng(seed)
        idxs = self.indices_of_class(cls, split=split)
        chosen = rng.choice(idxs, size=n, replace=False)

        X = self.X_train if split == "train" else self.X_test
        return X[chosen], cls


    def sample_random_class_series(self, X, y, n, seed=0):
        """
        Randomly pick a class, then sample n series from it.

        Parameters
        X : np.ndarray
            Input series.
        y : np.ndarray
            Corresponding labels.
        n : int
            Number of samples.
        seed : int
            Random seed.

        Returns
        X_sel : np.ndarray
            Sampled series.
        chosen_class : int
            Randomly selected class.
        """
        rng = np.random.default_rng(seed)
        classes = np.unique(y)
        chosen_class = rng.choice(classes)
        X_class = X[y == chosen_class]
        idx = rng.choice(len(X_class), size=n, replace=False)
        return X_class[idx], chosen_class


    def sample_one_per_class(self, split="train", seed=None):
        """
        Sample exactly one series for each class.

        Useful for prototypical networks (one prototype per class).

        Parameters
        split : {"train", "test"}
        seed : int or None

        Returns
        list of np.ndarray
            One example per class.
        """
        rng = np.random.default_rng(seed)
        classes = self.unique_classes(split=split)

        X = self.X_train if split == "train" else self.X_test
        y = self.y_train if split == "train" else self.y_test

        out = []
        for cls in classes:
            idxs = np.where(y == cls)[0]
            chosen = rng.choice(idxs)
            out.append(X[chosen])

        return out  

