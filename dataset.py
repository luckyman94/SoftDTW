import numpy as np
import os

class Dataset:
    def __init__(self, name="ECG200"):
        self.dirname = '/Users/ilan/sdtw_data/UCR_TS_Archive_2015'
        self.name = name



    def load_ucr(self):
        print("Loading UCR dataset:", self.name)
        folder = os.path.join(self.dirname, self.name)
        tr = os.path.join(folder, "%s_TRAIN" % self.name)
        te = os.path.join(folder, "%s_TEST" % self.name)

        print("Path train:", tr)
        print("Path test:", te)

        try:
            X_tr, y_tr = self._parse_file(tr)
            X_te, y_te = self._parse_file(te)
        except IOError:
            raise IOError("Please copy UCR_TS_Archive_2015/ to $HOME/sdtw_data/. "
                        "Download from www.cs.ucr.edu/~eamonn/time_series_data.")

        y_tr = np.array(y_tr)
        y_te = np.array(y_te)
        X_tr = np.array(X_tr)
        X_te = np.array(X_te)

        return X_tr, y_tr, X_te, y_te



    def _parse_file(self,filename):
        X = []
        y = []

        with open(filename, "r") as f:
            for line in f:
                values = line.strip().split(",")
                label = int(values[0])

                features = np.array(list(map(float, values[1:])), dtype=np.float64)
                features = features.reshape(-1, 1)

                y.append(label)
                X.append(features)

        return X, np.array(y)


        
