import os
import json
import numpy as np

class PathManager:
    def __init__(self, cfg_path=None):
        self.environ = parse_environ(cfg_path)

    @property
    def base(self):
        return self.environ['DATASET']

    @property
    def info(self):
        ratio = 0.9
        import pandas as pd
        df = pd.read_csv(os.path.join(self.base, 'train_val.csv'))
        df.loc[:, 'subset'] = pd.Series(1, index=df.index)
        for i in range(df.shape[0]):
            if np.random.rand() < ratio:
                df["subset"][i] = 0
            else:
                df["subset"][i] = 1
        return df

    @property
    def nodule_path(self):
        return os.path.join(self.base, 'train_val')

class TestPathManager:
    def __init__(self, cfg_path=None):
        self.environ = parse_environ(cfg_path)

    @property
    def base(self):
        return self.environ['DATASET']

    @property
    def info(self):
        import pandas as pd
        df = pd.read_csv(os.path.join(self.base, 'test.csv'))
        return df

    @property
    def nodule_path(self):
        return os.path.join(self.base, 'test')
    
def parse_environ(cfg_path=None):
    if cfg_path is None:
        cfg_path = os.path.join(os.path.dirname(__file__), "ENVIRON")
    assert os.path.exists(cfg_path), "`ENVIRON` does not exists."
    with open(cfg_path) as f:
        environ = json.load(f)
    return environ


PATH = PathManager()
TESTPATH = TestPathManager()

