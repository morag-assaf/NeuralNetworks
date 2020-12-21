import pandas as pd
import numpy as np
from config import CONF

class DataSynthesizer:
    """
    This class create additional samples according to the existing data.

    A Synthetic Data Generator is a Python function (or class method) that takes as input some data, which we call the
    real data, learns a model from it, and outputs new synthetic data that has similar mathematical properties as the real one.

    Reference:
    [1]  https://github.com/sdv-dev/SDGym

    """
    def __init__(self, input_file = None, output_file= None, num_samples = 10000):

        # Initialize variables for the synthesizer
        self.df                  = pd.read_csv(input_file) if input_file else pd.read_csv('/Users/assafmorag/Projects/Sakranut_Project/modules/model.csv')
        self.out_file            = output_file if output_file else '/Users/assafmorag/Projects/Sakranut_Project/modules/syn_data_200k.csv'
        self.data                = self.df.to_numpy().transpose()
        self.cov                 = np.cov(self.data)
        self.mean                = np.mean(self.data, axis=1)
        self.df_synthesis        = None
        self.num_samples         = num_samples


    def runSynthesizer(self):
        # Synthesize new samples according to the model
        sampled = np.random.multivariate_normal(self.mean, self.cov, self.num_samples)
        sampled = self.post_synthesis(sampled)
        # Saving the samples as dataframe
        self.df_synthesis = pd.DataFrame(data=sampled, columns=['male', 'age', 'education', 'cigsPerDay', 'BPMeds',
       'prevalentStroke', 'prevalentHyp', 'diabetes', 'totChol', 'sysBP',
       'diaBP', 'BMI', 'heartRate', 'glucose', 'TenYearCHD'])
        # # Combine the original data with the synthesize one
        df_combine = pd.concat([self.df, self.df_synthesis])
        # # Save a new csv
        df_combine.to_csv(self.out_file, index=False)

    def post_synthesis(self, sampled):
        sampled[:, [0, 3, 5, 6, 7, 8, 14]] = np.clip(sampled[:, [0, 3, 5, 6, 7, 8, 14]], 0, 1)
        sampled[:, 1] = np.clip(sampled[:, 1], 30, 70)
        sampled[:, 2] = np.clip(sampled[:, 2], 1, 4)
        sampled[:, 4] = np.clip(sampled[:, 4], 0, 150)
        return np.rint(sampled).astype(int)

synthesizer = DataSynthesizer(num_samples=CONF['num_syn_samples'])
synthesizer.runSynthesizer()