import pandas as pd
from sklearn.neighbors import NearestNeighbors
import pickle
import os

class AD:
    
    def __init__(self, ad_dirpath="adparams/", image_dirpath="images/", error_dirpath="errors/",model_dirpath="models/", other_dirpath="others/"):
        self.ad_dirpath = ad_dirpath
        self.image_dirpath = image_dirpath
        self.error_dirpath = error_dirpath
        self.model_dirpath = model_dirpath
        self.other_dirpath = other_dirpath

    def get_threshold(self, df, k=5):
        if not os.path.exists(self.ad_dirpath+"nn_train.pickle") and not os.path.exists(self.ad_dirpath+"th_train.pickle"):
            if not os.path.exists(self.ad_dirpath):
                os.mkdir(self.ad_dirpath)

            neigh = NearestNeighbors(n_neighbors=len(df), radius=None)
            neigh.fit(df)
            dij = pd.DataFrame(neigh.kneighbors(df, return_distance=True)[0]).T[1:]
            di_ave = dij[:k].mean()
            q1 = di_ave.quantile(0.25)
            q3 = di_ave.quantile(0.75)
            refval = q3 + 1.5*(q3-q1)
            Ki = dij[dij <= refval].count().values
            ti = (dij[dij <= refval].sum()/Ki)
            mint = ti[ti > 0].min()
            ti = ti.fillna(mint).values

            # REFACTOR: need to consider file extensions
            with open(os.path.join(self.ad_dirpath,'nn_train.pickle'), 'wb') as f:
                pickle.dump(neigh, f)

            with open(os.path.join(self.ad_dirpath,'th_train.pickle'), 'wb') as f:
                pickle.dump(ti, f)
        else:
            # REFACTOR: need to consider file extensions
            with open(os.path.join(self.ad_dirpath,'nn_train.pickle'), 'rb') as f:
                neigh = pickle.load(f)
            with open(os.path.join(self.ad_dirpath,'th_train.pickle'), 'rb') as f:
                ti = pickle.load(f)

        return neigh, ti

    def count_AD(self, nn, df, thlist):
        dists = nn.kneighbors(df, return_distance=True)[0]
        return (dists <= thlist).sum(axis=1)