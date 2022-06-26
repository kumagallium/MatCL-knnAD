import tqdm
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
import pandas as pd
from concurrent.futures import ProcessPoolExecutor


class MCAD:
    def ad(self, df_ad, klist, it):
        results = {}
        BF_train, BF_test = train_test_split(df_ad, test_size=0.2, random_state=it)

        neigh = NearestNeighbors(n_neighbors=len(BF_train), radius=None)
        neigh.fit(BF_train)
        dij = pd.DataFrame(neigh.kneighbors(BF_train, return_distance=True)[0]).T[1:]
        dists = neigh.kneighbors(BF_test, return_distance=True)[0]
        for k in klist:
            di_ave = dij[:k].mean()
            q1 = di_ave.quantile(0.25)
            q3 = di_ave.quantile(0.75)
            refval = q3 + 1.5 * (q3 - q1)
            Ki = dij[dij <= refval].count().values
            ti = dij[dij <= refval].sum() / Ki
            mint = ti[ti > 0].min()
            ti = ti.fillna(mint).values

            inAD = (sum((dists <= ti).sum(axis=1) > 0) / len(BF_test)) * 100
            results[k] = inAD
        return results

    def mcad(self, df_ad, klist, iternum=1000, n_workers=None):
        results = []
        n_workers = None
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(self.ad, df_ad, klist, it) for it in range(iternum)
            ]
            for f in tqdm.tqdm(futures):
                results.append(f.result())

        return pd.DataFrame(results).to_dict(orient="list")
