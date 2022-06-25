
import pymatgen.core as mg
import numpy as np
from sklearn import mixture
from bokeh.sampledata.periodic_table import elements
import collections
import pickle
import os
import matplotlib.pyplot as plt

class Cluster:
    
    def __init__(self,model_dirpath="models/", other_dirpath="others/"):
        self.model_dirpath = model_dirpath
        self.other_dirpath = other_dirpath

    def get_matfamily_cluster(self, df_data, inputsize, kind="BGM", clusternum=15, covariance_type="tied", random_state=10):
        df_cluster = df_data.copy()
        clusterinputs  = df_cluster.iloc[:,:inputsize].values
        if kind == "BGM":
            ms = mixture.BayesianGaussianMixture(n_components=clusternum, random_state=random_state, init_params="kmeans", covariance_type=covariance_type)  # diag, full,spherical,tied
            ms.fit(clusterinputs)
            labels = ms.predict(clusterinputs)

        if not os.path.exists(self.model_dirpath):
            os.mkdir(self.model_dirpath)
        pickle.dump(ms, open(os.path.join(self.model_dirpath,kind+"model"), 'wb'))

        df_cluster["cluster"] = labels
        clusters = np.sort(df_cluster["cluster"].unique())

        matfamily = []
        clusterelements = elements.copy()
        for c in clusters:
            clustercomp = df_cluster[df_cluster["cluster"] == c]["composition"].unique()
            clustereltmp = [0] * len(clusterelements)
            for comp in clustercomp:
                for el, frac in mg.Composition(comp).fractional_composition.as_dict().items():
                    clustereltmp[mg.Element(el).number-1] += 1  # frac
            clusterelements["count"] = clustereltmp
            matfamily.append("-".join(clusterelements.sort_values("count", ascending=False)[:3]["symbol"].values))

        return clusters, matfamily, df_cluster

    def get_year_cluster_list(self,df_cluster, clusters):
        years = np.sort(df_cluster["year"].unique())
        ylist = []
        for c in range(len(clusters)):
            ylist.append([])
            for y in years:
                ylist[c].append(df_cluster[(df_cluster["year"]<=y)&(df_cluster["cluster"]==c)]["cluster"].count())

        return ylist

    def get_matfamily_matcolor(self, df_cluster, matfamily):
        matcolor = {}
        cmap = plt.get_cmap("tab20c").colors
        for idx, mf in enumerate(np.array(matfamily)[df_cluster["cluster"].value_counts().index.values]):
            matcolor.update({matfamily.index(mf): list(cmap)[idx]})

        if not os.path.exists(self.other_dirpath):
            os.mkdir(self.other_dirpath)

        with open(os.path.join(self.other_dirpath,'matcolor'), 'wb') as f:
            pickle.dump(np.array(matfamily)[df_cluster["cluster"].value_counts().index.values], f)

        sort_matfamily = np.array(matfamily)[df_cluster["cluster"].value_counts().index.values]
        with open(os.path.join(self.other_dirpath,'matfamily'), 'wb') as f:
            pickle.dump(sort_matfamily, f)

        return matcolor, sort_matfamily

    def get_stack_ad_cluster(self, df_data, ad_reliability, clustermodel, clusters, inputsize, tick=20):
        matfamlylist = []
        for i in range(int(ad_reliability.max()/tick)+1):
            relfil = (ad_reliability >= ((tick*(i))))
            if sum(relfil) > 0:
                matfamlylist.append(collections.Counter(clustermodel.predict(df_data[relfil].iloc[:, :inputsize])))

        stack = []
        for idx, c in enumerate(clusters):
            stack.append([])
            for mf in matfamlylist:
                if c in mf:
                    stack[idx].append(mf[c])
                else:
                    stack[idx].append(0)

        return stack
