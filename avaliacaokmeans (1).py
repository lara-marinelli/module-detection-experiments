import os
import sys
sys.path.insert(0,os.path.abspath("lib/"))
sys.path.append(os.getcwd()) 

import json
#from sklearn.metrics import pairwise_distances
from sklearn import metrics

import pandas as pd
import numpy as np
methodnames = ["dummy", "agglom", "ica_zscore", "meanshift","kmeans","spectral_biclust"]
#methodnames=["spectral_biclust"]

def _get_clust_pairs(clusters):
    return [(i, j) for i in clusters for j in clusters if i > j]
  
def _dunn(data=None, dist=None, labels=None):
    clusters = set(labels)
    inter_dists = [
        dist[np.ix_(labels == i, labels == j)].min()
        for i, j in _get_clust_pairs(clusters)
    ]
    intra_dists = [
        dist[np.ix_(labels == i, labels == i)].max()
        for i in clusters
    ]
    return min(inter_dists) / max(intra_dists)


finalscores = []
for methodname in methodnames:
    settings_name = "paramexplo/" + methodname
    settings = json.load(open("conf/settings/{}.json".format(settings_name)))
    #settings_dataset = pd.DataFrame([dict(settingid=setting["settingid"], **json.load(open("../" + setting["dataset_location"]))["params"]) for setting in settings])
    #settings_method = pd.DataFrame([dict(settingid=setting["settingid"], **json.load(open("../" + setting["method_location"]))["params"]) for setting in settings])
    
    for setting in settings:
      resultlinha = {}
      try:
        modules =  pd.read_json("" + setting["output_folder"] + "modules.json")
        dataset = json.load(open("" + setting["dataset_location"]))
        expression = pd.read_csv("" + dataset["expression"], sep="\t", index_col=0)
        expression = expression.T    
        method_location = json.load(open("" + setting["method_location"]))


        for i, linha_atual in modules.iterrows():
          linha_atual = linha_atual[~(linha_atual.isna())]
          expression.loc[linha_atual,"cluster"] = i

        labels = expression["cluster"]
        expression = expression.drop(["cluster"], axis=1)
      except:
        continue
 
      try:
        silhouette = metrics.silhouette_score(expression, labels, metric='euclidean')
        db = metrics.davies_bouldin_score(expression, labels)
        dist = metrics.pairwise_distances(expression)
        expression = expression.reset_index()
        expression = expression.drop("index",axis=1)
        dunn = _dunn(dist=dist, labels=labels.values)

      except Exception as e:
        print("Oops!", e, "occurred.")
        print("Next entry.")
        silhouette=None
        db=None
        dunn = None
      resultlinha["silhouette"] = silhouette
      resultlinha["davies-bouldin"] = db 
      resultlinha["dunn"] = dunn 
      resultlinha["dataset"] = setting["dataset_name"]
      resultlinha["setting_id"] = setting["settingid"]
      resultlinha["params"] = str(method_location["params"])
      resultlinha["method"] = method_location["params"]["method"]
      finalscores.append(resultlinha)
      

finalscores
pd.DataFrame(finalscores).to_csv("NOVOS_PARAMETROS_PROGESTERONA_TODOS_RESULTADOS.CSV", sep="\t")