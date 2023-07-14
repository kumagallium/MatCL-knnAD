import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_log_error
from pycaret import regression
import pickle
import os
from tqdm.notebook import tqdm


class MLModel:
    def __init__(self, error_dirpath="errors/", model_dirpath="models/"):
        self.error_dirpath = error_dirpath
        self.model_dirpath = model_dirpath

    def compare_models(self, target, df_train, inputsize):
        train = pd.concat([df_train.iloc[:, :inputsize], df_train[[target]]], axis=1)
        reg_models = regression.setup(
            train,
            target=target,
            session_id=1000,
            #silent=True,
            verbose=False,
            transform_target=True,
        )  # ,transformation=True,transform_target=True
        best_model_te = regression.compare_models()

    def get_params(self, target):
        selected_model = regression.load_model(
            "models/model_" + target.replace(" ", "_")
        )
        parmas_table = selected_model.get_params()["steps"][-1][1].get_params()
        return parmas_table

    def plot_model(self, target, plot="feature_all"):
        selected_model = regression.load_model(
            "models/model_" + target.replace(" ", "_")
        )
        regression.plot_model(selected_model.get_params()["steps"][-1][1], plot)

    def predicted_properties(self, target, data):
        selected_model = regression.load_model(
            "models/model_" + target.replace(" ", "_")
        )
        pred_model = regression.predict_model(selected_model, data=data)
        print(pred_model["prediction_label"].values[0])

    def create_ad_starrydata_models(self, targets, df_train, inputsize, tune_params={}):
        models = {}
        for target in tqdm(targets):
            train = pd.concat(
                [df_train.iloc[:, :inputsize], df_train[[target]]], axis=1
            )
            reg_models = regression.setup(
                train,
                target=target,
                session_id=1000,
                fold=5,
                #silent=True,
                verbose=False,
                transform_target=True,
            )
            print("create")
            selected_model = regression.create_model(
                "rf", verbose=False
            )
            if len(tune_params) > 0:
                print("tune")
                tune_model = regression.tune_model(
                    selected_model,
                    verbose=False,
                    custom_grid=tune_params,
                    fold=2,
                    search_algorithm="grid",
                    early_stopping=True,
                )
            else:
                tune_model = selected_model
            final_model = regression.finalize_model(tune_model)
            if not os.path.exists(self.model_dirpath):
                os.mkdir(self.model_dirpath)
            regression.save_model(
                final_model,
                model_name=os.path.join(
                    self.model_dirpath, "model_" + target.replace(" ", "_")
                ),
            )
            models[target] = final_model
        return models

    def get_errors_targets(
        self, targets, ad_reliability, df_test_inAD, df_test_outAD, inputsize, tick=20
    ):
        for idx, tg in enumerate(targets):
            if tg == "ZT":
                tg = "Z"
                test_inAD = pd.concat(
                    [df_test_inAD.iloc[:, :inputsize], df_test_inAD[[tg]]], axis=1
                )
                test_outAD = pd.concat(
                    [df_test_outAD.iloc[:, :inputsize], df_test_outAD[[tg]]], axis=1
                )
                selected_model = regression.load_model(
                    "models/model_" + tg.replace(" ", "_")
                )
                pred_model = regression.predict_model(selected_model, data=test_inAD)
                pred_model["prediction_label"] = (
                    pred_model["prediction_label"] * pred_model["Temperature"] * 10**-3
                )
                predAD = pred_model["prediction_label"].values
                trueAD = (
                    pred_model.loc[:, tg].values * pred_model["Temperature"] * 10**-3
                )
            elif tg == "ZTcalc":
                test_inAD_S = pd.concat(
                    [
                        df_test_inAD.iloc[:, :inputsize],
                        df_test_inAD[["Seebeck coefficient"]],
                    ],
                    axis=1,
                )
                selected_model_S = regression.load_model(
                    "models/model_Seebeck_coefficient"
                )
                pred_model_S = regression.predict_model(
                    selected_model_S, data=test_inAD_S
                )
                predAD_S = pred_model_S["prediction_label"].values

                test_inAD_El = pd.concat(
                    [
                        df_test_inAD.iloc[:, :inputsize],
                        df_test_inAD[["Electrical conductivity"]],
                    ],
                    axis=1,
                )
                selected_model_El = regression.load_model(
                    "models/model_Electrical_conductivity"
                )
                pred_model_El = regression.predict_model(
                    selected_model_El, data=test_inAD_El
                )
                predAD_El = pred_model_El["prediction_label"].values

                test_inAD_k = pd.concat(
                    [
                        df_test_inAD.iloc[:, :inputsize],
                        df_test_inAD[["Thermal conductivity"]],
                    ],
                    axis=1,
                )
                selected_model_k = regression.load_model(
                    "models/model_Thermal_conductivity"
                )
                pred_model_k = regression.predict_model(
                    selected_model_k, data=test_inAD_k
                )
                predAD_k = pred_model_k["prediction_label"].values

                predAD = (
                    ((predAD_S * 10**-6) ** 2) * (predAD_El) / predAD_k
                ) * df_test_inAD["Temperature"]
                trueAD = df_test_inAD.loc[:, "ZTcalc"].values
            elif tg == "PFcalc":
                test_inAD_S = pd.concat(
                    [
                        df_test_inAD.iloc[:, :inputsize],
                        df_test_inAD[["Seebeck coefficient"]],
                    ],
                    axis=1,
                )
                selected_model_S = regression.load_model(
                    "models/model_Seebeck_coefficient"
                )
                pred_model_S = regression.predict_model(
                    selected_model_S, data=test_inAD_S
                )
                predAD_S = pred_model_S["prediction_label"].values

                test_inAD_El = pd.concat(
                    [
                        df_test_inAD.iloc[:, :inputsize],
                        df_test_inAD[["Electrical conductivity"]],
                    ],
                    axis=1,
                )
                selected_model_El = regression.load_model(
                    "models/model_Electrical_conductivity"
                )
                pred_model_El = regression.predict_model(
                    selected_model_El, data=test_inAD_El
                )
                predAD_El = pred_model_El["prediction_label"].values

                predAD = ((predAD_S * 10**-6) ** 2) * (predAD_El) * (10**3)
                trueAD = df_test_inAD.loc[
                    :, "PFcalc"
                ].values  # ((df_test_inAD.loc[:, "Seebeck coefficient"]*10**-6)**2)*(df_test_inAD.loc[:, "Electrical conductivity"])

            else:
                test_inAD = pd.concat(
                    [df_test_inAD.iloc[:, :inputsize], df_test_inAD[[tg]]], axis=1
                )
                test_outAD = pd.concat(
                    [df_test_outAD.iloc[:, :inputsize], df_test_outAD[[tg]]], axis=1
                )
                selected_model = regression.load_model(
                    "models/model_" + tg.replace(" ", "_")
                )
                pred_model = regression.predict_model(selected_model, data=test_inAD)
                predAD = pred_model["prediction_label"].values
                trueAD = df_test_inAD.loc[:, tg].values

            r2list = []
            rmslelist = []
            mapelist = []
            for i in range(int(ad_reliability.max() / tick) + 1):
                relfil = ad_reliability >= ((tick * (i)))
                if sum(relfil) > 0:
                    r2 = r2_score(trueAD[relfil], predAD[relfil])
                    rmsle = mean_squared_log_error(
                        trueAD[relfil], predAD[relfil]
                    )  # np.sqrt(np.sum((np.log(predAD[relfil]+1)-np.log(trueAD[relfil]+1))**2)/len(trueAD[relfil]))
                    mape = (
                        np.sum(np.abs(predAD[relfil] - trueAD[relfil]) / trueAD[relfil])
                        / len(trueAD[relfil])
                    ) * 100
                    r2list.append(r2)
                    rmslelist.append(rmsle)
                    mapelist.append(mape)

            if not os.path.exists(self.error_dirpath):
                os.mkdir(self.error_dirpath)

            if tg == "Z":
                with open(
                    os.path.join(self.error_dirpath, "mapelist_" + tg + "T.pickle"),
                    "wb",
                ) as f:
                    pickle.dump(mapelist, f)
                with open(
                    os.path.join(self.error_dirpath, "rmslelist_" + tg + "T.pickle"),
                    "wb",
                ) as f:
                    pickle.dump(rmslelist, f)
                with open(
                    os.path.join(self.error_dirpath, "r2list_" + tg + "T.pickle"), "wb"
                ) as f:
                    pickle.dump(r2list, f)
            else:
                with open(
                    os.path.join(
                        self.error_dirpath,
                        "mapelist_" + tg.replace(" ", "_") + ".pickle",
                    ),
                    "wb",
                ) as f:
                    pickle.dump(mapelist, f)
                with open(
                    os.path.join(
                        self.error_dirpath,
                        "rmslelist_" + tg.replace(" ", "_") + ".pickle",
                    ),
                    "wb",
                ) as f:
                    pickle.dump(rmslelist, f)
                with open(
                    os.path.join(
                        self.error_dirpath, "r2list_" + tg.replace(" ", "_") + ".pickle"
                    ),
                    "wb",
                ) as f:
                    pickle.dump(r2list, f)

    def set_matfamily(self, val, matfamily):
        try:
            return matfamily[int(val)]
        except:
            return np.nan

    def count_AD(self, nn, df, thlist):
        dists = nn.kneighbors(df, return_distance=True)[0]
        return (dists <= thlist).sum(axis=1)

    def get_properties_tables(
        self,
        target,
        df_decriptor_tmp,
        nn_train,
        th_train,
        cluster_model,
        matfamily,
        Tmin=300,
        Tmax=1300,
        Ttick=100,
    ):
        df_decriptor = df_decriptor_tmp.iloc[:, 1:].reset_index(drop=True).copy()
        table = {}
        reltable = {}
        clstable = {}
        if target == "ZTcalc":
            model_S = regression.load_model("models/model_Seebeck_coefficient")
            model_k = regression.load_model("models/model_Thermal_conductivity")
            model_El = regression.load_model("models/model_Electrical_conductivity")
            for T in tqdm(range(Tmin, Tmax, Ttick)):
                df_decriptor["Temperature"] = T
                filterAD = self.count_AD(nn_train, df_decriptor, th_train) > 0
                if sum(filterAD) > 0:
                    ad_reliability = self.count_AD(
                        nn_train,
                        df_decriptor[filterAD].reset_index(drop=True),
                        th_train,
                    )
                    new_prediction_S = regression.predict_model(
                        model_S, data=df_decriptor[filterAD].reset_index(drop=True)
                    )
                    new_prediction_El = regression.predict_model(
                        model_El, data=df_decriptor[filterAD].reset_index(drop=True)
                    )
                    new_prediction_k = regression.predict_model(
                        model_k, data=df_decriptor[filterAD].reset_index(drop=True)
                    )
                    new_prediction_S["ZT"] = (
                        ((new_prediction_S["prediction_label"] * 10**-6) ** 2)
                        * (new_prediction_El["prediction_label"])
                        / new_prediction_k["prediction_label"]
                    ) * new_prediction_S["Temperature"]
                    clusterlist = cluster_model.predict(
                        df_decriptor[filterAD].reset_index(drop=True)
                    )
                    # df_test = pd.merge(new_prediction_S, df_decriptor_tmp[filterAD].reset_index(drop=True), left_index=True, right_index=True).copy()
                    new_prediction_S["composition"] = df_decriptor_tmp[filterAD][
                        "composition"
                    ].values

                    idx = 0
                    for comp, value in new_prediction_S[["composition", "ZT"]].values:
                        table.setdefault(comp, {})
                        table[comp][T] = value
                        reltable.setdefault(comp, {})
                        reltable[comp][T] = ad_reliability[idx]
                        clstable.setdefault(comp, {})
                        clstable[comp][T] = clusterlist[idx]
                        idx += 1
        elif target == "PFcalc":
            model_S = regression.load_model("models/model_Seebeck_coefficient")
            model_El = regression.load_model("models/model_Electrical_conductivity")
            for T in tqdm(range(Tmin, Tmax, Ttick)):
                df_decriptor["Temperature"] = T
                filterAD = self.count_AD(nn_train, df_decriptor, th_train) > 0
                if sum(filterAD) > 0:
                    ad_reliability = self.count_AD(
                        nn_train,
                        df_decriptor[filterAD].reset_index(drop=True),
                        th_train,
                    )
                    new_prediction_S = regression.predict_model(
                        model_S, data=df_decriptor[filterAD].reset_index(drop=True)
                    )
                    new_prediction_El = regression.predict_model(
                        model_El, data=df_decriptor[filterAD].reset_index(drop=True)
                    )
                    new_prediction_S["PF"] = (
                        ((new_prediction_S["prediction_label"] * 10**-6) ** 2)
                        * (new_prediction_El["prediction_label"])
                        * 10**3
                    )
                    clusterlist = cluster_model.predict(
                        df_decriptor[filterAD].reset_index(drop=True)
                    )
                    # df_test = pd.merge(new_prediction_S, df_decriptor_tmp[filterAD].reset_index(drop=True), left_index=True, right_index=True).copy()
                    new_prediction_S["composition"] = df_decriptor_tmp[filterAD][
                        "composition"
                    ].values

                    idx = 0
                    for comp, value in new_prediction_S[["composition", "PF"]].values:
                        table.setdefault(comp, {})
                        table[comp][T] = value
                        reltable.setdefault(comp, {})
                        reltable[comp][T] = ad_reliability[idx]
                        clstable.setdefault(comp, {})
                        clstable[comp][T] = clusterlist[idx]
                        idx += 1
        else:
            if target == "ZT":
                model = regression.load_model("models/model_Z")
            else:
                model = regression.load_model(
                    "models/model_" + target.replace(" ", "_")
                )
            for T in tqdm(range(Tmin, Tmax, Ttick)):
                df_decriptor["Temperature"] = T
                filterAD = self.count_AD(nn_train, df_decriptor, th_train) > 0
                if sum(filterAD) > 0:
                    ad_reliability = self.count_AD(
                        nn_train, df_decriptor[filterAD], th_train
                    )
                    new_prediction = regression.predict_model(
                        model, data=df_decriptor[filterAD].reset_index(drop=True)
                    )
                    if target == "ZT":
                        new_prediction["prediction_label"] = (
                            new_prediction["prediction_label"]
                            * (10**-3)
                            * new_prediction["Temperature"]
                        )
                    elif target == "Electrical conductivity":
                        new_prediction["prediction_label"] = new_prediction["prediction_label"] * (10**-5)
                    clusterlist = cluster_model.predict(
                        df_decriptor[filterAD].reset_index(drop=True)
                    )
                    # df_test = pd.merge(new_prediction, df_decriptor_tmp.reset_index(drop=True), left_index=True, right_index=True).copy()
                    new_prediction["composition"] = df_decriptor_tmp[filterAD][
                        "composition"
                    ].values

                    idx = 0
                    for comp, value in new_prediction[["composition", "prediction_label"]].values:
                        table.setdefault(comp, {})
                        table[comp][T] = value
                        reltable.setdefault(comp, {})
                        reltable[comp][T] = ad_reliability[idx]
                        clstable.setdefault(comp, {})
                        clstable[comp][T] = clusterlist[idx]
                        idx += 1
        df_clstable_tmp = pd.DataFrame(clstable).T

        df_clstable = df_clstable_tmp.applymap(
            self.set_matfamily, matfamily=matfamily
        ).copy()

        return pd.DataFrame(table).T, pd.DataFrame(reltable).T, df_clstable

    def get_mape(self, val, mapedlist, tick):
        try:
            return mapedlist[int(val / tick)]
        except:
            return np.nan

    def get_mape_table(self, target, df_reltable, tick):
        mapedlist = []
        with open(
            os.path.join(
                self.error_dirpath, "mapelist_" + target.replace(" ", "_") + ".pickle"
            ),
            "rb",
        ) as f:
            mapedlist = pickle.load(f)

        df_mape = df_reltable.applymap(
            self.get_mape, mapedlist=mapedlist, tick=tick
        ).copy()

        return df_mape
