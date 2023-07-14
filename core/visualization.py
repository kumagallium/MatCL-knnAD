import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_log_error
from pycaret import regression
import os
import matplotlib
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

import seaborn as sns
from matplotlib.patches import Rectangle

plt.rcParams["font.size"] = 11
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial"]
plt.rcParams["xtick.direction"] = "in"
plt.rcParams["ytick.direction"] = "in"
plt.rcParams["xtick.major.width"] = 1.2
plt.rcParams["ytick.major.width"] = 1.2
plt.rcParams["xtick.major.size"] = 3
plt.rcParams["ytick.major.size"] = 3
plt.rcParams["axes.grid.axis"] = "both"
plt.rcParams["axes.linewidth"] = 1.2
plt.rcParams["axes.grid"] = True
plt.rcParams["axes.edgecolor"] = "black"
plt.rcParams["grid.linestyle"] = "--"
plt.rcParams["grid.linewidth"] = 0.3
plt.rcParams["legend.markerscale"] = 2
plt.rcParams["legend.fancybox"] = False
plt.rcParams["legend.framealpha"] = 1
plt.rcParams["legend.edgecolor"] = "black"


class Visualization:
    def __init__(self, image_dirpath="images/"):
        self.image_dirpath = image_dirpath

    def save_numofdata_years_cluster(self, yclist, matfamily, df_cluster):
        years = np.sort(df_cluster["year"].unique())
        fig = plt.figure(figsize=(3.4, 3), dpi=300, facecolor="w", edgecolor="k")
        ax = fig.add_subplot(1, 1, 1)
        # ax.set_yscale("log")
        ax.set_xlabel("Published year")
        ax.set_ylabel("Number of training data")
        ax.set_xlim(2001, 2020)
        # ax.set_ylim(1, 3*10**4)
        cmap = plt.get_cmap("tab20c").colors

        ax.stackplot(
            years,
            np.array(yclist)[df_cluster["cluster"].value_counts().index.values],
            labels=np.array(matfamily)[
                df_cluster["cluster"].value_counts().index.values
            ],
            colors=cmap,
        )

        ax.legend(
            loc="upper left",
            bbox_to_anchor=(0.01, 0.99),
            fontsize=6.5,
            facecolor="white",
            framealpha=1,
        ).get_frame().set_linewidth(0.5)
        plt.xticks([2000 + 5 * i for i in range(5)])
        plt.tight_layout()

        if not os.path.exists(self.image_dirpath):
            os.mkdir(self.image_dirpath)
        fig.savefig(os.path.join(self.image_dirpath, "numofdata_years_cluster.png"))

    def save_parityplot_ad_starrydata_targets(
        self, targets, ad_reliability, df_test_inAD, df_test_outAD, inputsize
    ):
        fig = plt.figure(figsize=(6, 6), dpi=300, facecolor="w", edgecolor="k")
        alflist = [
            "a",
            "b",
            "c",
            "d",
            "e",
            "f",
            "g",
            "h",
            "i",
            "j",
            "k",
            "l",
            "m",
            "n",
            "o",
            "p",
            "q",
            "r",
            "s",
            "t",
            "u",
            "v",
            "w",
            "x",
            "y",
            "z",
        ]
        error_table = {}
        for idx, tg in enumerate(targets):
            ax = fig.add_subplot(2, 2, idx + 1)
            ax.xaxis.set_ticks_position("both")
            ax.yaxis.set_ticks_position("both")
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
                trueAD = pred_model.loc[:, tg].values

                pred_model_out = regression.predict_model(
                    selected_model, data=test_outAD
                )
                pred_model_out["prediction_label"] = (
                    pred_model_out["prediction_label"] * pred_model_out["Temperature"] * 10**-3
                )
                predAD_out = pred_model_out["prediction_label"].values
                trueAD_out = df_test_outAD.loc[:, tg].values

                ax.set_xlabel("Experimental $zT$")
                ax.set_ylabel("Predicted $zT$")
                t_min = 5
                t_max = -2
                xmin = 0
                xmax = 2
                ymin = 0
                ymax = 2
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
            elif tg == "ZTcalc":
                test_inAD_S = pd.concat(
                    [
                        df_test_inAD.iloc[:, :inputsize],
                        df_test_inAD[["Seebeck coefficient"]],
                    ],
                    axis=1,
                )
                test_outAD_S = pd.concat(
                    [
                        df_test_outAD.iloc[:, :inputsize],
                        df_test_outAD[["Seebeck coefficient"]],
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
                pred_model_out_S = regression.predict_model(
                    selected_model_S, data=test_outAD_S
                )
                predAD_out_S = pred_model_out_S["prediction_label"].values

                test_inAD_El = pd.concat(
                    [
                        df_test_inAD.iloc[:, :inputsize],
                        df_test_inAD[["Electrical conductivity"]],
                    ],
                    axis=1,
                )
                test_outAD_El = pd.concat(
                    [
                        df_test_outAD.iloc[:, :inputsize],
                        df_test_outAD[["Electrical conductivity"]],
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
                pred_model_out_El = regression.predict_model(
                    selected_model_El, data=test_outAD_El
                )
                predAD_out_El = pred_model_out_El["prediction_label"].values

                test_inAD_k = pd.concat(
                    [
                        df_test_inAD.iloc[:, :inputsize],
                        df_test_inAD[["Thermal conductivity"]],
                    ],
                    axis=1,
                )
                test_outAD_k = pd.concat(
                    [
                        df_test_outAD.iloc[:, :inputsize],
                        df_test_outAD[["Thermal conductivity"]],
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
                pred_model_out_k = regression.predict_model(
                    selected_model_k, data=test_outAD_k
                )
                predAD_out_k = pred_model_out_k["prediction_label"].values

                predAD = (
                    ((predAD_S * 10**-6) ** 2) * (predAD_El) / predAD_k
                ) * df_test_inAD["Temperature"]
                trueAD = df_test_inAD.loc[:, "ZTcalc"].values

                predAD_out = (
                    ((predAD_out_S * 10**-6) ** 2) * (predAD_out_El) / predAD_out_k
                ) * df_test_outAD["Temperature"]
                trueAD_out = df_test_outAD.loc[:, "ZTcalc"].values

                ax.set_xlabel("Experimental $zT_{ \mathrm{calc}}$")
                ax.set_ylabel("Predicted $zT_{ \mathrm{calc}}$")
                t_min = 5
                t_max = -2
                xmin = 0
                xmax = 2
                ymin = 0
                ymax = 2
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
            elif tg == "PFcalc":
                test_inAD_S = pd.concat(
                    [
                        df_test_inAD.iloc[:, :inputsize],
                        df_test_inAD[["Seebeck coefficient"]],
                    ],
                    axis=1,
                )
                test_outAD_S = pd.concat(
                    [
                        df_test_outAD.iloc[:, :inputsize],
                        df_test_outAD[["Seebeck coefficient"]],
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
                pred_model_out_S = regression.predict_model(
                    selected_model_S, data=test_outAD_S
                )
                predAD_out_S = pred_model_out_S["prediction_label"].values

                test_inAD_El = pd.concat(
                    [
                        df_test_inAD.iloc[:, :inputsize],
                        df_test_inAD[["Electrical conductivity"]],
                    ],
                    axis=1,
                )
                test_outAD_El = pd.concat(
                    [
                        df_test_outAD.iloc[:, :inputsize],
                        df_test_outAD[["Electrical conductivity"]],
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
                pred_model_out_El = regression.predict_model(
                    selected_model_El, data=test_outAD_El
                )
                predAD_out_El = pred_model_out_El["prediction_label"].values

                predAD = ((predAD_S * 10**-6) ** 2) * (predAD_El) * (10**3)
                trueAD = df_test_inAD.loc[:, "PFcalc"].values
                # trueAD =((df_test_inAD.loc[:, "Seebeck coefficient"]*10**-6)**2)*(df_test_inAD.loc[:, "Electrical conductivity"])*(10**3)

                predAD_out = (
                    ((predAD_out_S * 10**-6) ** 2) * (predAD_out_El) * (10**3)
                )
                trueAD_out = df_test_outAD.loc[:, "PFcalc"].values
                # trueAD_out = ((df_test_outAD.loc[:, "Seebeck coefficient"]*10**-6)**2)*(df_test_outAD.loc[:, "Electrical conductivity"])*(10**3)

                ax.set_xlabel("Experimental $PF_{ \mathrm{calc}}$ [mWm$^{-1}$K$^{-2}$]")
                ax.set_ylabel("Predicted $PF_{ \mathrm{calc}}$ [mWm$^{-1}$K$^{-2}$]")
                xmin = 0
                xmax = 5
                ymin = 0
                ymax = 5
                ax.set_xlim(xmin, xmax)
                ax.set_ylim(ymin, ymax)
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

                pred_model_out = regression.predict_model(
                    selected_model, data=test_outAD
                )
                predAD_out = pred_model_out["prediction_label"].values
                trueAD_out = df_test_outAD.loc[:, tg].values

                if tg == "Thermal conductivity":
                    ax.set_xlabel("Experimental $\u03BA$ [Wm$^{-1}$K$^{-1}$]")
                    ax.set_ylabel("Predicted  $\u03BA$ [Wm$^{-1}$K$^{-1}$]")
                    xmin = 0
                    xmax = 7
                    ymin = 0
                    ymax = 7
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)
                elif tg == "Seebeck coefficient":
                    ax.set_xlabel("Experimental $S$ [\u03BCVK$^{-1}$]")
                    ax.set_ylabel("Predicted $S$ [\u03BCVK$^{-1}$]")
                    xmin = 0
                    xmax = 450
                    ymin = 0
                    ymax = 450
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)
                elif tg == "Electrical conductivity":
                    trueAD = trueAD / 1000000
                    predAD = predAD / 1000000
                    trueAD_out = trueAD_out / 1000000
                    predAD_out = predAD_out / 1000000
                    ax.set_xlabel(
                        "Experimental $\u03C3$ [10$^{6}$\u03A9$^{-1}$m$^{-1}$]"
                    )
                    ax.set_ylabel("Predicted  $\u03C3$ [10$^{6}$\u03A9$^{-1}$m$^{-1}$]")
                    xmin = 0
                    xmax = 0.6
                    ymin = 0
                    ymax = 0.6
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)
                elif tg == "PF":
                    ax.set_xlabel(
                        "Experimental $PF_{ \mathrm{calc}}$ [mWm$^{-1}$K$^{-2}$]"
                    )
                    ax.set_ylabel(
                        "Predicted $PF_{ \mathrm{calc}}$ [mWm$^{-1}$K$^{-2}$]"
                    )
                    xmin = 0
                    xmax = 5
                    ymin = 0
                    ymax = 5
                    ax.set_xlim(xmin, xmax)
                    ax.set_ylim(ymin, ymax)
                t_min = 0
                t_max = trueAD.max()

            mape = np.sum(np.abs(predAD - trueAD) / trueAD) / len(trueAD)
            rmsle = mean_squared_log_error(trueAD, predAD)
            mape_100 = (np.sum(np.abs(predAD - trueAD) / trueAD) / len(trueAD)) * 100
            rmsle_out = mean_squared_log_error(trueAD_out, predAD_out)
            mape_100_out = (
                np.sum(np.abs(predAD_out - trueAD_out) / trueAD_out) / len(trueAD_out)
            ) * 100
            r2_score_in = r2_score(trueAD, predAD)
            r2_score_out = r2_score(trueAD_out, predAD_out)
            error_table[tg] = {
                "mape": mape_100,
                "rmsle": rmsle,
                "mape_out": mape_100_out,
                "rmsle_out": rmsle_out,
                "r2_score": r2_score_in,
                "r2_score_out": r2_score_out,
            }

            ax.plot(
                [t_min - mape, t_max + mape * 2],
                [t_min - mape, t_max + mape * 2],
                alpha=0.1,
                lw=1,
                c="k",
            )

            trueAD_viz = np.array([trueAD.max() + 100])
            trueAD_viz = np.append(trueAD_viz, trueAD)
            predAD_viz = np.array([predAD.max() + 100])
            predAD_viz = np.append(predAD_viz, predAD)
            ad_reliability_viz = np.array([1])
            ad_reliability_viz_tmp = ad_reliability.copy()
            ad_reliability_viz_tmp = (
                ad_reliability_viz_tmp - ad_reliability_viz_tmp.min()
            ) / (ad_reliability_viz_tmp.max() - ad_reliability_viz_tmp.min())
            ad_reliability_viz = np.append(ad_reliability_viz, ad_reliability_viz_tmp)
            np.place(
                ad_reliability_viz, ad_reliability_viz == 0, ad_reliability_viz.min()
            )
            ax.scatter(
                trueAD_viz,
                predAD_viz,
                s=10,
                c="r",
                alpha=ad_reliability_viz,
                lw=0,
                label="Inside AD",
            )
            ax.scatter(
                trueAD_out,
                predAD_out,
                s=10,
                c="b",
                alpha=0.8,
                lw=0,
                marker="^",
                label="Outside AD",
            )

            ax.text(
                xmin - ((xmax - xmin) / 3),
                ymax + ((ymax - ymin) / 10),
                "(" + alflist[idx] + ")",
                size=11,
                ha="left",
                va="top",
            )

            ax.legend(
                loc="upper left",
                bbox_to_anchor=(0.01, 0.99),
                fontsize=8,
                facecolor="white",
                framealpha=1,
            ).get_frame().set_linewidth(0.5)

        plt.tight_layout()
        fig.savefig(os.path.join(self.image_dirpath, "TE_parityplot.png"))
        return pd.DataFrame(error_table)

    def show_errors_vs_number_of_adjacents(
        self, proplist, xlist, mapedict, rmsledict, sort_stack, matcolor
    ):
        linestyle = ["dashdot", "dotted", "dashdot", "dashed", "solid"]
        linecolor = ["C0", "C1", "C2", "C3", "C4"]

        fig = plt.figure(figsize=(7.5, 3), dpi=300, facecolor="w", edgecolor="k")
        ax = fig.add_subplot(1, 2, 1)
        ax.yaxis.set_ticks_position("both")
        ax.set_xlim(0, 400)
        ax.set_ylim(0, 100)
        ax.set_xlabel("Number of adjacent known materials")
        ax.set_ylabel("MAPE [%]")

        for idx, prop in enumerate(proplist):
            if prop == "ZT":
                ax.plot(
                    xlist,
                    mapedict[prop],
                    label="$zT$",
                    linestyle=linestyle[idx],
                    color=linecolor[idx],
                )
            elif prop == "ZTcalc":
                ax.plot(
                    xlist,
                    mapedict[prop],
                    label="$zT_{ \mathrm{calc}}$",
                    linestyle=linestyle[idx],
                    color=linecolor[idx],
                )
            elif prop == "PFcalc":
                ax.plot(
                    xlist,
                    mapedict[prop],
                    label="$PF_{ \mathrm{calc}}$",
                    linestyle=linestyle[idx],
                    color=linecolor[idx],
                )
            else:
                ax.plot(
                    xlist,
                    mapedict[prop],
                    label=prop,
                    linestyle=linestyle[idx],
                    color=linecolor[idx],
                )

        ax2 = ax.twinx()
        ax2.set_ylim(0, 1000)
        ax2.set_ylabel("Number of test data")
        ax.patch.set_visible(False)
        ax2.patch.set_visible(True)
        ax2.stackplot(xlist, sort_stack, colors=matcolor.values(), alpha=1)
        ax.set_zorder(1)
        ax2.set_zorder(0)
        ax.legend(
            loc="upper right",
            fontsize=7,
            facecolor="white",
            framealpha=1,
            markerscale=0.7,
        ).get_frame().set_linewidth(0.1)
        xmin = 0
        xmax = 400
        ymin = 0
        ymax = 100
        ax.text(
            xmin - ((xmax - xmin) / 3),
            ymax + ((ymax - ymin) / 10),
            "(a)",
            size=11,
            ha="left",
            va="top",
        )

        ax = fig.add_subplot(1, 2, 2)
        ax.yaxis.set_ticks_position("both")
        ax.set_xlim(0, 400)
        ax.set_ylim(
            0,
        )
        ax.set_xlabel("Number of adjacent known materials")
        ax.set_ylabel("RMSLE")

        for idx, prop in enumerate(proplist):
            if prop == "ZT":
                ax.plot(
                    xlist,
                    rmsledict[prop],
                    label="$zT$",
                    linestyle=linestyle[idx],
                    color=linecolor[idx],
                )
            elif prop == "ZTcalc":
                ax.plot(
                    xlist,
                    rmsledict[prop],
                    label="$zT_{ \mathrm{calc}}$",
                    linestyle=linestyle[idx],
                    color=linecolor[idx],
                )
            elif prop == "PFcalc":
                ax.plot(
                    xlist,
                    rmsledict[prop],
                    label="$PF_{ \mathrm{calc}}$",
                    linestyle=linestyle[idx],
                    color=linecolor[idx],
                )
            else:
                ax.plot(
                    xlist,
                    rmsledict[prop],
                    label=prop,
                    linestyle=linestyle[idx],
                    color=linecolor[idx],
                )

        ax2 = ax.twinx()
        ax2.set_ylim(0, 1000)
        ax2.set_ylabel("Number of test data")
        ax.patch.set_visible(False)
        ax2.patch.set_visible(True)
        ax2.stackplot(xlist, sort_stack, colors=matcolor.values(), alpha=1)
        ax.set_zorder(1)
        ax2.set_zorder(0)
        ax.legend(
            loc="upper right",
            fontsize=7,
            facecolor="white",
            framealpha=1,
            markerscale=0.7,
        ).get_frame().set_linewidth(0.1)
        xmin = 0
        xmax = 400
        ymin = 0
        ymax = 1
        ax.text(
            xmin - ((xmax - xmin) / 3),
            ymax + ((ymax - ymin) / 10),
            "(b)",
            size=11,
            ha="left",
            va="top",
        )

        plt.tight_layout()

        fig.savefig(os.path.join(self.image_dirpath, "stack_ad_errors.png"))

    def show_ranking_table_detail(
        self,
        df_table,
        df_mape,
        df_reltable,
        df_clstable,
        df_leaningdata,
        filrel=50,
        rank=20,
        Tmin=300,
        Tmax=900,
        Ttick=100,
        height=5,
        width=10,
        imagename="",
        ascending=False,
    ):
        df_table_max = (df_table + (df_table * (df_mape / 100))).copy()
        df_table_max = df_table_max.applymap(lambda x: "{:.3g}".format(x))
        df_table_min = (df_table - (df_table * (df_mape / 100))).copy()
        df_table_min[df_table_min < 0] = 0
        df_table = df_table[df_table_min > 0]
        df_table_min = df_table_min.applymap(lambda x: "{:.3g}".format(x))

        mpcomps = list(df_table.index)
        dftop = pd.DataFrame([], columns=list(range(Tmin, Tmax, Ttick)))
        dftopval = pd.DataFrame([], columns=list(range(Tmin, Tmax, Ttick)))
        dftoprel = pd.DataFrame([], columns=list(range(Tmin, Tmax, Ttick)))
        dftopcls = pd.DataFrame([], columns=list(range(Tmin, Tmax, Ttick)))
        dftopmax = pd.DataFrame([], columns=list(range(Tmin, Tmax, Ttick)))
        dftopmin = pd.DataFrame([], columns=list(range(Tmin, Tmax, Ttick)))
        for T in range(Tmin, Tmax + Ttick, Ttick):
            dftop[T] = list(df_table.sort_values(by=[T], ascending=ascending).index)
            index = df_table.sort_values(by=[T], ascending=ascending)[T].index
            dftoprel[T] = list(df_reltable.loc[index, T].values)
            dftopcls[T] = list(df_clstable.loc[index, T].values)
            dftopmax[T] = list(df_table_max.loc[index, T].values)
            dftopmin[T] = list(df_table_min.loc[index, T].values)
            dftopval[T] = list(
                df_table.sort_values(by=[T], ascending=ascending)[T].values
            )

        dftoprel = dftoprel.fillna(0)
        dftopcls = dftopcls.fillna(0)

        dftop_filrel = dftop[dftoprel > filrel].apply(
            lambda s: pd.Series(s.dropna().tolist()), axis=0
        )
        dftopval_filrel = dftopval[dftoprel > filrel].apply(
            lambda s: pd.Series(s.dropna().tolist()), axis=0
        )
        dftopcls_filrel = dftopcls[dftoprel > filrel].apply(
            lambda s: pd.Series(s.dropna().tolist()), axis=0
        )
        dftoprel_filrel = dftoprel[dftoprel > filrel].apply(
            lambda s: pd.Series(s.dropna().tolist()), axis=0
        )
        dftopmax_filrel = dftopmax[dftoprel > filrel].apply(
            lambda s: pd.Series(s.dropna().tolist()), axis=0
        )
        dftopmin_filrel = dftopmin[dftoprel > filrel].apply(
            lambda s: pd.Series(s.dropna().tolist()), axis=0
        )

        dftop_filrel = dftop_filrel.fillna("")
        # dftopval_filrel = dftopval_filrel.fillna(0)
        dftoprel_filrel = dftoprel_filrel.fillna(0)
        dftopcls_filrel = dftopcls_filrel.fillna(0)

        learning_materials = df_leaningdata["composition"].unique()

        dftop.index = dftop.index + 1
        dftopval.index = dftopval.index + 1
        dftoprel.index = dftoprel.index + 1
        dftopcls.index = dftopcls.index + 1
        dftopmin.index = dftopmin.index + 1
        dftopmax.index = dftopmax.index + 1
        dftop = dftop[dftopval > 0]

        dftop_filrel.index = dftop_filrel.index + 1
        dftopval_filrel.index = dftopval_filrel.index + 1
        dftoprel_filrel.index = dftoprel_filrel.index + 1
        dftopcls_filrel.index = dftopcls_filrel.index + 1
        dftopmin_filrel.index = dftopmin_filrel.index + 1
        dftopmax_filrel.index = dftopmax_filrel.index + 1
        dftop_filrel = dftop_filrel[dftopval_filrel > 0]

        plt.rcParams["font.size"] = 4.2
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["font.sans-serif"] = ["Arial"]
        plt.rcParams["axes.linewidth"] = 1.2
        plt.rcParams["axes.grid"] = False

        fig = plt.figure(figsize=(width, height), dpi=400, facecolor="w", edgecolor="k")
        ax = fig.add_subplot(1, 1, 1)
        ax.tick_params(pad=1)
        ax.xaxis.set_ticks_position("top")
        ax.tick_params(bottom="off", top="off")
        ax.tick_params(left="off")
        ax.tick_params(bottom=False, left=False, right=False, top=False)

        temprange = []
        for T in range(Tmin, Tmax + Ttick, Ttick):
            temprange.append(str(T) + " K")

        dfstr = dftop_filrel  # + '\n'
        dfstr2 = (
            "Predicted cluster: "
            + dftopcls_filrel.astype(str)
            + "\nNumber of adjacent: "
            + dftoprel_filrel.astype(int).astype(str)
            + "\nPredicted value: "
            + dftopmin_filrel.round(1).astype(str)
            + "-"
            + dftopmax_filrel.round(1).astype(str)
        )
        sns.heatmap(
            dftopval_filrel.loc[:rank, Tmin:Tmax],
            cmap="jet",
            annot=False,
            vmin=0,
            vmax=max(dftopval_filrel.max().values),
            yticklabels=1,
            xticklabels=temprange,
            cbar_kws={"pad": 0.01},
        )
        sns.heatmap(
            dftopval_filrel.loc[:rank, Tmin:Tmax],
            cmap="jet",
            annot=dfstr.loc[:rank, Tmin:Tmax],
            fmt="",
            annot_kws={"size": 7, "va": "bottom"},
            cbar=False,
        )
        sns.heatmap(
            dftopval_filrel.loc[:rank, Tmin:Tmax],
            cmap="jet",
            annot=dfstr2.loc[:rank, Tmin:Tmax],
            fmt="",
            annot_kws={"size": 5, "va": "top"},
            cbar=False,
        )

        for t in ax.texts:
            trans = t.get_transform()
            offs = matplotlib.transforms.ScaledTranslation(
                0, -0.15, matplotlib.transforms.IdentityTransform()
            )
            t.set_transform(offs + trans)

        for i, T in enumerate(tqdm(range(Tmin, Tmax + Ttick, Ttick))):
            uniqcomp = []
            for mat in learning_materials:
                if (
                    len(
                        dftop_filrel.loc[:rank, T][
                            dftop_filrel.loc[:rank, T] == mat
                        ].index
                    )
                    > 0
                ):
                    if mat not in uniqcomp:
                        ax.add_patch(
                            Rectangle(
                                (
                                    i,
                                    dftop_filrel.loc[:rank, T][
                                        dftop_filrel.loc[:rank, T] == mat
                                    ].index[0]
                                    - 1,
                                ),
                                1,
                                1,
                                fill=False,
                                edgecolor="grey",
                                lw=0.5,
                            )
                        )
                        ax.add_patch(
                            Rectangle(
                                (
                                    i,
                                    dftop_filrel.loc[:rank, T][
                                        dftop_filrel.loc[:rank, T] == mat
                                    ].index[0]
                                    - 1,
                                ),
                                1,
                                1,
                                fill=True,
                                edgecolor=None,
                                facecolor="grey",
                                alpha=0.8,
                                lw=0,
                            )
                        )
                        uniqcomp.append(mat)
            uniqcomp = []

        plt.tight_layout()  # pad=0.4, w_pad=1, h_pad=1.0)

        if imagename != "":
            if not os.path.exists(self.image_dirpath):
                os.mkdir(self.image_dirpath)
            fig.savefig(os.path.join(self.image_dirpath, imagename + ".png"))

        plt.show()

    def show_mcad(self, results, klist):
        fig = plt.figure(figsize=(4, 4), dpi=300, facecolor="w", edgecolor="k")
        ax = fig.add_subplot(1, 1, 1)
        ax.yaxis.set_ticks_position("both")
        ax.set_xlim(0, max(klist))
        ax.set_ylim(70, 101)
        ax.set_xlabel("$k$")
        ax.set_ylabel("Distribution of samples retained in AD [%]")

        out_style = dict(markeredgecolor="r", marker="+", ms=5)
        mean_style = dict(
            markerfacecolor="b",
            markeredgecolor="b",
            marker="D",
            linestyle="--",
            ms=3,
            lw=1,
        )
        medi_style = dict(color="r", lw=1)
        boxplot = ax.boxplot(
            results.values(),
            showmeans=True,
            meanprops=mean_style,
            medianprops=medi_style,
            flierprops=out_style,
        )

        meanx = []
        meany = []
        for meanplot in boxplot["means"]:
            meanxy = meanplot.get_xydata()
            meanx.append(meanxy[0][0])
            meany.append(meanxy[0][1])

        ax.plot(meanx, meany, color="b", lw=1)

        plt.xticks(list(klist)[::4], list(klist)[::4])
        plt.tight_layout()
        fig.savefig(os.path.join(self.image_dirpath, "MCAD.png"))
