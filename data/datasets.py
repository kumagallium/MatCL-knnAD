import urllib.error
import urllib.request
import zipfile
import os
import shutil
import pandas as pd
from matminer.datasets import load_dataset
import requests
import json


class Datasets:
    def __init__(
        self,
        dbname="starrydata",
        dtype="interpolated",
        filetype="csv",
        data_dir="datasets/",
    ):
        self.dbname = dbname
        self.dtype = dtype
        self.filetype = filetype
        self.data_dir = data_dir

    def info(self):
        if self.dbname == "starrydata":
            print("Database is '" + self.dbname + "(" + self.dtype + ")'")
        else:
            print("Database is '" + self.dbname + "'")

    def get_versions(self):
        base_url = "https://github.com"
        file_url = base_url + "/starrydata/starrydata_datasets/tree/master/datasets"
        response = requests.get(file_url)
        html = json.loads(response.text)
        datalist = []
        for item in html["payload"]["tree"]["items"]:
            datalist.append(item["path"].replace("datasets/",""))
        versionlist = sorted(
            [dl.split("/")[-1].split(".")[0] for dl in datalist if ".zip" in dl],
            reverse=True,
        )
        return versionlist

    def starrydata_download(self, version="last"):
        base_url = "https://github.com"
        file_url = base_url + "/starrydata/starrydata_datasets/tree/master/datasets"
        data_url = "https://github.com/starrydata/starrydata_datasets/blob/master/datasets/"
        response = requests.get(file_url)
        html = json.loads(response.text)
        datalist = []
        for item in html["payload"]["tree"]["items"]:
            datalist.append(item["path"].replace("datasets/",""))
        if version == "last":
            zippath = (
                data_url
                + sorted([dl for dl in datalist if ".zip" in dl], reverse=True)[0]
            )
        else:
            versionidx = self.get_versions().index(version)
            zippath = (
                data_url
                + sorted([dl for dl in datalist if ".zip" in dl], reverse=True)[
                    versionidx
                ]
            )
        zippath = zippath.replace("/blob/", "/raw/")
        print(self.data_dir)

        data_path = os.path.join(
            self.data_dir, self.dtype + "_starrydata_" + version + "." + self.filetype
        )
        if not os.path.exists(
            self.data_dir + self.dtype + "_starrydata_" + version + ".csv"
        ):
            if not os.path.exists(self.data_dir):
                os.mkdir(self.data_dir)
            save_path = self.data_dir + "download.zip"

            print("download " + zippath)
            try:
                with urllib.request.urlopen(zippath) as download_file:
                    data = download_file.read()
                    with open(save_path, mode="wb") as save_file:
                        save_file.write(data)
            except urllib.error.URLError as e:
                print(e)

            print("...")
            with zipfile.ZipFile(
                os.path.join(self.data_dir, "download.zip")
            ) as obj_zip:
                obj_zip.extractall(self.data_dir)
            print("unzip")

            dirname = zippath.split("/")[-1].split(".")[0]

            if self.dtype == "interpolated":
                shutil.copyfile(
                    os.path.join(
                        self.data_dir, dirname, dirname + "_interpolated_data.csv"
                    ),
                    data_path,
                )
            elif self.dtype == "raw":
                shutil.copyfile(
                    os.path.join(self.data_dir, dirname, dirname + "_rawdata.csv"),
                    data_path,
                )
            shutil.rmtree(self.data_dir + dirname)
            os.remove(self.data_dir + "download.zip")
        print("finished: " + data_path)

    def get_alldata(self, version="last"):
        if self.dbname == "starrydata":
            self.starrydata_download(version)
            data_path = os.path.join(
                self.data_dir,
                self.dtype + "_starrydata_" + version + "." + self.filetype,
            )
            try:
                if self.filetype == "csv":
                    df_data = pd.read_csv(data_path, index_col=0)
            except:
                df_data = pd.DataFrame([])
        elif self.dbname == "materials project":
            if self.filetype == "csv":
                if not os.path.exists(self.data_dir + "mp_all_20181018.csv"):
                    if not os.path.exists(self.data_dir):
                        os.mkdir(self.data_dir)
                    print("Downloading data")
                    df_data = load_dataset("mp_all_20181018")
                    print("...")
                    df_data.to_csv(self.data_dir + "mp_all_20181018.csv", index=False)
                    print("Download completed")
                else:
                    df_data = pd.read_csv(self.data_dir + "mp_all_20181018.csv")
            elif self.filetype == "pkl":
                if not os.path.exists(self.data_dir + "mp_all_20181018.pkl"):
                    if not os.path.exists(self.data_dir):
                        os.mkdir(self.data_dir)
                    df_data = load_dataset("mp_all_20181018")
                    df_data.to_pickle(self.data_dir + "mp_all_20181018.pkl")
                else:
                    df_data = pd.read_pickle(self.data_dir + "mp_all_20181018.pkl")

        return df_data
