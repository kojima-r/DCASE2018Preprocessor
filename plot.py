import numpy as np
import joblib
import json
import sys
import os
import shutil
from matplotlib.colors import LinearSegmentedColormap
import argparse
from dmm.plot_input import load_plot_data, get_default_argparser
from matplotlib import pylab as plt
def generate_cmap(colors):
    values = range(len(colors))
    vmax = np.ceil(np.max(values))
    color_list = []
    for v, c in zip(values, colors):
        color_list.append((v / vmax, c))
    return LinearSegmentedColormap.from_list("custom_cmap", color_list)


def draw_heatmap(h1, cmap):
    plt.imshow(
        h1, aspect="auto", interpolation="none", cmap=cmap, vmin=-1.0, vmax=1.0
    )  #
    plt.gca().xaxis.set_ticks_position("none")
    plt.gca().yaxis.set_ticks_position("none")


def main():
    cmap = generate_cmap(["#0000FF", "#FFFFFF", "#FF0000"])
    data=[]
    filename="BirdVox-DCASE-20k.csv"
    feature="spec"
    fp=open(filename)
    next(fp)
    for line in fp:
        arr=line.strip().split(",")
        name=arr[0]
        label=arr[2]
        data.append({"label":label,"name":name,"feature":feature})
        #filename="wav/28d7bccb-30ce-40ae-a33a-c0bb3cb4c1e5.wav"
        #f=get_feature(filename)

    os.makedirs("sample_data",exist_ok=True)
    for el in data[:100]:
        if el["label"]=="0":
            continue
        #filename="npy/5446c27e-d029-4153-a84e-650431095f83.spec.npy"
        filename="npy/"+el["name"]+".spec.npy"
        o=np.load(filename)
        print(o.shape)
        plt.imshow(
            o, aspect="auto", interpolation="none", cmap=cmap
        )  #
        plt.gca().xaxis.set_ticks_position("none")
        plt.gca().yaxis.set_ticks_position("none")

        out_filename="sample_data/"+el["name"]+"."+el["label"]+".png"
        print(out_filename)
        plt.savefig(out_filename)
        wav_filename="wav/"+el["name"]+".wav"
        shutil.copy(wav_filename,"./sample_data/"+el["name"]+"."+el["label"]+".wav")
main()

