from .kl_distance import compute_statistics

import glob
import os
import re
import warnings

import bilby
import matplotlib.pyplot as plt
import numpy as np
from typing import Union
import pandas as pd
from tqdm.auto import tqdm
from scipy.interpolate import UnivariateSpline


def get_npts_from_fname(fname: str) -> np.number:
    search_res = re.search(r"round\d+_(\d+)pts", fname)
    if search_res is None:
        warnings.warn(
            f"Filename {fname} does not match the expected format"
        )
        return np.nan
    return int(search_res.group(1))


def get_result_fnames(res_regex: str) -> pd.DataFrame:
    res_files = glob.glob(res_regex)
    npts = [get_npts_from_fname(f) for f in res_files]
    df = pd.DataFrame({"npts": npts, "fname": res_files})
    df = df.sort_values("npts", ascending=True)
    return df


def save_kl_distances(res_regex: str, ref_res_fname: Union[str, bilby.result.Result], fname: str = None) -> np.ndarray:
    """
    Get a list of KL distances for a set of results files
    fnames are like round1_100pts.json, round2_200pts.json, etc
    """
    if not os.path.exists(fname):
        print(f"Getting KLs for all results with {res_regex} + ref: {ref_res_fname}")
        result_df = get_result_fnames(res_regex)
        if ref_res_fname is None:
            ref_res_fname = result_df.fname.values[-1]
        ref_res = bilby.read_in_result(ref_res_fname)
        kl, ks, js = [], [], []
        params = list(ref_res.injection_parameters.keys())
        for i, f in enumerate(tqdm(result_df.fname, desc="Calculating staistics")):
            r = bilby.read_in_result(f)
            stats = compute_statistics(r.posterior[params], ref_res.posterior[params])
            kl.append(stats.kl)
            ks.append(stats.ks)
            js.append(stats.js)
        result_df["ref_res"] = ref_res_fname
        result_df["kl"] = kl
        result_df["ks"] = ks
        result_df["js"] = js
        result_df.to_csv(fname, index=False)
    result_df = pd.read_csv(fname)
    plot_kl_distances(result_df, fname.replace(".csv", ".png"))
    return result_df


def plot_kl_distances(kl_data, fname):
    fig, ax = plt.subplots()
    ax.plot(kl_data['kl'], color="tab:green", label="KL Divergence")
    ax2 = ax.twinx()
    ax2.plot( kl_data["ks"], label="KS Statistic", color="tab:red")
    ax3 = ax.twinx()
    ax3.plot(kl_data["js"], label="JS Divergence", color="tab:blue")
    ax.set_xlabel("Number of y-pts")
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax.legend(lines + lines2+lines3, labels + labels2 + labels3, loc=0)
    fig.savefig(fname)


def get_kls_for_entire_result_dir(base_dir):
    result_dirs = glob.glob(os.path.join(base_dir, "out_surr_*/out_mcmc"))
    for res_dir in tqdm(result_dirs, desc="Processing result directories"):
        high_mcmc_fnames = get_result_fnames(os.path.join(res_dir, "round*_highres_result.json"))
        # get the reference result (the one with the most points)
        ref_res_fname = high_mcmc_fnames.fname.values[-1]
        round_num = int(re.search(r"round(\d+)_", ref_res_fname).group(1))
        if round_num>33:
            print(f"Getting KLs for all results with {os.path.join(res_dir, '*variable_lnl_result.json')} + ref: {ref_res_fname}")
            save_kl_distances(res_regex=os.path.join(res_dir, "*variable_lnl_result.json"), ref_res_fname=ref_res_fname,
                              fname=os.path.join(res_dir, "variable_kldists_hist.csv"))
    print("\n\n------Done!!!------")



def load_all_kl_datasets(base_dir):
    kl_distance_csv = glob.glob(os.path.join(base_dir, "out_surr_*/out_mcmc/variable_kldists_hist.csv"))
    dfs = [pd.read_csv(f) for f in kl_distance_csv]
    max_len = max(len(df) for df in dfs)
    dfs = [df for df in dfs if len(df) == max_len]
    data = {f"{stat}_{i}": np.pad(df[stat], (0, max_len - len(df)), 'constant')
            for i, df in enumerate(dfs)
            for stat in ['kl', 'ks','js']}
    data['npts'] = dfs[np.argmax([len(df) for df in dfs])]['npts']
    df = pd.DataFrame(data)
    return df


def _smooth(x, y, new_x, s=0.005, k=1):
    return UnivariateSpline(x, y, s=s,k=k)(new_x)


def pre_process_all_kl_data(df, smooth_factor=0.00005, metric='kl_distance'):
    # preprocessing data
    x = df['npts']
    new_x = np.arange(df['npts'].min(), df['npts'].max(), 10)
    new_x = np.unique(np.round(new_x).astype(int))
    metric_data = df.filter(like=metric).values
    _, n_data = metric_data.shape
    medians = _smooth(x, np.median(metric_data, axis=1), new_x, s=smooth_factor, k=3)
    lower_ci = _smooth(x, np.percentile(metric_data, 10, axis=1), new_x, s=smooth_factor / 10, k=2)
    upper_ci = _smooth(x, np.percentile(metric_data, 80, axis=1), new_x, s=smooth_factor * 150,k=1)
    ci_data = pd.DataFrame(dict(
        npts=new_x,
        lower_ci_95=lower_ci,
        upper_ci_95=upper_ci,
        medians=medians
    ))
    ci_data.to_csv(f"{metric}_ci_data.csv", index=False)


def plot_smoothed_metrics(metric='kl'):
    data = pd.read_csv(f"{metric}_ci_data.csv")
    x, y, yu, yl = data.npts, data.medians, data.upper_ci_95, data.lower_ci_95
    x, y, yu, yl = np.array(x.tolist()), y.tolist(), yu.tolist(), yl.tolist()
    fig, axs = plt.subplots(1, 1, figsize=(6, 5))
    axs.plot(x, y, label='Median')
    axs.fill_between(x.tolist(), yl, yu, alpha=0.3, label='95% CI')
    axs.set_ylabel(f'{metric.capitalize().replace("_", " ")}')
    axs.set_xlabel('Number of GP training Points')
    axs.legend()
    axs.set_xlim(min(data.npts), 900)
    plt.tight_layout()
    axs.set_ylim(bottom=0)
    fig.savefig(f"{metric}.png")


get_kls_for_entire_result_dir(base_dir)
df = load_all_kl_datasets(base_dir)
pre_process_all_kl_data(df, metric='kl')
plot_smoothed_metrics(metric='kl')

# plot_smoothed_metrics(df)
#
#
# get_kls_for_entire_result_dir(base_dir="/fred/oz303/avajpeyi/studies/cosmic_int/simulation_study/4dim/out_pp_8/")
#
#
#
#
