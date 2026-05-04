# adapted from https://github.com/ermshaua/multivariate-clasp/blob/main/benchmark/metrics.py
# and https://github.com/ermshaua/multivariate-clasp/blob/main/benchmark/utils.py
import os
from glob import glob
from tqdm import tqdm
import collections

import numpy as np
import pandas as pd
import scipy.signal as spsig


def true_positives(T, X, margin=5):
    """Compute true positives without double counting
    """
    # make a copy so we don't affect the caller
    X = set(list(X))
    TP = set()
    for tau in T:
        close = [(abs(tau - x), x) for x in X if abs(tau - x) <= margin]
        close.sort()
        if not close:
            continue
        dist, xstar = close[0]
        TP.add(tau)
        X.remove(xstar)
    return TP


def f_measure(annotations, predictions, margin=5, alpha=0.5, return_PR=False):
    """Compute the F-measure based on human annotations.

    annotations : dict from user_id to iterable of CP locations
    predictions : iterable of predicted CP locations
    alpha : value for the F-measure, alpha=0.5 gives the F1-measure
    return_PR : whether to return precision and recall too

    Remember that all CP locations are 0-based!
    """
    # ensure 0 is in all the sets
    Tks = {k + 1: set(annotations[uid]) for k, uid in enumerate(annotations)}
    for Tk in Tks.values():
        Tk.add(0)

    X = set(predictions)
    X.add(0)

    Tstar = set()
    for Tk in Tks.values():
        for tau in Tk:
            Tstar.add(tau)

    K = len(Tks)

    P = len(true_positives(Tstar, X, margin=margin)) / len(X)

    TPk = {k: true_positives(Tks[k], X, margin=margin) for k in Tks}
    R = 1 / K * sum(len(TPk[k]) / len(Tks[k]) for k in Tks)

    F = P * R / (alpha * R + (1 - alpha) * P)
    if return_PR:
        return F, P, R
    return F


def overlap(A, B):
    """ Return the overlap (i.e. Jaccard index) of two sets
    """
    return len(A.intersection(B)) / len(A.union(B))


def partition_from_cps(locations, n_obs):
    """ Return a list of sets that give a partition of the set [0, T-1], as
    defined by the change point locations.
    """
    T = n_obs
    partition = []
    current = set()

    all_cps = iter(sorted(set(locations)))
    cp = next(all_cps, None)
    for i in range(T):
        if i == cp:
            if current:
                partition.append(current)
            current = set()
            cp = next(all_cps, None)
        current.add(i)
    partition.append(current)
    return partition


def cover_single(Sprime, S):
    """Compute the covering of a segmentation S by a segmentation Sprime.

    This follows equation (8) in Arbaleaz, 2010.
    """
    T = sum(map(len, Sprime))
    assert T == sum(map(len, S))
    C = 0
    for R in S:
        C += len(R) * max(overlap(R, Rprime) for Rprime in Sprime)
    C /= T
    return C


def covering(annotations, predictions, n_obs):
    """Compute the average segmentation covering against the human annotations.

    annotations : dict from user_id to iterable of CP locations
    predictions : iterable of predicted Cp locations
    n_obs : number of observations in the series
    0.8189300411522634

    """
    Ak = {
        k + 1: partition_from_cps(annotations[uid], n_obs)
        for k, uid in enumerate(annotations)
    }
    pX = partition_from_cps(predictions, n_obs)

    Cs = [cover_single(pX, Ak[k]) for k in Ak]
    return sum(Cs) / len(Cs)

# evaluates results from segmentation algorithm
def evaluate_segmentation_algorithm(dataset, n_timestamps, cps_true, cps_pred):
    f1_score = np.round(f_measure({0: cps_true}, cps_pred, margin=n_timestamps*0.01), 3)
    covering_score = np.round(covering({0: cps_true}, cps_pred, n_timestamps), 3)

    # print(f"{dataset}: F1-Score: {f1_score}, Covering-Score: {covering_score} Found CPs: {cps_pred}")

    return dataset, cps_true.tolist(), cps_pred, f1_score, covering_score


# load the data
def load_master_data(data_path=r"C:\Users\lucas\Data\HAS2023-Challenge\has2023_master.csv"):
    """
    Load the given CSV file containing the labelled challenge data.
    Returns a pandas DataFrame where each column is a sensor measurement
    or label and each row corresponds to a single time series.

    Parameters
    ----------
    data_path : str, default: "../datasets/has2023_master.csv.zip".
        Path to the csv file to be loaded.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the sensor data for the challenge.
    """
    if not os.path.exists("./change_points.parquet"):
        np_cols = ["change_points"]
        converters = {
            col: lambda val: np.array([]) if len(val) == 0 else np.array(eval(val)) for col
            in np_cols}
        df = pd.read_csv(data_path, converters=converters, compression="zip", usecols=["change_points", 'group', 'split'])
        df.to_parquet("./change_points.parquet")
    else:
        df = pd.read_parquet("./change_points.parquet")
    return df


def detect_changepoints(df: pd.DataFrame, window_size: int):

    # get the score as numpy array
    score = df["score"].to_numpy()
    orig_score = score.copy()
    score = (score-score.min()) / (score.max() - score.min())

    # get the peaks
    peaks, *_ = spsig.find_peaks(score, width=10, distance=100, prominence=0.35)
    return peaks


def main():

    # load the score files
    score_files = {tuple(os.path.splitext(os.path.split(file)[-1])[0].split('_')): pd.read_parquet(file) for file in tqdm(glob(os.path.join("scores", "*.parquet")), desc='Loading Scores')}

    # load the groundtruth
    gtdf = load_master_data(r".\has2023_master.csv.zip")

    # go through and reorder the files
    score_tmp = collections.defaultdict(dict)

    # go through the scores and fuse them
    for key, val in score_files.items():
        score_tmp[(key[2], key[3], key[1])][key[0]] = val
    score_files = dict(score_tmp)

    # go through a window size
    algorithm = "MSST"
    method = "rsvd"
    window_size = 300
    scores = score_files[(algorithm, method, str(window_size))]
    results = []
    f1_results = []
    for idx, score in scores.items():

        # get the ground truth change points and the group
        gtdf_selected = gtdf.iloc[int(idx), :]
        cp_gt = gtdf_selected["change_points"]
        split = gtdf_selected["split"]
        if split == 'private': continue

        # get the detections
        cp_detected = detect_changepoints(score, window_size)
        print(cp_detected)
        print(cp_gt)
        print()

        # make the evaluation
        *_, f1_score, coverscore = evaluate_segmentation_algorithm(idx, score.shape[0], cp_gt, cp_detected)
        results.append(coverscore)
        f1_results.append(f1_score)
    results = np.array(results)
    f1_results = np.array(f1_results)
    print('Average Cover', results.mean(), f1_results.mean())
    print('Median Cover', np.median(results), np.median(f1_results))



if __name__ == "__main__":
    main()
