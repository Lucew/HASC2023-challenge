import multiprocessing as mp
import os
import sys
from glob import glob

import threadpoolctl
import pandas as pd
from tqdm import tqdm

from changepoynt.algorithms.msst import MSST
from changepoynt.algorithms.messt import MESST


@threadpoolctl.threadpool_limits.wrap(limits=1, user_api='blas')
def process_data(input_tuple: tuple[str, str, int, str, str]):

    # unpack the input tuple
    data_path, output_folder, window_size, algorithm, method = input_tuple

    # get the signal name
    signal_name = os.path.splitext(os.path.split(data_path)[-1])[0]

    # get the change point detection method
    if algorithm == "MSST":
        transformer = MSST(window_size, method=method, use_fast_hankel=True)
    elif algorithm == "MESST":
        transformer = MESST(window_size, method=method, use_fast_hankel=True)
    else:
        raise NotImplementedError(f"Unknown algorithm", algorithm)

    # load the data from the memory
    df = pd.read_parquet(data_path)
    df = df.drop(['lat','lon','speed', 'x-mag', 'y-mag', 'z-mag'], axis=1, errors='ignore')

    # transform to numpy
    signal = df.to_numpy()

    # transform the data
    try:
        score = transformer.transform(signal)
    except AssertionError:
        return

    # save the score to disc
    score = pd.DataFrame(data=score, columns=["score"])
    score.to_parquet(os.path.join(output_folder, f"{signal_name}_{window_size}_{algorithm}_{method}.parquet"))


def main():

    # get the signals from the folder
    signals = glob(os.path.join("signals", "*.parquet"))

    # create the folder for the scores
    os.mkdir("scores")

    # make a list of methods
    methods = {"MESST": ["rsvd"], "MSST": ["ika", "rsvd", "weighted"]}
    window_sizes = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500]

    # create the iterator for the processing
    iterator =  list((signal, "scores", window_size, algo, method) for signal in signals for window_size in window_sizes for algo in methods.keys() for method in methods[algo])

    # orchestrate the transformation
    with mp.Pool(mp.cpu_count()//2) as pool:
        for _ in tqdm(pool.imap_unordered(process_data, iterator), desc="Processing Signals", total=len(iterator)):
            pass


if __name__ == "__main__":
    main()
