import os
import multiprocessing as mp
from glob import glob

from tqdm import tqdm
import threadpoolctl
import pandas as pd

from claspy.segmentation import BinaryClaSPSegmentation


@threadpoolctl.threadpool_limits.wrap(limits=1, user_api='blas')
def process_data(input_tuple: tuple[str, str]):

    # unpack the input tuple
    data_path, output_folder = input_tuple

    # get the signal name
    signal_name = os.path.splitext(os.path.split(data_path)[-1])[0]

    # load the data from the memory
    df = pd.read_parquet(data_path)
    df = df.drop(['lat','lon','speed', 'x-mag', 'y-mag', 'z-mag'], axis=1, errors='ignore')

    # transform to numpy
    signal = df.to_numpy()

    # get the clasp detection
    clasp = BinaryClaSPSegmentation()
    detections = clasp.fit_predict(signal)

    # save the score to disc
    score = pd.DataFrame(data=detections, columns=["detections"])
    score.to_parquet(os.path.join(output_folder, f"{signal_name}_{window_size}_{algorithm}_{method}.parquet"))


def main():

    # get the signals from the folder
    signals = glob(os.path.join("signals", "*.parquet"))

    # create the iterator for the processing
    iterator =  list((signal, "clasp") for signal in signals)

    # orchestrate the transformation
    with mp.Pool(mp.cpu_count()//2) as pool:
        for _ in tqdm(pool.imap_unordered(process_data, iterator), desc="Processing Signals", total=len(iterator)):
            pass


if __name__ == "__main__":
    main()
