import os
import pathlib

import numpy as np
import pandas as pd
from urllib.request import urlretrieve

def download_from_github(repo_url: str = "https://github.com/ermshaua/multivariate-clasp/raw/refs/heads/main/datasets", file_path: str = "has2023_master.csv.zip"):
    url = f"{repo_url}/{file_path}"
    output_path = pathlib.Path.cwd() / pathlib.Path(file_path).name
    urlretrieve(url, output_path)
    return output_path


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
    np_cols = ["change_points", "activities", "x-acc", "y-acc", "z-acc",
               "x-gyro", "y-gyro", "z-gyro",
               "x-mag", "y-mag", "z-mag",
               "lat", "lon", "speed"]
    converters = {
        col: lambda val: np.array([]) if len(val) == 0 else np.array(eval(val)) for col
        in np_cols}
    return pd.read_csv(data_path, converters=converters, compression="zip")

def main():

    # download the file
    filepath = download_from_github()

    # load the data into memory
    df = load_master_data(filepath)

    # make a folder with individual time series
    os.mkdir("signals")

    # define the channels we want to use
    channels = ['x-acc', 'y-acc', 'z-acc', 'x-gyro', 'y-gyro', 'z-gyro', 'x-mag', 'y-mag', 'z-mag', 'lat', 'lon',
                'speed']

    # go through the data and save the signals
    for idx in range(df.shape[0]):

        # get the info from one time series
        ts = df.iloc[idx, :]

        # extract the data to a list of numpy arrays
        multivariate = ts[channels].to_numpy()

        # sort out unfilled time series
        multivariate, tmpchan = zip(*[(ele, channel) for ele, channel in zip(multivariate, channels) if ele.shape[0] > 0])

        # create one combined array
        multivariate = np.array(multivariate)
        multivariate = multivariate.T

        # make dataframe
        tsdf = pd.DataFrame(multivariate, columns=tmpchan)
        tsdf.to_parquet(os.path.join("signals", f"{idx}.parquet"))


if __name__ == "__main__":
    main()
