import collections
import os
from glob import glob
import collections

from tqdm import tqdm
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

import evaluate_results as er
import transform_data as td


# transform nested defaultdict into dict
def transform_defauldict(input_dict: collections.defaultdict):

    # return if no defaultdict
    if not isinstance(input_dict, collections.defaultdict):
        return input_dict

    # transform the current
    input_dict = dict(input_dict)
    for key, val in input_dict.items():
        input_dict[key] = transform_defauldict(val)
    return input_dict


@st.cache_data
def load_data(score_path: str = "scores"):

    # load the groundtruth
    gtdf = er.load_master_data(r"has2023_master.csv.zip")

    # load the score files
    score_files = {tuple(os.path.splitext(os.path.split(file)[-1])[0].split('_')): pd.read_parquet(file) for file in
                   tqdm(glob(os.path.join("scores3", "*.parquet")), desc='Loading Scores')}

    # reorder the files into a dictionary
    tmp_score_files = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(dict)))
    for (idx, ws, algorithm, method), score in score_files.items():
        # score[score > 0.99] = 1-score[score > 0.99]
        tmp_score_files[idx][algorithm][method][ws] = score
    score_files = transform_defauldict(tmp_score_files)

    # load the full file
    print("Loading ground truth data")
    df = td.load_master_data('./has2023_master.csv.zip')
    return score_files, gtdf, df


# -----------------------------
# Plotting
# -----------------------------

def plot_scores_with_ground_truth(
    scores: pd.DataFrame,
    ground_truth: pd.DataFrame,
    title: str,
):
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            y=scores["score"],
            mode="lines",
            name="Detection score",
        )
    )

    for ele in ground_truth:

        fig.add_vline(
            x=ele,
            line_width=2,
            line_dash="dash",
            annotation_text="",
            annotation_position="top",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Score",
        hovermode="x unified",
        height=550,
    )

    return fig


def plot_signals_with_ground_truth(
    scores: pd.DataFrame,
    ground_truth: pd.DataFrame,
    title: str,
):
    fig = go.Figure()
    for signame in ['x-acc', 'y-acc', 'z-acc', 'x-gyro', 'y-gyro', 'z-gyro', 'x-mag', 'y-mag', 'z-mag', 'lat', 'lon', 'speed']:
        if not len(scores[signame]):
            continue
        fig.add_trace(
            go.Scatter(
                y=(scores[signame]-scores[signame].min())/(scores[signame].max()-scores[signame].min()),
                mode="lines",
                name=signame,
            )
        )

    for ele in ground_truth:

        fig.add_vline(
            x=ele,
            line_width=2,
            line_dash="dash",
            annotation_text="",
            annotation_position="top",
        )

    fig.update_layout(
        title=title,
        xaxis_title="Time",
        yaxis_title="Signals",
        hovermode="x unified",
        height=550,
    )

    return fig


# -----------------------------
# Streamlit app
# -----------------------------

st.set_page_config(
    page_title="Detection Score Viewer",
    layout="wide",
)


def main():
    st.title("Detection Score Viewer")

    with st.sidebar:
        st.header("Selection")

        # get the data
        score_dict, ground_truth, signal_df = load_data(score_path="scores")


        signal = st.selectbox(
            "Signal",
            score_dict.keys(),
        )
        print(signal)

        algorithm = st.selectbox(
            "Algorithm",
            score_dict[signal].keys(),
        )

        method = st.selectbox(
            "Algorithm",
            score_dict[signal][algorithm].keys(),
        )

        window_size = st.selectbox(
            "Window size",
            score_dict[signal][algorithm][method].keys(),
        )

        show_raw_data = st.checkbox("Show raw data", value=False)


    scores = score_dict[signal][algorithm][method][window_size]
    if st.checkbox("Apply log10", value=False):
        window_size = int(window_size)
        offset = max(np.quantile(scores[int(1.5*window_size):-window_size*5//6], 0.1), 1e-20)
        print(offset)
        scores = (np.log10(scores + offset) - np.log10(offset)) / (np.log10(1 + offset) - np.log10(offset))

    print(ground_truth.iloc[int(signal), :]['change_points'])
    ground_truth = signal_df.iloc[int(signal), :]['change_points']
    print(ground_truth)

    st.subheader(
        f"{method} / {algorithm} / window={window_size} / {signal}"
    )

    if scores.empty:
        st.warning("No detection scores found for this selection.")
    else:
        fig = plot_scores_with_ground_truth(
            scores=scores,
            ground_truth=ground_truth,
            title="Detection Scores with Ground Truth Events",
        )

        st.plotly_chart(fig, width='stretch')

    if show_raw_data:
        fig = plot_signals_with_ground_truth(
            scores=signal_df.iloc[int(signal), :],
            ground_truth=ground_truth,
            title="Signals with Ground Truth Events",
        )

        st.plotly_chart(fig, width='stretch')

if __name__ == "__main__":
    main()