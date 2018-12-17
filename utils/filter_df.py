import os
import multiprocessing as mp
import argparse

import numpy as np
import pandas as pd
import cv2

def worker(path):
    global size
    img = cv2.imread(path)
    return (img.shape[0] < size or img.shape[1] < size)

def main(args):
    global size
    input_path = os.path.join(os.environ["data"], "humpback-whale-identification", "train.csv")
    output_path = os.path.join(os.environ["data"], "humpback-whale-identification", args.output_name)
    size = args.size
    df = pd.read_csv(input_path)
    image_paths = [os.path.join("dataset", "train", name) for name in df.Image]
    pool = mp.Pool(processes=10)
    df_filter = pool.map(worker, image_paths)
    breakpoint()
    assert df.shape[0] == len(df_filter)
    df_filter = np.array(df_filter)
    df = df[~df_filter].reset_index(drop=True)
    df.to_csv(output_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CLI for filtering CSV file based on image size")
    parser.add_argument("-out", "--output_name", metavar="FILENAME", default='train_1.csv', type=str, help="filename for the output .csv file")
    parser.add_argument("-size", "--size", metavar="INT", default=120, type=int, help="threshold size")
    args = parser.parse_args()
    main(args)
