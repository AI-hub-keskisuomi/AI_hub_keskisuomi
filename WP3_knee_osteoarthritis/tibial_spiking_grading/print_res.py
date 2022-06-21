""" Simple script that prints the hyperparameter search results in descending order """

import argparse
import os
import pandas as pd


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", default="", help="Experiment directory path")
    args = parser.parse_args()

    # experiment source path
    exp_dir = args.s

    # read experiment results
    df = pd.read_csv(os.path.join(exp_dir, "results.csv"))

    # print results sorted
    print(df.sort_values(by="val_acc", ascending=False))


if __name__ == "__main__":
    main()
