import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __name__ == "__main__":
    file_list = ["a/face_]Box.csv", "b/face_]Box.csv`"]
    names = ["improve", "baseline"]
    ap = ["", ""]
    plt.figure(figsize=[8, 8])
    for i in range(len(file_list)):
        pr_data = pd.read_csv(file_list[i], header=None)
        recall, precision = np.array(pr_data[0]), np.array(pr_data[1])

        plt.plot(recall, precision, label=names[i], ap=ap[i])
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"b/{names[i]}_pr.png")
