import os
import pandas as pd


def get_mlrrun_metrics():
    models_path = os.getcwd() + '/mlruns/241478715816179234/models'  # go to file containing each model


    df = pd.DataFrame({"accuracy": [None],
                       "recall": [None],
                       "roc_auc": [None],
                       "train_time": [None],
                       "precision": [None],
                       "f1": [None]})
    model_num = 0
    for model in os.listdir(models_path): # search through each model folder
        model = models_path + '/' + model + '/metrics' # add metric directory onto model path
        metrics = []

        for metric_folder in os.listdir(model):  # go through each metric file
            if metric_folder == df.columns[len(metrics)]:       # len(metrics) is index -- checks if the folder is in the same order as the df
                metric_folder_path = model + '/' + metric_folder
                value = float([line.split() for line in open(metric_folder_path)][0][1])  # go word by word in file but only grab the value
                metrics.append(value)   # append to the
            else:
                print("Folder not found")


        df.loc[model_num] = metrics
        model_num += 1


    print("\nFinal metrics:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    get_mlrrun_metrics()