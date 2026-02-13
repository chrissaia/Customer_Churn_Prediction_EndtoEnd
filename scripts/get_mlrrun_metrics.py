import os
import pandas as pd



models_path = os.getcwd() + '/../mlruns/241478715816179234/models'  # go to file containing each model


df = pd.DataFrame()
model_num = -1


for model in os.listdir(models_path): # search through each model folder
    model = models_path + '/' + model + '/metrics' # add metric directory onto model path
    model_num += 1
    metrics = []

    for metric_folder in os.listdir(model):  # go through each metric file
        if metric_folder not in df.columns:
            df[metric_folder] = None

        metric_folder_path = model + '/' + metric_folder
        value = [line.split() for line in open(metric_folder_path)][0][1]  # go word by word in file but only grab the value
        metrics.append(value)

    print(df)
    df.loc[model_num] = metrics
    model_num += 1


print(df)
df.head()


'''

print("\nFinal metrics:")
for k, v in metrics.items():
    print(f"  {k}: {v:.4f}")
'''