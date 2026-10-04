import matplotlib
matplotlib.use('Agg')
from ML_classification import ML_classification
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import resample
import os
def scalingDF(df):
  # scaler = QuantileTransformer(output_distribution='normal')
  scaler = MinMaxScaler()
  scaled = scaler.fit_transform(df)
  df_s = pd.DataFrame(scaled,index=df.index, columns=df.columns)
  return df_s


import sys
import os
import argparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

parser = argparse.ArgumentParser(description="Train classical ML models for student engagement detection")
parser.add_argument("--data_dir", type=str, default=os.path.join(BASE_DIR, "WACV data"), help="Path to WACV data directory")
parser.add_argument("--output_dir", type=str, default=os.path.join(BASE_DIR, "Results"), help="Path to save results and plots")
parser.add_argument("--model_dir", type=str, default=os.path.join(SCRIPT_DIR, "trained_models"), help="Path to save trained models")
args, _ = parser.parse_known_args()

data_path = os.path.abspath(args.data_dir)
path_trainedmodel = os.path.abspath(args.model_dir)
results_path = os.path.abspath(args.output_dir)

os.makedirs(path_trainedmodel, exist_ok=True)
os.makedirs(results_path, exist_ok=True)

df0 = pd.read_csv(os.path.join(data_path,"merged_data0.csv"))
df1 = pd.read_csv(os.path.join(data_path,"merged_data1.csv"))
df2 = pd.read_csv(os.path.join(data_path,"merged_data2.csv"))

df00 = df0.loc[df0['confidence'] >= 0.7]
df11 = df1.loc[df1['confidence'] >= 0.7]
df22 = df2.loc[df2['confidence'] >= 0.7]

#Resampling for treating the class imabalnce problem
n = min(len(df00),len(df11),len(df22))
df1_ds = resample(df11, replace=True, n_samples=n, random_state=42)
df2_ds = resample(df22, replace=True, n_samples=n, random_state=42)

df = pd.concat([df00,df1_ds,df2_ds])

df = df.sample(frac=1)
df_x = df.loc[:,"x0":"AU45_c"] #extract the AU columns from the dataframe
df_y = df.loc[:,"Label_y"]

df = pd.concat([scalingDF(df_x),df_y],axis=1)

#create an object for ML_classification
obj = ML_classification()

'''Generate the training data for the four feature sets: 
    S1(facial landmark feature)
    S2(Eye gaze and head pose features)
    S3(AU features)
    S4(Combined features)'''

training_data=obj.feature_sets(df)

dfML_result=[]
roc_plot=[]

#train the feature sets with our classifiction models
for i in range(4):
  temp , plot= obj.classifier_result(training_data[i],i,path_trainedmodel)
  print(temp)
  #save the result in dfML_result dataframe
  dfML_result.append(pd.DataFrame(temp,columns=["model","Accuracy","Precision","Recall","F-measure"],index=["Random forest","XGBoost","Decision tree","SVM","Gradient Boosting"]).round(decimals=2))
  roc_plot.append(plot)

  #save the results and the plots
for i in range(4):
    dfML_result[i].to_csv(os.path.join(results_path, f"result_s{i}.csv"))
    roc_plot[i].savefig(os.path.join(results_path, f"oc_plot_s{i}.pdf"), bbox_inches='tight')
    roc_plot[i].savefig(os.path.join(results_path, f"oc_plot_s{i}.png"), dpi=150, bbox_inches='tight')
print(f"Classical ML results and ROC plots saved successfully in {results_path}")