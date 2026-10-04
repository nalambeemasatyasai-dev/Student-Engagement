import matplotlib
matplotlib.use('Agg')
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os
import matplotlib.pyplot as plt

def scalingDF(df):
  # scaler = QuantileTransformer(output_distribution='normal')
  scaler = MinMaxScaler()
  scaled = scaler.fit_transform(df)
  df_s = pd.DataFrame(scaled,index=df.index, columns=df.columns)
  return df_s

import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)
BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))

data_path = os.path.join(BASE_DIR, "WACV data") # dynamically resolved data path
result_base_path = BASE_DIR
os.makedirs(os.path.join(result_base_path, "Results", "AU_features"), exist_ok=True)

df0 = pd.read_csv(os.path.join(data_path,"merged_data0.csv"))
df1 = pd.read_csv(os.path.join(data_path,"merged_data1.csv"))
df2 = pd.read_csv(os.path.join(data_path,"merged_data2.csv"))

df00 = df0.loc[df0['confidence'] >= 0.7]
df11 = df1.loc[df1['confidence'] >= 0.7]
df22 = df2.loc[df2['confidence'] >= 0.7]

#concatenate all the data files (disengaged[0], partially engaged[1] and engaged[2])
df = pd.concat([df00,df11,df22])

df = df.sample(frac=1)
df_x = df.loc[:,"x0":"AU45_c"] #extract the AU columns from the dataframe
df_y = df.loc[:,"Label_y"]

df = pd.concat([scalingDF(df_x),df_y],axis=1)

#Conditional Probability Mapping of AUs with Engagement Labels. 
from AU_mapping import AU_mapping
map = AU_mapping()
fig, df_mapAU = map.au_heatmap(df)
fig.savefig(os.path.join(result_base_path,"Results/AU_features/CondProb_AU_mapping.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(result_base_path,"Results/AU_features/CondProb_AU_mapping.png"), dpi=150, bbox_inches='tight')

#Relative Activation Ratios for AUs across Engagement Labels.
from AU_mapping_relative import AU_mapping
map = AU_mapping()
fig, df_mapR = map.au_heatmap(df)
fig.savefig(os.path.join(result_base_path,"Results/AU_features/Relative_AU_mapping.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(result_base_path,"Results/AU_features/Relative_AU_mapping.png"), dpi=150, bbox_inches='tight')


#Log-Ratio Statistical Discriminative Coefficient (SDC) Scores for AUs.
from AU_mapping_SDC import AU_mapping
map = AU_mapping()
fig, df_mapC = map.au_heatmap(df)
fig.savefig(os.path.join(result_base_path,"Results/AU_features/SDC_Scores.pdf"), bbox_inches='tight')
fig.savefig(os.path.join(result_base_path,"Results/AU_features/SDC_Scores.png"), dpi=150, bbox_inches='tight')



#Conditional Probability Mapping of AUs with Engagement Labels Line Plot
plt.figure(figsize=(12,5))
for index, row in df_mapAU.T.iterrows():
    plt.plot(row, label=index, marker='o', linewidth=1.5)
plt.title('Conditional Probability Mapping of Facial Action Units', fontsize=14, fontweight='bold')
plt.xlabel('Action Units (AUs)', fontsize=12)
plt.ylabel('Conditional Probability', fontsize=12)
plt.xticks(rotation=45)
plt.grid(True, linestyle='--', alpha=0.5)
plt.legend(fontsize=11)
plt.savefig(os.path.join(result_base_path,"Results/AU_features/CondProb_AU_mappingLinePlot.pdf"), bbox_inches='tight')
plt.savefig(os.path.join(result_base_path,"Results/AU_features/CondProb_AU_mappingLinePlot.png"), dpi=150, bbox_inches='tight')
print("Successfully generated all AU Mapping plots in Results/AU_features/ (.pdf and .png)")

