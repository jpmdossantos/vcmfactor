import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def data_load():
    df = pd.read_csv("vcms.csv")
    df['Identifier'] = df.Serial + " " + df.Amostra.astype(str)
    return df

def extract_factor(data):
    for id in data.Identifier.unique():
        print(id)
        selected = data.Identifier == id
        base_vcm = data.loc[(selected) & (data.Ganho==30), 'VCM'].astype(float).values
        data.loc[selected,'Fator VCM'] =  base_vcm / data.loc[selected,'VCM'] 
        
def plot_raw(data):
    sns.set_style("darkgrid")
    plt.figure(figsize=(12,6))   

    sns.scatterplot(x='Ganho', y='Fator VCM',data=data,hue='Identifier')
    plt.legend(loc='upper right')
    
    sns.lineplot(x='Ganho', y='Fator VCM',data=data)
    
#   sns.lmplot(x='Ganho', y='Fator VCM',data=data,hue='Identifier',size=(7),aspect=(12.0/7),legend=False)
#   plt.legend(loc='upper left')

    plt.show()
    
def bootstrap(array):
    B_repeats = 100
    replicates = []
    for _ in range(1000):
        boot_factor = np.random.choice(array,len(array))
        m = np.median(boot_factor)
        replicates.append(m)
    confidence = np.percentile(replicates, [2.5,97.5])
    result = [confidence[0], np.median(replicates), confidence[1]]
    return result
    
def create_samples_bootstrap(df):
    x = []
    y = []
    ylower = []
    yupper = []

    for ganho in df.Ganho.unique():
        x.append( ganho )
        bootstrapped = bootstrap( df.loc[df.Ganho==ganho, 'Fator VCM'] ) 
        y.append(bootstrapped[1])
        ylower.append(bootstrapped[0])
        yupper.append(bootstrapped[2])
        
    plot_bootstrap(x,y,ylower,yupper)

def plot_bootstrap(x, central, lower, upper):
    plt.plot(x,central)
    plt.fill_between(x,lower,upper,alpha=.5)
    plt.show()

bs = False     
    
measures = data_load()
measures.dropna()
extract_factor(measures)
if bs:
    create_samples_bootstrap(measures)
else:
    plot_raw(measures)


#sns.scatterplot(x=data.a, y=data.w)
#sns.regplot(x=data.a, y=data.t)
#sns.barplot(x=data.a,y=data.t)
#sns.boxplot(x=data.a,y=data.w)
#sns.jointplot(x=data.a, y=data.w, kind="kde")
#sns.heatmap(data=data[['a','w']])
