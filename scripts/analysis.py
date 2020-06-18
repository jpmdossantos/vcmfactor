import pandas as pd
pd.plotting.register_matplotlib_converters()
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def data_load():
    df = pd.read_csv("vcms.csv")
    df['Identifier'] = df.Serial + " " + df.Amostra.astype(str)
    return df

def extract_new_column(data, column, extra = 0):
    for id in data.Identifier.unique():
        selected = data.Identifier == id
        base_vcm = data.loc[(selected) & (data.Ganho==30), 'VCM'].astype(float).values
        if column == "factor":
            data.loc[selected,'Fator VCM'] =  base_vcm / data.loc[selected,'VCM'] 
        elif column == "error":
            data.loc[selected,'Error']=calculate_error(data.loc[selected], extra, base_vcm)
    
def bootstrap(array):
    B_repeats = 10000
    replicates = []
    for _ in range(B_repeats):
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
        
    result = pd.DataFrame({'Ganho': x, 'Central': y,
                           'Lower': ylower, 'Upper':yupper}) 
    return(result)

def plot_bootstrap(boot_df):
    plt.plot(boot_df.Ganho,boot_df.Central)
    plt.fill_between(boot_df.Ganho,boot_df.Lower,boot_df.Upper,alpha=.5)
    plt.show()

def plot_raw(data, xname, yname):
    sns.set_style("darkgrid")
    plt.figure(figsize=(12,6))   
    sns.scatterplot(x=xname, y=yname, data=data,hue='Identifier')
    plt.legend(loc='upper right')
    
    sns.lineplot(x=xname, y=yname,data=data)
    
#   sns.lmplot(x='Ganho', y='Fator VCM',data=data,hue='Identifier',size=(7),aspect=(12.0/7),legend=False)
#   plt.legend(loc='upper left')

    plt.show()

def plot_error(data, xname, yname):  
    sns.set_style("darkgrid")
    plt.figure(figsize=(12,6))   
    plt.ylim(-0.5,6)
    #for iden in data.Serial.unique():
    #    selected = data.loc[data.Serial==iden]
    #    sns.lineplot(x=xname, y=yname,data=selected)
            
    sns.scatterplot(x=xname, y=yname, data=data)#,hue='Serial')
    ax = sns.lineplot(x=xname, y=yname,data=data,color='r')
    plt.title("Erro médio absoluto do VCM obtido utilizando os fatores por ganho")
    ax.set(xlabel='Ganho', ylabel='Erro (fL)')
    plt.legend(loc='upper left')
    plt.show()
    

def calculate_error(df, boot_result, base_vcm):
    errors = []
    for index, row in df.iterrows():
        error = boot_result.loc[boot_result.Ganho==row.Ganho,'Central']*row.VCM - base_vcm
        errors.append(abs(error.values[0]))
    error_series = pd.Series(errors, dtype=float, index=df.index)

    return error_series
        
        
    
    

bs = True
    
measures = data_load()
measures = measures.dropna()
extract_new_column(measures, "factor")
if bs:
    boot_df = create_samples_bootstrap(measures)
#    plot_bootstrap(boot_df)
    extract_new_column(measures, "error", boot_df)
    plot_error(measures,'Ganho','Error')
    
else:
    plot_raw(measures,'Ganho', 'Fator VCM')


#sns.scatterplot(x=data.a, y=data.w)
#sns.regplot(x=data.a, y=data.t)
#sns.barplot(x=data.a,y=data.t)
#sns.boxplot(x=data.a,y=data.w)
#sns.jointplot(x=data.a, y=data.w, kind="kde")
#sns.heatmap(data=data[['a','w']])
