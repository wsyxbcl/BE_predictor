import pandas as pd
elements = ['H', 'Li', 'B', 'C', 'N', 'O', 'F', 'Na', 'Mg', 'Al',\
'Si', 'P', 'S', 'Cl']
df = pd.read_csv('./sourcedata.csv', header = None)
for i in range(14):
    x = range(200)
    x.append(200 + i)
    df_original = df.ix[:, x]
    filename = 'data_'+elements[i]+'.csv'
    #locals()['df_%s' %elements[i]] = df_original.dropna()
    df_cleaned = df_original.dropna()
    df_cleaned.to_csv(filename, index = False, header = False)