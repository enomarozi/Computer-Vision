import pandas as pd
df = pd.DataFrame({'No':['1','2','3'],'Nama':['Eno','Doni','Jack'],'Jam':[1,3,11]})
print(df.head())
print('\n\n\n')
for i,j in df.iterrows():
    if j['Nama'] == 'Doni':
        df.replace(to_replace = j['Jam'],
                   value='100',
                   inplace=True)
print(df.head())
df.to_csv('file.csv')
