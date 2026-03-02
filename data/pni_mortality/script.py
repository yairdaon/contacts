import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from epiweeks import Week
from scipy.interpolate import interp1d


def clean_numeric(x):
    x = str(x).strip().replace(',', '')
    if x == 'Insufficient Data':
        return np.nan
    return float(x)

def trim(pp, n=3, trim_last=True):                                                                                        
    if trim_last:
        f = lambda x: x.iloc[:-n]
    else:
        f = lambda x: x.iloc[n:]
    return pp.sort_values(['state', 'date']).groupby('state').apply(f).reset_index(drop=True)                                                                  

## from https://wonder.cdc.gov/controller/datarequest/D178;jsessionid=C26CAC4920B0B8609983D92A2AA5
pop = pd.read_csv("pops.csv").drop(['Notes', 'Yearly July 1st Estimates Code'], axis=1)
pop.columns = ['date', 'state', 'state code', 'population']
pop['date'] = pd.to_datetime({'year': pop['date'], 'month': 7, 'day': 1})

## Data from https://gis.cdc.gov/grasp/fluview/mortality.html
df = pd.read_csv('pni_deaths.csv')
df = df.query("geoid == 'State' and age == 'All'")
df['date'] = df['MMWR Year/Week'].apply(
    lambda x: Week(int(x) // 100, int(x) % 100).startdate()
)
df['deaths'] = df['Deaths from pneumonia and influenza'] 
df = df[['date', 'State', 'deaths']]
df.columns = ['date', 'state', 'deaths']
df = trim(df, trim_last=True)


## Data from https://healthdata.gov/cdc/Deaths-from-Pneumonia-and-Influenza-P-I-and-all-de/xvnn-3pyb
aa = pd.read_csv("pnis.csv")
aa['date'] = aa.apply(
    lambda row: Week(
        int(row['SEASON'].split('-')[0]) + (0 if row['WEEK'] >= 40 else 1),
        row['WEEK']
    ).startdate(),
    axis=1
)
aa['NUM INFLUENZA DEATHS'] = aa['NUM INFLUENZA DEATHS'].apply(clean_numeric)
aa['NUM PNEUMONIA DEATHS'] = aa['NUM PNEUMONIA DEATHS'].apply(clean_numeric)
aa['deaths'] = aa['NUM INFLUENZA DEATHS'] + aa['NUM PNEUMONIA DEATHS']
aa = aa[['date', 'SUB AREA', 'deaths']]
aa.columns = ['date', 'state', 'deaths']
aa = trim(aa, trim_last=False)


fig, axes = plt.subplots(4, 1, figsize=(12, 8), sharex=True)

for ax, state in zip(axes, ['New York', 'California', 'Illinois', 'Massachusetts']):
    dff = df.query("state == @state").sort_values('date')
    aaa = aa.query("state == @state").sort_values('date')

    dff = dff[dff['date'].apply(lambda x: 2017 <= x.year and x.year <= 2020)]          
    aaa = aaa[aaa['date'].apply(lambda x: 2017 <= x.year and x.year <= 2020)]          
    
    ax.plot(dff['date'], dff['deaths'], label='df (pni_deaths.csv)', alpha=0.8)
    ax.plot(aaa['date'], aaa['deaths'], label='aa (pnis.csv)', alpha=0.8)
    ax.set_ylabel('Deaths')
    ax.set_title(state)
    ax.legend()

axes[-1].set_xlabel('Date')
plt.tight_layout()
plt.close('all')

# merged = aaa.merge(dff, on=['state', 'date'], how='inner', suffixes=('_aa', '_df'))
# merged['err'] = merged['deaths_aa'] - merged['deaths_df']

comb = pd.concat([df, aa]).groupby(['state', 'date'], as_index=False)['deaths'].mean()                                                              
comb['date'] = pd.to_datetime(comb.date)

res = []
for state, pop_data in pop.groupby('state'):
    pni_data = comb.query("state == @state").copy()
    interpolant = interp1d(pop_data.date.astype(int), pop_data.population, kind='linear', fill_value='extrapolate')
    pni_data['population'] = interpolant(pni_data.date.astype(int))
    res.append(pni_data)

res = pd.concat(res).reset_index(drop=True)
res.to_csv("output.csv")
