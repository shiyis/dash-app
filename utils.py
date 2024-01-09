from arcgis.gis import GIS
from arcgis.geocoding import geocode, reverse_geocode
from arcgis.geometry import Point
import pandas as pd

df = pd.read_csv('./data/2022/processed_weball.csv')

addresses = df[['Street one','Street two','City or town','ZIP code']].astype(str).agg(' '.join, axis=1)

gis = GIS()


for n, i in enumerate(addresses):
    print(i)
    geocode_result = geocode(address=i) 
    if geocode_result:
        df.iloc[n,list(df.columns).index('lat')] = geocode_result[0]['location']['y']
        df.iloc[n,list(df.columns).index('lon')] = geocode_result[0]['location']['x']
    else:
        pass

df.to_csv('./data/2022/processed_weball_updated_address.csv')