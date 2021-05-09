#%%
try:
    import os
    import numpy as np
    import matplotlib.pyplot as plt
    import pandas as pd
    from sklearn.preprocessing import MinMaxScaler
    plt.style.use("ggplot")
    print("----Libraries Loaded----")
except:
    print("----Libraries Not Loaded----")


#%%
os.listdir()

# %%
# Reading Data

df = pd.read_csv('BTC-INR.csv')
df.head()

# %%
# Data Pre-processing

df = df.set_index("Date")[['Close']].tail(1000)
df = df.set_index(pd.to_datetime(df.index))

#%%
# Normalizing/Scaling the Data
scaler = MinMaxScaler()
df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns, index=df.index)
