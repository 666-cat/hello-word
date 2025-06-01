import pandas as pd
import numpy as np
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
med=pd.read_csv('/Users/weiliangyu/Downloads/Medicine_Details.csv')
print(med.columns)
med.drop_duplicates(inplace=True)
encoder_medicine=LabelEncoder()
encoder_manufacturer=LabelEncoder()
med['Medicine']=encoder_medicine.fit_transform(med['Medicine'])
med['Manufacturer']=encoder_manufacturer.fit_transform(med['Manufacturer']) 
print(med.head())
med.info()
print('重复行数:',med.duplicated().sum())
print('缺失值:',med.isnull().sum())
sns.set_style('darkgrid')
fig,ax=plot.subplots(1,3,figsize=(15,5))
sns.histplot(med['Excellent  Review %'],kde=True,ax=ax[0],color='green')
ax[0]=set_title('Distribution of Excellent Review %')
sns.histplot(med['Average  Review %'],kde=True,ax=ax[1],color='green')
ax[1]=set_title('Distribution of Excellent Review %')
sns.histplot(med['Poor  Review %'],kde=True,ax=ax[2],color='green')
ax[2]=set_title('Distribution of Excellent Review %')
plt.show()
manufacturer_counts=med['manufacturer'].value_counts()
print(manufacturer_counts.head(10))

fig=px.bar(manufacturer_counts,x=manufacturer_counts.index,y=manufacturer_counts,title='number of medicines by manufacturer')
fig.update_xaxes(tickangle=45)
fig.update_layout(xaxis_title='manufacturer',yaxis_title='number of medicines')
fig.show()
from mol_toolkits.mplot3d import Axes3D
encoder_composition=LabelEncoder()
med['Composition_encoded']=encoder_composition.fit_transform(med['Composition'])
fig=plt.figure(figsize=(12,12))
ax=fig.add_subplot(111,projection='3d')
sc=ax.scatter(med['Excellent  Review %'],med['Average  Review %'],med['Poor  Review %'],c=med['Composition_encoded'],marker='o')