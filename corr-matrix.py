import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

file_path = 'abalone_dataset.csv'
df = pd.read_csv(file_path, delimiter=',')

X = df.drop(['sex', 'type'], axis=1)
y = df['type']

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

corr_matrix = X.corr()

fig, axs = plt.subplots(1, 1, figsize=(10, 8))

sns.heatmap(corr_matrix, annot=True, cmap='viridis', ax=axs)
axs.set_title('Matriz de Correlação')

plt.show()
