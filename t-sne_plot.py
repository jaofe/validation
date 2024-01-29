import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

training_set = pd.read_csv("abalone_dataset.csv")

sex_mapping = {'I': 0, 'F': -1, 'M': 1}
training_set['sex'] = training_set['sex'].map(sex_mapping)

X = training_set.drop('type', axis=1)  
Y = training_set['type'].values  

scaler = StandardScaler()
X_std = scaler.fit_transform(X)

tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_std)

fig, axs = plt.subplots(1, 1, figsize=(8, 6))

scatter = axs.scatter(X_tsne[:, 0], X_tsne[:, 1], c=Y, cmap='viridis', alpha=0.7)
axs.set_title('t-SNE Plot')

legend = axs.legend(*scatter.legend_elements(), title='Tipos:')
axs.add_artist(legend)

plt.show()
