import matplotlib.pyplot as plt
import seaborn as sns

confusion_mtx = [[810, 225, 43], [179, 560, 264], [52, 288, 711]]

fig, axs = plt.subplots(1, 1, figsize=(6, 6))

xticklabels = ['1', '2', '3']
yticklabels = ['1', '2', '3']

sns.heatmap(confusion_mtx, annot=True, fmt='.0f', cmap='viridis', ax=axs, xticklabels=xticklabels, yticklabels=yticklabels)
axs.set_title('Matriz de Confusão')

axs.set_xlabel('Classes Previstas')
axs.set_ylabel('Classes Reais')

plt.show()