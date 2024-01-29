import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt  # Add this line for matplotlib

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

file_path = 'abalone_dataset.csv'
df = pd.read_csv(file_path, delimiter=',')

X1 = df.drop(['sex', 'type'], axis=1)
y1 = df['type']

label_encoder = LabelEncoder()
y_encoded1 = label_encoder.fit_transform(y1)

scaler = StandardScaler()
X_std1 = scaler.fit_transform(X1)

X_std1_with_type = pd.DataFrame(X_std1, columns=X1.columns)
X_std1_with_type['type'] = y1

sns.set(style="ticks")
sns.pairplot(X_std1_with_type, hue='type', palette='viridis', height=2.5)

plt.show()
