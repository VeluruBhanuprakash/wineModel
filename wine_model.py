
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


df = pd.read_csv("https://raw.githubusercontent.com/elleobrien/wine/master/wine_quality.csv")

scalar = StandardScaler()
X = df.drop(['quality'],axis=1)
X_scaled = scalar.fit_transform(X)
y=df['quality']

x_train, x_test, y_train, y_test = train_test_split(X_scaled , y , test_size = .20 , random_state = 144)
logr_liblinear = LogisticRegression(solver='saga',multi_class='multinomial',max_iter=1000)
logr_liblinear.fit(x_train,y_train)

train_score=logr_liblinear.score(x_train,y_train)
test_score=logr_liblinear.score(x_test,y_test)


with open("metrics.txt", 'w') as outfile:
        outfile.write("Training variance explained: %2.1f%%\n" % train_score)
        print("Training variance explained: %2.1f%%\n" % train_score)
        outfile.write("Test variance explained: %2.1f%%\n" % test_score)
        print("Test variance explained: %2.1f%%\n" % test_score)

y_pred = logr_liblinear.predict(x_test)
cm = confusion_matrix(y_test,y_pred)

plt.figure(figsize=(10,7))
sns.heatmap(cm,annot=True)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confussion.png",dpi=120)