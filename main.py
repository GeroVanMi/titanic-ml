# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np   # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from sklearn.ensemble import RandomForestClassifier

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

data_path = './assets/data'

for dirname, _, filenames in os.walk(data_path):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/)
# that gets preserved as output when you create a  version using "Save & Run All"
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session

train_data = pd.read_csv(data_path + "/train.csv")
test_data = pd.read_csv(data_path + "/test.csv")


# women = train_data.loc[train_data.Sex == 'female']["Survived"]
# rate_women = sum(women)/len(women) * 100
#
# men = train_data.loc[train_data.Sex == 'male']['Survived']
# rate_men = sum(men)/len(men) * 100
#
# print("% of women who survived:", rate_women)
# print("% of men who survived:", rate_men)


target_array = train_data["Survived"]

features = ["Pclass", "Sex", "SibSp", "Parch"]
train_data_dummies = pd.get_dummies(train_data[features])
test_data_dummies = pd.get_dummies(test_data[features])

model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=1)
model.fit(train_data_dummies, target_array)
predictions = model.predict(test_data_dummies)

output = pd.DataFrame({'PassengerId': test_data.PassengerId, 'Survived': predictions})
output.to_csv(data_path + '/submission.csv', index=False)
print("Your submission was successfully saved!")
