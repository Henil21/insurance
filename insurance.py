import pandas as pd
import tensorflow as  tf
import matplotlib.pyplot as plt
from sklearn.compose import make_column_transformer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split


# readind insurance 
insurance=pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/master/insurance.csv")
print(insurance)

# transfering column
ct=make_column_transformer(
    (MinMaxScaler(),["age","bmi","children"]) ,#turn all the data in 0 and 1
    (OneHotEncoder(handle_unknown="ignore"),["sex","smoker","region"])
)
X=insurance.drop("charges",axis=1)
y=insurance["charges"]

X_train, X_test, y_train, y_test = train_test_split(X, 
                                                    y, 
                                                    test_size=0.2, 
                                                    random_state=42) # set random state for reproducible splits

# fit the column transformer to our traning data
ct.fit(X_train)
# transform traning and test data with normalization and onehot encoding
x_train_normal=ct.transform(X_train)
x_test_normal=ct.transform(X_test)


tf.random.set_seed(42)
final_model=tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1),
])
# final_model.compile(tf.keras.losses.mae,
#          optimizer=tf.keras.optimizers.Adam(),
#          metrics=["mae"])
final_model.compile(loss=tf.keras.losses.mae,
                          optimizer=tf.keras.optimizers.Adam(0.01),
                          metrics=['mae'])

# fitting
final_model.fit(x_train_normal,y_train,epochs=300)