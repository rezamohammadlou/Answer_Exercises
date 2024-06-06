from ML_Excerxis1_Extraction import *
from ML_Transformation import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def display_data(data):
    print(data)
    print(80*'*')

loan_data=extract_from_csv("fifth semester/ML_Exercises/Tmarin1/f_88e6654a-46e0-43bd-9d85-aa19e6dc7e7a/تمرین1/loans.csv")
display_data(loan_data)

display_data(loan_data.columns)
display_data(loan_data.dtypes)

# حذف رکورد کاملا خالی
drop_record_all_nans(loan_data)

# پرکردن فیلد های خالی
fillna(loan_data)

# تشخیص داده های پرت با رسم نمودار
# check_outlier_columns_by_plotly(loan_data,["loan_amount"])

#حذف داده های پرت
loan_data=remove_loan_amount_outliers(loan_data,8320,11745)
display_data(loan_data)
display_data(loan_data["loan_amount"].min())
display_data(loan_data["loan_amount"].max())

#تغییر فرمت داده ها
# loan_data=change_data_types(loan_data)
# display_data(loan_data.dtypes)

# #One hot encoder
# loan_data=one_hot_encoder(loan_data,["loan_type"])
# display_data(loan_data)

#Lable Encoding
lable_encoding(loan_data,['loan_type'])
display_data(loan_data)

#حذف ستون های مورد نظر
drop_columns(loan_data,['loan_id'])

#گسسته سازی
K_bins_discretizer(loan_data,['client_id'])
display_data(loan_data)

#مقیاس بندی (نرمال سازی)
print(loan_data.columns)
min_max_scaler(loan_data,[ 'loan_amount'])

print(loan_data['loan_amount'].skew())
plt.hist(x=loan_data['loan_amount'],bins='auto',alpha=0.9,rwidth=0.95,facecolor='magenta')
plt.show()
print(loan_data['rate'].skew())
plt.hist(x=loan_data['rate'],bins='auto',alpha=0.9,rwidth=0.95,facecolor='magenta')
plt.show()

# Create X feature and y Target
X=loan_data[['loan_type','rate']]
y=loan_data[['loan_amount']]
X=np.c_[np.ones((X.shape[0],1)),X]

# Split to TEST AND TRAIN
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#Create model
model=LinearRegression()

#Learning the model
model.fit(X_train,y_train)

#Using the model to prdict test data
y_pred=model.predict(X_test)

#Calculating the mean squared error (mse)
mse=mean_squared_error(y_test,y_pred)

#Calculating the R_Squared score (r2_score)
r2=r2_score(y_test,y_pred)

#show results
print(f"Mean Squared Error : {mse}")
print(f"R_Squared score : {r2}")
print(f"Model intercept: {model.intercept_}")
print(f"Model Coef : {model.coef_}")


