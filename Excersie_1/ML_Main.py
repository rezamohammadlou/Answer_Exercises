from ML_Excerxis1_Extraction import *
from ML_Transformation import *

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
loan_data=change_data_types(loan_data)
display_data(loan_data.dtypes)

#One hot encoder
loan_data=one_hot_encoder(loan_data,["loan_type"])
display_data(loan_data)

#Lable Encoding
# lable_encoding(loan_data,['loan_type'])
# display_data(loan_data)

#حذف ستون های مورد نظر
drop_columns(loan_data,['loan_id'])

#گسسته سازی
K_bins_discretizer(loan_data,['client_id'])
display_data(loan_data)

#مقیاس بندی (نرمال سازی)
print(loan_data.columns)
min_max_scaler(loan_data,[ 'loan_amount'])