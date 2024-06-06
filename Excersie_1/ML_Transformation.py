import pandas as pd 


# حذف رکورد کاملا خالی
def drop_record_all_nans(df):
    return df.dropna(how="all",axis=0)

# پرکردن فیلد های خالی
def fillna(data):
    data.fillna(value=
                    {'client_id':55555,
                     'loan_type':data.loan_type.mode()[0],
                     'loan_amount ':data.loan_amount.mean(),
                     'repaid':data.repaid.mode()[0],
                     'loan_id':10000,
                     'loan_start':'2022/0/01',
                     'loan_end':'2023/0/01',
                     'rate ':data.rate.mean(),
                     },inplace=True)
    return data

#تشخیص داده های پرت
import plotly.express as px
def check_outlier_columns_by_plotly(data,columns):
    fig=px.box(data,y=columns)
    fig.show()

#حذف داده های پرت
def remove_loan_amount_outliers(loan_data,min_L,max_L):
    df=pd.DataFrame(loan_data)
    loan_data=df[(df['loan_amount']>=min_L) & (df['loan_amount']<=max_L)]
    return loan_data

# #تغییر فرمت داده ها
# def change_data_types(loan_data):
#     loan_data.rate=round(loan_data.rate)
#     loan_data['loan_start']=pd.to_datetime(loan_data['loan_start'])
#     loan_data['loan_end']=pd.to_datetime(loan_data['loan_end'])
#     return loan_data

#One hot encoder
def one_hot_encoder(loan_data,columns):
    return pd.get_dummies(loan_data,columns=columns,dtype='int')

#Lable Encoding
from sklearn.preprocessing import LabelEncoder
def lable_encoding(loan_data,columns):
    le=LabelEncoder()
    for col in columns:
        loan_data[col]=le.fit_transform(loan_data[col])
    return loan_data

#حذف ستون های مورد نظر
def drop_columns(data,columns):
    for col in columns:
        data.drop(col,axis=1,inplace=True)
    return data

#گسسته سازی
from sklearn.preprocessing import KBinsDiscretizer 
def K_bins_discretizer(data,columns):
    dis=KBinsDiscretizer(n_bins=3,encode='ordinal',strategy='uniform')
    for col in columns:
        data[col]=dis.fit_transform(data[[col]])
    return data

#مقیاس بندی (نرمال سازی)
from sklearn.preprocessing import MinMaxScaler 
def min_max_scaler(data,columns):
    scaler=MinMaxScaler()
    data=scaler.fit_transform(data)
    data=pd.DataFrame(data)
    data.columns=columns
    return data