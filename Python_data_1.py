#Data import 예제

import pandas as pd
import openpyxl

a=pd.read_csv("excel_data.csv")

b=openpyxl.load_workbook("excel_data.xlsx")

c=pd.read_excel("excel_data.xlsx")
print(c)
print("xlsx 읽어온 파일: \n", c)
print('-----------------------------------------------------------------------------------------------')
print("c에서 column A 값은: \n", c['A'])
print('-----------------------------------------------------------------------------------------------')
print("c에서 column A를 column B로 나눈 값은: \n", c['A']/c['B'])
