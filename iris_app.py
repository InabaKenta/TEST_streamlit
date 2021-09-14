from pandas.io.formats.format import Datetime64TZFormatter
import streamlit as st
from PIL import Image
import pickle
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

# モデルの読み込み
with open("model.pickle", mode="rb")as f:
    model = pickle.load(f)

pred_text=""

# ボタンが押された時の処理
def predict(num1,num2,num3,num4):
    if num1== 0 or num2== 0 or num3== 0 or num4== 0:
        pred_text="まだ未定"
        return pred_text
        pass

    X_test = np.array([[num1,num2,num3,num4]])
    X_test = X_test.astype(float)
    X_test = X_test*0.1
    pred = model.predict(X_test)
    if pred == 0:
        pred_text = "setosa"
    elif pred ==1 :
        pred_text = "virgicolor"
    elif pred == 2:
        pred_text = "virginica"
    else:
        pred_text = "もう一度入力し直してください！"
    return pred_text



st.title("アヤメの種別予測 超入門")

num1 = st.slider("ガクの長さ(mm)", 0, 70, 0, 1)
num2 = st.slider("ガクの幅(mm)", 0, 70, 0, 1)
num3 = st.slider("花びらの長さ(mm)", 0, 70, 0, 1)
num4 = st.slider("花びらの幅(mm)", 0, 70, 0, 1)


pred_text = predict(num1,num2,num3,num4)


"予測結果は", pred_text , "です！"

if pred_text=="setosa":
    
    image = Image.open(f'image/iris_{pred_text}.jpg')
    st.image(image, caption=f'{pred_text}の写真',use_column_width=True)

elif pred_text=="virgicolor":
    image = Image.open(f'image/iris_{pred_text}.jpg')
    st.image(image, caption=f'{pred_text}の写真',use_column_width=True)

elif pred_text=="virginica":
    image = Image.open(f'image/iris_{pred_text}.jpg')
    st.image(image, caption=f'{pred_text}の写真',use_column_width=True)

else:
    pass



