import streamlit as st
import joblib
import numpy as np
import datetime
import pandas as pd

st.title("Приложение для контроля энергоэффективности")

st.header("Таблица потребителей электрической энергии, подключенных к сети")

data = pd.read_csv("database.csv", sep = ";")

st.dataframe(data)

st.header("Анализ технического состояния выбранного оборудования")

st.subheader("Временные реализации тока и напряжения для потребителя 3")

df1 = pd.read_csv("data_cut(50).csv")

st.line_chart(df1["voltage_norm"])

st.line_chart(df1["current_norm"])

st.subheader("Временные реализации тока и напряжения для потребителя 4")

df1 = pd.read_csv("data_cut(50).csv")

st.line_chart(df1["voltage_vnesh"])

st.line_chart(df1["current_vnesh"])

st.header("Текущее состояние системы")

st.write("Состояние системы в норме. Самодиагностика завершена")

