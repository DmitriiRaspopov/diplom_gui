import streamlit as st
import joblib
import numpy as np
import datetime
import pandas as pd
from numpy.fft import rfft, rfftfreq
from scipy.signal import find_peaks
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM, Activation, Dropout
from keras.models import load_model
from keras.optimizers import adam_v2


from sklearn.metrics import median_absolute_error, r2_score




st.title("Контроль, мониторинг и диагностика электрооборудования")

st.sidebar.title('Потребители электрической энергии')

vars = ["Исторические данные из базы данных", "Сбор данных"]

p1 = st.sidebar.checkbox('Потребитель 1')
p2 = st.sidebar.checkbox('Потребитель 2')
p3 = st.sidebar.checkbox('Потребитель 3')
p4 = st.sidebar.checkbox('Потребитель 4')
p5 = st.sidebar.checkbox('Потребитель 5')
p6 = st.sidebar.checkbox('Потребитель 6')
p7 = st.sidebar.checkbox('Потребитель 7')
p8 = st.sidebar.checkbox('Потребитель 8')
p9 = st.sidebar.checkbox('Потребитель 9')
p10 = st.sidebar.checkbox('Потребитель 10')
p11 = st.sidebar.checkbox('Потребитель 11')
p12 = st.sidebar.checkbox('Потребитель 12')




opt = st.sidebar.selectbox("Выберите период данные", ("Исторические данные из базы данных", "Сбор новых данных"))

#with st.expander("Таблица потребителей электрической энергии, подключенных к электрической сети"):
#    data = pd.read_csv("database.csv", sep = ";")
#    st.dataframe(data)

with st.expander("Показатели качества электрической энергии"):
    if p1:
        st.write("Показатели качества для потребителя 1")
        df = pd.read_excel("data/quality_data2.xlsx")
        st.dataframe(df)
    if p2:
        st.write("Показатели качества для потребителя 2")
        df = pd.read_excel("data/quality_data3.xlsx")
        st.dataframe(df)
    if p3:
        st.write("Показатели качества для потребителя 3")
        df = pd.read_excel("data/quality_data4.xlsx")
        st.dataframe(df)
    if p4:
        st.write("Показатели качества для потребителя 4")
        df = pd.read_excel("data/quality_data111.xlsx")
        st.dataframe(df)
    if p5:
        st.write("Показатели качества для потребителя 5")
        df = pd.read_excel("data/quality_data111.xlsx")
        st.dataframe(df)
    if p6:
        st.write("Показатели качества для потребителя 6")
        df = pd.read_excel("data/quality_data111.xlsx")
        st.dataframe(df)



    
def make_spectrum(data, N):
    rfft_abs=pd.DataFrame()
    for col in data.columns[:-1]:
        spectrum = rfft(data[col])
        rfft_abs[col]=np.abs(spectrum)/N
    return rfft_abs


def spectral_analysis():
    vol1 = pd.read_csv("data/vol_dest_all1.csv", skiprows=5, header = None)
    cur1 = pd.read_csv("data/cur_dest_all1.csv", skiprows=5, header = None)
    df1 = pd.concat([cur1[0], vol1[0]], axis = 1)
    df1.columns = ['ток_01', 'напряжение_01']
    
    
    # формирование списка со значениями границ точек для каждой секунды

    lis=[]
    for i in range(0,len(df1), 5000):
        if i <= len(df1):
            lis.append(i)
    
    lis.append(len(df1))
    
    # список датафреймов со значениями тока и напряжения для каждой секунды

    l = []
    for i in range(len(lis)):
        if i < 60:
            df2 = df1[lis[i]:lis[i+1]]
            l.append(df2)
            
    d = {}
    for i in range(len(l)):
        col = f'1_{i}_sec'
        val = l[i]["ток_01"].reset_index(drop = True)
        d[col] = val
    cur_df = pd.DataFrame(d)

    d = {}
    for i in range(len(l)):
        col = f'1_{i}_sec'
        val = l[i]["напряжение_01"].reset_index(drop = True)
        d[col] = val
    vol_df = pd.DataFrame(d)
    
    FD = 5000;

    N = len(cur_df.iloc[:,1])
    Freq_rfft_cur=rfftfreq(N, 1./FD) #возвращает частоты для выходных массивов функций rfft*.

    
    rfft_abs_cur=make_spectrum(cur_df, N=N)
    rfft_abs_vol=make_spectrum(vol_df, N=N)

    st.write("Спектр сигнала тока")
    st.line_chart(np.minimum(rfft_abs_cur.iloc[:2000,1],1))
    st.write("Спектр сигнала напряжения")
    st.line_chart(np.minimum(rfft_abs_vol.iloc[:2000,1],1))


def model_tech_state(cur1, vol1, n_potr):

    df1 = pd.concat([cur1[0], vol1[0]], axis = 1)
    df1.columns = ['ток_01', 'напряжение_01']
    l=[]
    for i in range(0,len(df1['напряжение_01']),5000):
        l.append(np.dot(np.max(abs(df1['напряжение_01'][i:i+5000])), np.max(abs(df1['ток_01'][i:i+5000]))))

    df_p1=pd.DataFrame({'power1':l})

    df_r = df_p1["power1"].rolling(window=40, min_periods = 1).median()

    # create a differenced series
    def difference(dataset, interval=1):
        diff = list()
        for i in range(interval, len(dataset)):
            value = dataset[i] - dataset[i - interval]
            diff.append(value)
        return pd.Series(diff)

    # convert time series into supervised learning problem
    def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
        n_vars = 1 if type(data) is list else data.shape[1]
        df = pd.DataFrame(data)
        cols, names = list(), list()
        # input sequence (t-n, ... t-1)
        for i in range(n_in, 0, -1):
            cols.append(df.shift(i))
            names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
        # forecast sequence (t, t+1, ... t+n)
        for i in range(0, n_out):
            cols.append(df.shift(-i))
            if i == 0:
                names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
            else:
                names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
        # put it all together
        agg = pd.concat(cols, axis=1)
        agg.columns = names
        # drop rows with NaN values
        if dropnan:
            agg.dropna(inplace=True)
        return agg

    # transform series into train and test sets for supervised learning
    def prepare_data(series, n_test, n_lag, n_seq):
        # extract raw values
        raw_values = series.values
        # transform data to be stationary
        diff_series = difference(raw_values, 1)
        diff_values = diff_series.values
        diff_values = diff_values.reshape(len(diff_values), 1)
        # rescale values to -1, 1
        scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_values = scaler.fit_transform(diff_values)
        scaled_values = scaled_values.reshape(len(scaled_values), 1)
        # transform into supervised learning problem X, y
        supervised = series_to_supervised(scaled_values, n_lag, n_seq)
        supervised_values = supervised.values
        # split into train and test sets
        train, test = supervised_values[0:-n_test], supervised_values[-n_test:]
        return scaler, train, test

    # configure
    n_lag = 60
    n_seq = 3
    n_test = 300
    # prepare data
    scaler, train, test = prepare_data(df_r, n_test, n_lag, n_seq)

    adam = tf.keras.optimizers.Adam(
        learning_rate=0.0033)


    # fit an LSTM network to training data
    def fit_lstm(train, n_lag, n_seq, n_batch):
        # reshape training into [samples, timesteps, features]
        X, y = train[:, 0:n_lag], train[:, n_lag:]
        X = X.reshape(X.shape[0], 1, X.shape[1])
        # design network
        model = Sequential()
        model.add(LSTM(100, batch_input_shape=(1, X.shape[1], X.shape[2]), return_sequences=True))
        model.add(LSTM(50, batch_input_shape=(1, X.shape[1], X.shape[2])))
        model.add(Dense(y.shape[1]))
        model.compile(loss='mae', optimizer=adam)
        model.fit(X, y, epochs=5, batch_size=n_batch, verbose=1, shuffle=False)
        return model

    # fit model
    model = fit_lstm(train, n_lag, n_seq, 1) # batch size = 1, т.к. берем целую секунду в наблюдение


    # make one forecast with an LSTM,
    def forecast_lstm(model, X, n_batch):
        # reshape input pattern to [samples, timesteps, features]
        X = X.reshape(1, 1, len(X))
        # make forecast
        forecast = model.predict(X, batch_size=n_batch)
        # convert to array
        return [x for x in forecast[0, :]]

    # evaluate the persistence model
    def make_forecasts(model, n_batch, train, test, n_lag, n_seq):
        forecasts = list()
        for i in range(len(test)):
            X, y = test[i, 0:n_lag], test[i, n_lag:]
            # make forecast
            forecast = forecast_lstm(model, X, n_batch)
            # store the forecast
            forecasts.append(forecast)
        return forecasts


    # make forecasts
    forecasts = make_forecasts(model, 1, train, test, n_lag, n_seq)

    # invert differenced forecast
    def inverse_difference(last_ob, forecast):
        # invert first forecast
        inverted = list()
        inverted.append(forecast[0] + last_ob)
        # propagate difference forecast using inverted first value
        for i in range(1, len(forecast)):
            inverted.append(forecast[i] + inverted[i-1])
        return inverted

    # inverse data transform on forecasts
    def inverse_transform(series, forecasts, scaler, n_test):
        inverted = list()
        for i in range(len(forecasts)):
            # create array from forecast
            forecast = np.array(forecasts[i])
            forecast = forecast.reshape(1, len(forecast))
            # invert scaling
            inv_scale = scaler.inverse_transform(forecast)
            inv_scale = inv_scale[0, :]
            # invert differencing
            index = len(series) - n_test + i - 1
            last_ob = series.values[index]
            inv_diff = inverse_difference(last_ob, inv_scale)
            # store
            inverted.append(inv_diff)
        return inverted




    # evaluate the RMSE for each forecast time step
    def evaluate_forecasts(test, forecasts, n_lag, n_seq):
        for i in range(n_seq):
            actual = [row[i] for row in test]
            predicted = [forecast[i] for forecast in forecasts]
            rmse = np.sqrt(mean_squared_error(actual, predicted))
            print('t+%d RMSE: %f' % ((i+1), rmse))




    # plot the forecasts in the context of the original dataset
    def plot_forecasts(series, forecasts, n_test):
        # plot the entire dataset in blue
        fig = plt.figure(figsize = (10,5))
        plt.plot(series.values, label = "Истинное значение")
        plt.xlabel("Время, с")
        plt.ylabel("Мощность, Вт")
        plt.title(f"Анализ и прогнозирование технического состояния и энергопотребления \n потребителя {n_potr}")

        #plt.vlines(2540, ymin = 0.010, ymax = 0.035, color = "black")
        #plt.vlines(2735, ymin = 0.010, ymax = 0.035,  color = "black")
        #
        #plt.vlines(3010, ymin = 0.010, ymax = 0.035,  color = "black")
        #plt.vlines(3210, ymin = 0.010, ymax = 0.035,  color = "black")
        #
        #plt.vlines(3500,ymin = 0.010, ymax = 0.035, color = "black")
        #plt.vlines(3780,ymin = 0.010, ymax = 0.035, color = "black")
        #
        #plt.hlines(0.017, xmin = 0, xmax = 4300, color = "yellow", label = "Контрольный порог")
        #plt.hlines(0.02, xmin = 0, xmax = 4300, color = "red", label = "Аварийный порог")

        # plot the forecasts in red
        for i in range(len(forecasts)):
            if i == (len(forecasts)-1):
                off_s = len(series) - n_test + i - 1
                off_e = off_s + len(forecasts[i]) + 1
                xaxis = [x for x in range(off_s, off_e)]
                yaxis = [series.values[off_s]] + forecasts[i]

                plt.plot(xaxis, yaxis, color='red', label = "Прогноз модели")
            else:    
                off_s = len(series) - n_test + i - 1
                off_e = off_s + len(forecasts[i]) + 1
                xaxis = [x for x in range(off_s, off_e)]
                yaxis = [series.values[off_s]] + forecasts[i]
                plt.plot(xaxis, yaxis, color='red')

        plt.grid()
        plt.legend(loc='upper left')
        return fig

    # inverse transform forecasts and test
    forecasts = inverse_transform(df_r, forecasts, scaler, n_test+2)
    actual = [row[n_lag:] for row in test]
    actual = inverse_transform(df_r, actual, scaler, n_test+2)
    # evaluate forecasts
    evaluate_forecasts(actual, forecasts, n_lag, n_seq)
    # plot forecasts
    fig = plot_forecasts(df_r, forecasts, n_test+2)
    st.pyplot(fig)



with st.expander("Анализ и прогноз технического состояния, энергопотребления и самодиагностики электрооборудования"):
    if p1:
        st.subheader("Временные реализации тока и напряжения для потребителя 1")
        df1 = pd.read_csv("data/data_cut(50).csv")
        st.line_chart(df1["voltage_norm"])
        st.line_chart(df1["current_norm"])
        st.subheader("Спектральный анализ сигналов")
        spectral_analysis()
    if p2:
        st.subheader("Временные реализации тока и напряжения для потребителя 2")
        df1 = pd.read_csv("data/data_cut(50).csv")
        st.line_chart(df1["voltage_norm"])
        st.line_chart(df1["current_norm"])
        st.subheader("Спектральный анализ сигналов")
        spectral_analysis()
    if p3:
        st.subheader("Временные реализации тока и напряжения для потребителя 3")
        df1 = pd.read_csv("data/data_cut(50).csv")
        st.line_chart(df1["voltage_norm"])
        st.line_chart(df1["current_norm"])
        st.subheader("Спектральный анализ сигналов")
        spectral_analysis()
    if p4:
        st.subheader("Временные реализации тока и напряжения для потребителя 4")
        df1 = pd.read_csv("data/data_cut(50).csv")
        st.line_chart(df1["voltage_norm"])
        st.line_chart(df1["current_norm"])
        st.subheader("Спектральный анализ сигналов")
        spectral_analysis()
    if p5:
        st.subheader("Временные реализации тока и напряжения для потребителя 5")
        df1 = pd.read_csv("data/data_cut(50).csv")
        st.line_chart(df1["voltage_norm"])
        st.line_chart(df1["current_norm"])
        st.subheader("Спектральный анализ сигналов")
        spectral_analysis()
    if p6:
        st.subheader("Временные реализации тока и напряжения для потребителя 6")
        df1 = pd.read_csv("data/data_cut(50).csv")
        st.line_chart(df1["voltage_norm"])
        st.line_chart(df1["current_norm"])
        st.subheader("Спектральный анализ сигналов")
        spectral_analysis()

    st.subheader("Прогноз технического состояния электрооборудования")
    if p1:
        vol1 = pd.read_csv("data/vol_dest_all1.csv", skiprows=5, header = None)
        cur1 = pd.read_csv("data/cur_dest_all1.csv", skiprows=5, header = None)
        model_tech_state(vol1 = vol1, cur1 = cur1, n_potr = 1)

    if p2:
        vol1 = pd.read_csv("data/vol_dest_all2.csv", skiprows=5, header = None)
        cur1 = pd.read_csv("data/cur_dest_all2.csv", skiprows=5, header = None)
        model_tech_state(vol1 = vol1, cur1 = cur1, n_potr = 2)
    if p3:
        vol1 = pd.read_csv("data/vol_dest_all3.csv", skiprows=5, header = None)
        cur1 = pd.read_csv("data/cur_dest_all3.csv", skiprows=5, header = None)
        model_tech_state(vol1 = vol1, cur1 = cur1, n_potr = 3)
    if p4:
        vol1 = pd.read_csv("data/vol_dest_all4.csv", skiprows=5, header = None)
        cur1 = pd.read_csv("data/cur_dest_all4.csv", skiprows=5, header = None)
        model_tech_state(vol1 = vol1, cur1 = cur1, n_potr = 4)
    
    





#with st.expander("Прогнозирование энергопотребления"):    