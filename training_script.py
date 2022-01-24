import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd

import pickle
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor


def import_data(MY_PATH, MY_CSV_DATA):
    """
    Import data to dataframes.
    :return: a dataframe containing dataset read from a csv file
    """

    # load data

    df = pd.read_csv(MY_PATH + MY_CSV_DATA, skiprows=1,
                     encoding="ISO-8859-1",
                     sep=',',
                     names=('id', 'vin', 'price', 'miles', 'stock_no', 'year', 'make', 'model', 'trim',
                            'body_type', 'vehicle_type', 'transmission', 'fuel_type',
                            'engine_size', 'seller_name', 'street', 'city', 'state', 'zip'))

    # Show all columns on one line.
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)

    return df


def acquire_data_description(dataset):
    """
    Display the general statistic information of the dataset.
    Including:
        1. data types
        2. Stat summaries for numeric columns
        3. Stat summaries for non-numeric columns
    :param dataset: a dataset
    """
    print(dataset.head())

    # check data length
    print("\n===DATA Length===")
    print(len(dataset))

    # print title with space before.
    print("\n===DATA TYPES===")
    print(dataset.dtypes)

    # Show statistical summaries for numeric columns.
    print("\n===STATISTIC SUMMARIES for NUMERIC COLUMNS===")
    print(dataset.describe().transpose())

    # Show summaries for objects like dates and strings.
    print("\n===STATISTIC SUMMARIES for NON NUMERIC COLUMNS===")
    print(dataset.describe(include=['object']).transpose())

    print("\n===CHECK IF CONTAINS NULL VALUES===")
    print(dataset.isnull().sum())

    print("\n===CHECK IF CONTAINS DUPLICATE VALUES===")
    data_len = len(dataset)
    dup_Data = dataset.copy().drop_duplicates()
    dup_len = len(dup_Data)
    if data_len == dup_len:
        print('No duplicated Value')
    else:
        print(f'There\'s {data_len - dup_len} duplicated value')


def display_missing_value_summary(data):
    print('\n\n=== Percentage of missing values ===\n')
    print(data.isnull().sum().sort_values(ascending=False) / len(data) * 100)


def check_unique_value(data):
    for col in data:
        print('\n==' + str(col))
        print(data[col].unique())
        print('Num of unique values:' + str(len(data[col].unique())))
        print('\n')


def plt_influence_on_price(df, col_name):
    sns.catplot(x=col_name, y="price", kind="boxen", data=df, height=5, aspect=3)
    plt.xticks(rotation=90)
    plt.title(f'Influence of manufactured {col_name} on price')
    plt.tight_layout()
    plt.show()


def plt_distribution(data, col_name):
    # plotting histogram
    plt.hist(data[col_name])
    plt.title(f'Distribution of {col_name}')
    plt.tight_layout()
    plt.show()


def eda_price_plot(df):
    sns.distplot(df['price'])
    plt.tight_layout()
    plt.show()


def eda_body_type_distribution_plot(df):
    plt.figure(figsize=(9, 5))
    chart = sns.countplot(x=df.body_type)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    plt.title('Distribution of Body_type')
    plt.tight_layout()
    plt.show()


def eda_car_make_summary_plot(df):
    print(round(df.make.value_counts() / sum(df.make.value_counts()) * 100, 2))
    chart = sns.countplot(x=df.make)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    plt.title('Car Make Summary')
    plt.tight_layout()
    plt.show()


def EDA(df):
    print('\n=== EDA Section ===\n')
    # print(df.head(10))
    # check_unique_value(df)
    #
    # eda_price_plot(df)
    #
    # print(round(df.transmission.value_counts() / sum(df.transmission.value_counts()) * 100, 2))
    #
    # print(round(df.body_type.value_counts() / sum(df.body_type.value_counts()) * 100, 2))

    # eda_body_type_distribution_plot(df)
    #
    # print(df.model.unique())
    # print(df.model.value_counts().head(10))
    # print(check_unique_value(df))
    #
    # plt_distribution(df, 'year')
    #
    # plt_distribution(df, 'miles')
    # plt_influence_on_price(df, 'miles')

    # print(round(df.fuel_type.value_counts() / sum(df.fuel_type.value_counts()) * 100, 2))
    #
    # print(round(df.trim.value_counts() / sum(df.trim.value_counts()) * 100, 2))

    eda_car_make_summary_plot(df)

    # plt_influence_on_price(df, 'make')
    #
    # plt_distribution(df, 'fuel_type')


def standardise_body_types(data):
    """standardises models types into specific body types"""

    # covert all uppercase to lowercase
    data['body_type'] = data['body_type'].str.lower()
    # print(data.body_type.unique())

    # create temp df for replacement
    df_temp = pd.DataFrame()
    df_temp['body_type'] = data['body_type'].copy()

    df_temp['body_type'] = df_temp['body_type'].replace('koup', 'coupe')
    df_temp['body_type'] = df_temp['body_type'].replace('crossover', 'suv')

    # print(df_temp['body_type'].unique())

    data = data[['price', 'miles', 'year', 'make', 'trim', 'transmission', 'fuel_type']]
    data = pd.concat([data, df_temp['body_type']], axis=1)

    # print(data['body_type'].unique())

    return data


def get_dummy_variable(myDF, col):
    tempDf = myDF
    df_dummy = pd.get_dummies(tempDf, columns=col)

    finalDF = pd.concat(([myDF, df_dummy]), axis=1)
    return finalDF


def convertNAcellsToNonNumericValue(colName, df):
    # Create two new column names based on original column name.
    indicatorColName = 'm_' + colName  # Tracks whether imputed.
    imputedColName = 'imp_' + colName  # Stores original & imputed data.

    # get the most frequent value from the model
    imputedValue = df[colName].mode()[0]
    # print(imputedValue)

    # Populate new columns with data.
    imputedColumn = []
    indictorColumn = []
    for i in range(len(df)):
        isImputed = False

        # mi_OriginalName column stores imputed & original data.
        if pd.isnull(df.iloc[i][colName]):
            isImputed = True
            imputedColumn.append(imputedValue)
        else:
            imputedColumn.append(df.iloc[i][colName])

        # mi_OriginalName column tracks if is imputed (1) or not (0).
        if (isImputed):
            indictorColumn.append(1)
        else:
            indictorColumn.append(0)

    # Append new columns to dataframe but always keep original column.
    df[indicatorColName] = indictorColumn
    df[imputedColName] = imputedColumn
    return df


def convertNAcellsToNum(colName, df, measureType):
    # Create two new column names based on original column name.
    indicatorColName = 'm_' + colName  # Tracks whether imputed.
    imputedColName = 'imp_' + colName  # Stores original & imputed data.

    # Get mean or median depending on preference.
    imputedValue = 0
    if (measureType == "median"):
        imputedValue = df[colName].median()
    elif (measureType == "mode"):
        imputedValue = float(df[colName].mode())
    else:
        imputedValue = df[colName].mean()

    # Populate new columns with data.
    imputedColumn = []
    indictorColumn = []
    for i in range(len(df)):
        isImputed = False
        # print(df.iloc[i])

        # mi_OriginalName column stores imputed & original data.
        if np.isnan(df.iloc[i][colName]):
            isImputed = True
            imputedColumn.append(imputedValue)
        else:
            imputedColumn.append(df.iloc[i][colName])

        # mi_OriginalName column tracks if is imputed (1) or not (0).
        if (isImputed):
            indictorColumn.append(1)
        else:
            indictorColumn.append(0)

    # Append new columns to dataframe but always keep original column.
    df[indicatorColName] = indictorColumn
    df[imputedColName] = imputedColumn

    return df


def prepare_imputed_dataset(dataset):
    # dataset = convertNAcellsToNum('miles', dataset, "median")

    from sklearn.impute import KNNImputer

    knn_imp = KNNImputer(n_neighbors=5, add_indicator=True)
    knn_result = knn_imp.fit_transform(dataset[['price', 'miles', 'year']])
    knn_df = pd.DataFrame(knn_result, columns=['price', 'm_miles', 'year', 'imp_miles'])
    # print(knn_df.head(5))
    # print(len(knn_df))
    #
    # print(len(dataset))
    # print(dataset.head(5))
    knn_df.reset_index(drop=True, inplace=True)
    dataset.reset_index(drop=True, inplace=True)

    result_df = pd.concat(([dataset, knn_df[['m_miles', 'imp_miles']]]), axis=1)
    # print(len(result_df))
    # print(result_df.head(5))

    result_df = convertNAcellsToNonNumericValue('body_type', result_df)
    result_df = convertNAcellsToNonNumericValue('transmission', result_df)
    col_names = ['body_type', 'transmission', 'make', 'trim', 'fuel_type']
    finalDF = get_dummy_variable(result_df, col_names)

    # remove duplicate columns
    finalDF = finalDF.loc[:, ~finalDF.columns.duplicated()]
    print(finalDF.head(10))

    return finalDF


def filter_out_imputed_cols(df, names):
    df_final = df.copy()

    for name in names:
        df_final = df_final.drop(name, axis=1)

    return df_final


def raw_model(dataframe):
    X = dataframe[['year', 'm_miles', 'imp_miles', 'm_body_type', 'm_transmission',
                   'body_type_convertible', 'body_type_coupe', 'body_type_hatchback',
                   'body_type_minivan', 'body_type_pickup', 'body_type_sedan', 'body_type_suv', 'body_type_wagon',
                   'transmission_Automatic', 'transmission_Manual', 'make_Audi', 'make_BMW', 'make_Bentley',
                   'make_Cadillac', 'make_Chevrolet', 'make_Dodge', 'make_Ford', 'make_GMC', 'make_Honda',
                   'make_Hummer', 'make_INFINITI', 'make_Jeep', 'make_Land Rover', 'make_Lexus',
                   'make_Lincoln', 'make_MINI', 'make_Maserati', 'make_Mercedes-Benz', 'make_Nissan',
                   'make_Porsche', 'make_RAM', 'make_Toyota', 'make_Volkswagen', 'make_Volvo', 'trim_Base',
                   'trim_Custom', 'trim_Sport', 'trim_Winter', 'trim_untrimmed', 'fuel_type_Diesel',
                   'fuel_type_Electric', 'fuel_type_Gas', 'fuel_type_Hybrid'
                   ]]

    X = sm.add_constant(X)

    y = dataframe['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)  # make the predictions by the model
    print(model.summary())
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))


def linear_model(dataframe):
    X = dataframe[['year', 'm_body_type', 'm_transmission',
                   'body_type_convertible', 'body_type_coupe', 'body_type_hatchback',
                   'body_type_minivan', 'body_type_pickup', 'body_type_sedan', 'body_type_suv', 'body_type_wagon',
                   'transmission_Automatic', 'transmission_Manual', 'make_Audi', 'make_BMW', 'make_Bentley',
                   'make_Cadillac', 'make_Chevrolet', 'make_Dodge', 'make_Ford', 'make_GMC', 'make_Honda',
                   'make_Hummer', 'make_INFINITI', 'make_Jeep', 'make_Land Rover', 'make_Lexus',
                   'make_Lincoln', 'make_MINI', 'make_Maserati', 'make_Mercedes-Benz', 'make_Nissan',
                   'make_Porsche', 'make_RAM', 'make_Toyota', 'make_Volkswagen', 'make_Volvo', 'trim_Base',
                   'trim_Custom', 'trim_Sport', 'trim_Winter', 'trim_untrimmed', 'fuel_type_Diesel',
                   'fuel_type_Electric', 'fuel_type_Gas', 'fuel_type_Hybrid'
                   ]]

    X = sm.add_constant(X)

    y = dataframe['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = sm.OLS(y_train, X_train).fit()
    predictions = model.predict(X_test)  # make the predictions by the model
    print(model.summary())
    print('Root Mean Squared Error:', np.sqrt(metrics.mean_squared_error(y_test, predictions)))

    Pkl_Filename = "linear_model.pkl"

    # dump model
    with open(Pkl_Filename, 'wb') as file:
        pickle.dump(model, file)

    return model


def neural_model_num(dataframe):
    X = dataframe[['year', 'imp_miles', 'm_body_type', 'm_transmission',
                   'body_type_convertible', 'body_type_coupe', 'body_type_hatchback',
                   'body_type_minivan', 'body_type_pickup', 'body_type_sedan', 'body_type_suv', 'body_type_wagon',
                   'transmission_Automatic', 'transmission_Manual', 'make_Audi', 'make_BMW', 'make_Bentley',
                   'make_Cadillac', 'make_Chevrolet', 'make_Dodge', 'make_Ford', 'make_GMC', 'make_Honda',
                   'make_Hummer', 'make_INFINITI', 'make_Jeep', 'make_Land Rover', 'make_Lexus',
                   'make_Lincoln', 'make_MINI', 'make_Maserati', 'make_Mercedes-Benz', 'make_Nissan',
                   'make_Porsche', 'make_RAM', 'make_Toyota', 'make_Volkswagen', 'make_Volvo', 'trim_Base',
                   'trim_Custom', 'trim_Sport', 'trim_Winter', 'trim_untrimmed', 'fuel_type_Diesel',
                   'fuel_type_Electric', 'fuel_type_Gas', 'fuel_type_Hybrid'
                   ]].values

    y = dataframe['price'].values

    ROW_DIM = 0
    COL_DIM = 1

    x_arrayReshaped = X.reshape(X.shape[ROW_DIM], X.shape[COL_DIM])
    y_arrayReshaped = y.reshape(y.shape[ROW_DIM], 1)

    # Split the data.
    X_train, X_test, y_train, y_test = train_test_split(x_arrayReshaped,
                                                        y_arrayReshaped, test_size=0.2)

    # Define the model.
    def create_model():
        model = Sequential()
        model.add(Dense(47, input_dim=47, kernel_initializer='normal',
                        activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer='adam')
        return model

    # ### Grid Building Section #######################
    # model = KerasRegressor(build_fn=create_model)
    #
    # # define the grid search parameters
    # batch_size = [10, 50, 100, 200]
    # epochs = [100, 200, 300, 400, 500]
    # param_grid = dict(batch_size=batch_size, epochs=epochs)
    # grid = GridSearchCV(estimator=model, param_grid=param_grid,
    #                     n_jobs=-1, cv=3, verbose=1)
    # #################################################
    #
    # grid_result = grid.fit(X_train, y_train)
    #
    # # summarize results
    # print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    # means = grid_result.cv_results_['mean_test_score']
    # stds = grid_result.cv_results_['std_test_score']
    # params = grid_result.cv_results_['params']
    #
    # for mean, stdev, param in zip(means, stds, params):
    #     print("%f (%f) with: %r" % (mean, stdev, param))

    # Since this is a linear regression use KerasRegressor.
    estimator = KerasRegressor(build_fn=create_model, epochs=100,
                               batch_size=5, verbose=1)

    # Use kfold analysis for a more reliable estimate.
    kfold = KFold(n_splits=10)
    results = cross_val_score(estimator, X_train, y_train, cv=kfold)
    print("Baseline Mean (%.2f) MSE (%.2f) " % (results.mean(), results.std()))
    print("Baseline RMSE: " + str(np.sqrt(results.std())))

    # Build the model.
    model = create_model()
    history = model.fit(X_train, y_train, epochs=500,
                        batch_size=10, verbose=1,
                        validation_data=(X_test, y_test))

    # Evaluate the model.
    predictions = model.predict(X_test)
    mse = metrics.mean_squared_error(y_test, predictions)
    print("Neural network MSE: " + str(mse))
    print("Neural network RMSE: " + str(np.sqrt(mse)))


def neural_model_optimizer(dataframe):
    X = dataframe[['year', 'imp_miles', 'm_body_type', 'm_transmission',
                   'body_type_convertible', 'body_type_coupe', 'body_type_hatchback',
                   'body_type_minivan', 'body_type_pickup', 'body_type_sedan', 'body_type_suv', 'body_type_wagon',
                   'transmission_Automatic', 'transmission_Manual', 'make_Audi', 'make_BMW', 'make_Bentley',
                   'make_Cadillac', 'make_Chevrolet', 'make_Dodge', 'make_Ford', 'make_GMC', 'make_Honda',
                   'make_Hummer', 'make_INFINITI', 'make_Jeep', 'make_Land Rover', 'make_Lexus',
                   'make_Lincoln', 'make_MINI', 'make_Maserati', 'make_Mercedes-Benz', 'make_Nissan',
                   'make_Porsche', 'make_RAM', 'make_Toyota', 'make_Volkswagen', 'make_Volvo', 'trim_Base',
                   'trim_Custom', 'trim_Sport', 'trim_Winter', 'trim_untrimmed', 'fuel_type_Diesel',
                   'fuel_type_Electric', 'fuel_type_Gas', 'fuel_type_Hybrid'
                   ]].values

    y = dataframe['price'].values

    ROW_DIM = 0
    COL_DIM = 1

    x_arrayReshaped = X.reshape(X.shape[ROW_DIM], X.shape[COL_DIM])
    y_arrayReshaped = y.reshape(y.shape[ROW_DIM], 1)

    # Split the data.
    X_train, X_test, y_train, y_test = train_test_split(x_arrayReshaped,
                                                        y_arrayReshaped,
                                                        test_size=0.2)

    # Define the model.
    def create_model(optimizer='Nadam'):
        model = Sequential()
        model.add(Dense(47, input_dim=47, kernel_initializer='normal',
                        activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        model.compile(loss='mean_squared_error', optimizer=optimizer)
        return model

    # ### Grid Building Section #######################
    model = KerasRegressor(build_fn=create_model, epochs=400, batch_size=100, verbose=1)

    # Define the grid search parameters.
    optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
    param_grid = dict(optimizer=optimizer)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    #################################################

    grid_result = grid.fit(X_train, y_train)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def neural_model_learning_rate(dataframe):
    X = dataframe[['year', 'imp_miles', 'm_body_type', 'm_transmission',
                   'body_type_convertible', 'body_type_coupe', 'body_type_hatchback',
                   'body_type_minivan', 'body_type_pickup', 'body_type_sedan', 'body_type_suv', 'body_type_wagon',
                   'transmission_Automatic', 'transmission_Manual', 'make_Audi', 'make_BMW', 'make_Bentley',
                   'make_Cadillac', 'make_Chevrolet', 'make_Dodge', 'make_Ford', 'make_GMC', 'make_Honda',
                   'make_Hummer', 'make_INFINITI', 'make_Jeep', 'make_Land Rover', 'make_Lexus',
                   'make_Lincoln', 'make_MINI', 'make_Maserati', 'make_Mercedes-Benz', 'make_Nissan',
                   'make_Porsche', 'make_RAM', 'make_Toyota', 'make_Volkswagen', 'make_Volvo', 'trim_Base',
                   'trim_Custom', 'trim_Sport', 'trim_Winter', 'trim_untrimmed', 'fuel_type_Diesel',
                   'fuel_type_Electric', 'fuel_type_Gas', 'fuel_type_Hybrid'
                   ]].values

    y = dataframe['price'].values

    ROW_DIM = 0
    COL_DIM = 1

    x_arrayReshaped = X.reshape(X.shape[ROW_DIM], X.shape[COL_DIM])
    y_arrayReshaped = y.reshape(y.shape[ROW_DIM], 1)

    # Split the data.
    X_train, X_test, y_train, y_test = train_test_split(x_arrayReshaped,
                                                        y_arrayReshaped,
                                                        test_size=0.2)

    from tensorflow.keras.optimizers import Nadam  # for adam optimizer

    def create_model(learningRate=0.0005):
        model = Sequential()
        model.add(Dense(47, input_dim=47, kernel_initializer='normal',
                        activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        opt = Nadam(lr=learningRate)
        model.compile(loss='mean_squared_error', optimizer=opt)
        return model

    ### Grid Building Section #######################
    model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=10, verbose=1)

    # Define the grid search parameters.
    learningRates = [0.00005, 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.015, 0.2]
    param_grid = dict(learningRate=learningRates)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    #################################################

    grid_result = grid.fit(X_train, y_train)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def neural_model_initializer_tuning(dataframe):
    X = dataframe[['year', 'imp_miles', 'm_body_type', 'm_transmission',
                   'body_type_convertible', 'body_type_coupe', 'body_type_hatchback',
                   'body_type_minivan', 'body_type_pickup', 'body_type_sedan', 'body_type_suv', 'body_type_wagon',
                   'transmission_Automatic', 'transmission_Manual', 'make_Audi', 'make_BMW', 'make_Bentley',
                   'make_Cadillac', 'make_Chevrolet', 'make_Dodge', 'make_Ford', 'make_GMC', 'make_Honda',
                   'make_Hummer', 'make_INFINITI', 'make_Jeep', 'make_Land Rover', 'make_Lexus',
                   'make_Lincoln', 'make_MINI', 'make_Maserati', 'make_Mercedes-Benz', 'make_Nissan',
                   'make_Porsche', 'make_RAM', 'make_Toyota', 'make_Volkswagen', 'make_Volvo', 'trim_Base',
                   'trim_Custom', 'trim_Sport', 'trim_Winter', 'trim_untrimmed', 'fuel_type_Diesel',
                   'fuel_type_Electric', 'fuel_type_Gas', 'fuel_type_Hybrid'
                   ]].values

    y = dataframe['price'].values

    ROW_DIM = 0
    COL_DIM = 1

    x_arrayReshaped = X.reshape(X.shape[ROW_DIM], X.shape[COL_DIM])
    y_arrayReshaped = y.reshape(y.shape[ROW_DIM], 1)

    # Split the data.
    X_train, X_test, y_train, y_test = train_test_split(x_arrayReshaped,
                                                        y_arrayReshaped,
                                                        test_size=0.2)

    # Define the model
    from tensorflow.keras.optimizers import Nadam  # for adam optimizer

    def create_model(init_mode='normal'):
        model = Sequential()
        model.add(Dense(47, input_dim=47, kernel_initializer=init_mode,
                        activation='relu'))
        model.add(Dense(1, kernel_initializer=init_mode))
        opt = Nadam(lr=0.0005)
        model.compile(loss='mean_squared_error', optimizer=opt)
        return model

    ############## Grid Building Section ############
    model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=10, verbose=1)

    # Define the grid search parameters.
    init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal',
                 'glorot_uniform', 'he_normal', 'he_uniform']
    param_grid = dict(init_mode=init_mode)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    #################################################

    grid_result = grid.fit(X_train, y_train)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))


def neural_model_activation_tuning(dataframe):
    X = dataframe[['year', 'imp_miles', 'm_body_type', 'm_transmission',
                   'body_type_convertible', 'body_type_coupe', 'body_type_hatchback',
                   'body_type_minivan', 'body_type_pickup', 'body_type_sedan', 'body_type_suv', 'body_type_wagon',
                   'transmission_Automatic', 'transmission_Manual', 'make_Audi', 'make_BMW', 'make_Bentley',
                   'make_Cadillac', 'make_Chevrolet', 'make_Dodge', 'make_Ford', 'make_GMC', 'make_Honda',
                   'make_Hummer', 'make_INFINITI', 'make_Jeep', 'make_Land Rover', 'make_Lexus',
                   'make_Lincoln', 'make_MINI', 'make_Maserati', 'make_Mercedes-Benz', 'make_Nissan',
                   'make_Porsche', 'make_RAM', 'make_Toyota', 'make_Volkswagen', 'make_Volvo', 'trim_Base',
                   'trim_Custom', 'trim_Sport', 'trim_Winter', 'trim_untrimmed', 'fuel_type_Diesel',
                   'fuel_type_Electric', 'fuel_type_Gas', 'fuel_type_Hybrid'
                   ]].values

    y = dataframe['price'].values
    ROW_DIM = 0
    COL_DIM = 1

    x_arrayReshaped = X.reshape(X.shape[ROW_DIM], X.shape[COL_DIM])
    y_arrayReshaped = y.reshape(y.shape[ROW_DIM], 1)

    # Split the data.
    X_train, X_test, y_train, y_test = train_test_split(x_arrayReshaped,
                                                        y_arrayReshaped, test_size=0.2)

    # Define the model
    from tensorflow.keras.optimizers import Nadam  # for adam optimizer

    def create_model(activation='softplus'):
        model = Sequential()
        model.add(Dense(47, input_dim=47, kernel_initializer='normal',
                        activation=activation))
        model.add(Dense(1, kernel_initializer='normal'))
        opt = Nadam(lr=0.0005)
        model.compile(loss='mean_squared_error', optimizer=opt)
        return model

    ### Grid Building Section #######################
    model = KerasRegressor(build_fn=create_model, epochs=300, batch_size=100, verbose=1)

    # Define the grid search parameters.
    activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh',
                  'sigmoid', 'hard_sigmoid', 'linear']
    param_grid = dict(activation=activation)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=3)
    #################################################
    grid_result = grid.fit(X_train, y_train)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    # # Since this is a linear regression use KerasRegressor.
    # estimator = KerasRegressor(build_fn=create_model, epochs=400,
    #                            batch_size=100, verbose=1)
    #
    # # Use kfold analysis for a more reliable estimate.
    # kfold = KFold(n_splits=10)
    # results = cross_val_score(estimator, X_train, y_train, cv=kfold)
    # print("Baseline Mean (%.2f) MSE (%.2f) " % (results.mean(), results.std()))
    # print("Baseline RMSE: " + str(np.sqrt(results.std())))
    #
    # # Build the model.
    # model = create_model()
    # history = model.fit(X_train, y_train, epochs=300,
    #                     batch_size=100, verbose=1,
    #                     validation_data=(X_test, y_test))
    #
    # # Evaluate the model.
    # predictions = model.predict(X_test)
    # mse = metrics.mean_squared_error(y_test, predictions)
    # print("Neural network MSE: " + str(mse))
    # print("Neural network RMSE: " + str(np.sqrt(mse)))


def neural_model_another_layer_tuning(dataframe):
    X = dataframe[['year', 'imp_miles', 'm_body_type', 'm_transmission',
                   'body_type_convertible', 'body_type_coupe', 'body_type_hatchback',
                   'body_type_minivan', 'body_type_pickup', 'body_type_sedan', 'body_type_suv', 'body_type_wagon',
                   'transmission_Automatic', 'transmission_Manual', 'make_Audi', 'make_BMW', 'make_Bentley',
                   'make_Cadillac', 'make_Chevrolet', 'make_Dodge', 'make_Ford', 'make_GMC', 'make_Honda',
                   'make_Hummer', 'make_INFINITI', 'make_Jeep', 'make_Land Rover', 'make_Lexus',
                   'make_Lincoln', 'make_MINI', 'make_Maserati', 'make_Mercedes-Benz', 'make_Nissan',
                   'make_Porsche', 'make_RAM', 'make_Toyota', 'make_Volkswagen', 'make_Volvo', 'trim_Base',
                   'trim_Custom', 'trim_Sport', 'trim_Winter', 'trim_untrimmed', 'fuel_type_Diesel',
                   'fuel_type_Electric', 'fuel_type_Gas', 'fuel_type_Hybrid'
                   ]].values

    y = dataframe['price'].values
    ROW_DIM = 0
    COL_DIM = 1

    x_arrayReshaped = X.reshape(X.shape[ROW_DIM], X.shape[COL_DIM])
    y_arrayReshaped = y.reshape(y.shape[ROW_DIM], 1)

    # Split the data.
    X_train, X_test, y_train, y_test = train_test_split(x_arrayReshaped,
                                                        y_arrayReshaped, test_size=0.2)

    # Define the model
    from tensorflow.keras.optimizers import RMSprop  # for adam optimizer

    def create_model(numNeurons=5, initializer='uniform', activation='softplus'):
        # create model
        model = Sequential()
        model.add(Dense(47, kernel_initializer='normal', input_dim=47, activation='relu'))

        model.add(Dense(numNeurons, kernel_initializer=initializer,
                        activation=activation))

        model.add(Dense(1, kernel_initializer='normal'))
        opt = RMSprop(lr=0.005)
        # Compile model
        model.compile(loss='mean_squared_error', optimizer=opt)
        return model

    ### Grid Building Section #######################
    # Define the parameters to try out
    params = {'activation': ['softmax', 'softplus', 'softsign', 'relu', 'tanh',
                             'sigmoid', 'hard_sigmoid', 'linear'],
              'numNeurons': [10, 15, 20, 25, 30, 35],
              'initializer': ['uniform', 'lecun_uniform', 'normal', 'zero',
                              'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
              }

    model = KerasRegressor(build_fn=create_model, epochs=100, batch_size=10, verbose=1)
    grid = GridSearchCV(estimator=model, param_grid=params, n_jobs=-1, cv=3)
    #################################################

    grid_result = grid.fit(X_train, y_train)

    # summarize results
    print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
    means = grid_result.cv_results_['mean_test_score']
    stds = grid_result.cv_results_['std_test_score']
    params = grid_result.cv_results_['params']

    for mean, stdev, param in zip(means, stds, params):
        print("%f (%f) with: %r" % (mean, stdev, param))

    # # Since this is a linear regression use KerasRegressor.
    # estimator = KerasRegressor(build_fn=create_model, epochs=400,
    #                            batch_size=100, verbose=1)
    #
    # # Use kfold analysis for a more reliable estimate.
    # kfold = KFold(n_splits=10)
    # results = cross_val_score(estimator, X_train, y_train, cv=kfold)
    # print("Baseline Mean (%.2f) MSE (%.2f) " % (results.mean(), results.std()))
    # print("Baseline RMSE: " + str(np.sqrt(results.std())))
    #
    # # Build the model.
    # model = create_model()
    # history = model.fit(X_train, y_train, epochs=300,
    #                     batch_size=100, verbose=1,
    #                     validation_data=(X_test, y_test))
    #
    # # Evaluate the model.
    # predictions = model.predict(X_test)
    # mse = metrics.mean_squared_error(y_test, predictions)
    # print("Neural network MSE: " + str(mse))
    # print("Neural network RMSE: " + str(np.sqrt(mse)))


def neural_model_tuned(dataframe):
    X = dataframe[['year', 'm_miles', 'imp_miles', 'm_body_type', 'm_transmission',
                   'body_type_convertible', 'body_type_coupe', 'body_type_hatchback',
                   'body_type_minivan', 'body_type_pickup', 'body_type_sedan', 'body_type_suv', 'body_type_wagon',
                   'transmission_Automatic', 'transmission_Manual', 'make_Audi', 'make_BMW', 'make_Bentley',
                   'make_Cadillac', 'make_Chevrolet', 'make_Dodge', 'make_Ford', 'make_GMC', 'make_Honda',
                   'make_Hummer', 'make_INFINITI', 'make_Jeep', 'make_Land Rover', 'make_Lexus',
                   'make_Lincoln', 'make_MINI', 'make_Maserati', 'make_Mercedes-Benz', 'make_Nissan',
                   'make_Porsche', 'make_RAM', 'make_Toyota', 'make_Volkswagen', 'make_Volvo', 'trim_Base',
                   'trim_Custom', 'trim_Sport', 'trim_Winter', 'trim_untrimmed', 'fuel_type_Diesel',
                   'fuel_type_Electric', 'fuel_type_Gas', 'fuel_type_Hybrid'
                   ]].values

    y = dataframe['price'].values
    ROW_DIM = 0
    COL_DIM = 1

    x_arrayReshaped = X.reshape(X.shape[ROW_DIM], X.shape[COL_DIM])
    y_arrayReshaped = y.reshape(y.shape[ROW_DIM], 1)

    # Split the data.
    X_train, X_test, y_train, y_test = train_test_split(x_arrayReshaped,
                                                        y_arrayReshaped, test_size=0.2)

    # Define the model
    from tensorflow.keras.optimizers import Nadam  # for adam optimizer

    def create_model():
        model = Sequential()
        model.add(Dense(48, input_dim=48, kernel_initializer='normal',
                        activation='relu'))
        model.add(Dense(1, kernel_initializer='normal'))
        opt = Nadam(lr=0.01)
        model.compile(loss='mean_squared_error', optimizer=opt)
        return model

    # # Since this is a linear regression use KerasRegressor.
    # estimator = KerasRegressor(build_fn=create_model, epochs=400,
    #                            batch_size=100, verbose=1)
    #
    # # Use kfold analysis for a more reliable estimate.
    # kfold = KFold(n_splits=10)
    # results = cross_val_score(estimator, X_train, y_train, cv=kfold)
    # print("Baseline Mean (%.2f) MSE (%.2f) " % (results.mean(), results.std()))
    # print("Baseline RMSE: " + str(np.sqrt(results.std())))

    # Build the model.
    model = create_model()
    # simple early stopping
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=200)
    mc = ModelCheckpoint("model_neural.h5",
                         monitor='val_loss', verbose=0, save_best_only=True,
                         save_weights_only=False,
                         mode='auto', save_freq='epoch')

    history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                        epochs=4000, batch_size=100, verbose=1, callbacks=[mc, es]
                        )

    # Evaluate the model.
    predictions = model.predict(X_test)
    mse = metrics.mean_squared_error(y_test, predictions)
    print("Neural network MSE: " + str(mse))
    print("Neural network RMSE: " + str(np.sqrt(mse)))

    # show loss
    showLoss(history)

    return model


def showLoss(history):
    # Get training and test loss histories
    training_loss = history.history['loss']
    validation_loss = history.history['val_loss']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)
    plt.subplot(1, 2, 1)
    # Visualize loss history for training data.
    plt.plot(epoch_count, training_loss, label='Train Loss', color='red')

    # View loss on unseen data.
    plt.plot(epoch_count, validation_loss, 'r--', label='Validation Loss',
             color='black')

    plt.xlabel('Epoch')
    plt.legend(loc="best")
    plt.title("Loss")
    plt.tight_layout()
    plt.show()


def evaluateModel(model, X_test, y_test, title):
    print("\n****** " + title)
    predictions = model.predict(X_test)
    print('Root Mean Squared Error:',
          np.sqrt(mean_squared_error(y_test, predictions)))


def bagged_model_tuning(dataframe):
    X = dataframe[['year', 'm_body_type', 'm_transmission',
                   'body_type_convertible', 'body_type_coupe', 'body_type_hatchback',
                   'body_type_minivan', 'body_type_pickup', 'body_type_sedan', 'body_type_suv', 'body_type_wagon',
                   'transmission_Automatic', 'transmission_Manual', 'make_Audi', 'make_BMW', 'make_Bentley',
                   'make_Cadillac', 'make_Chevrolet', 'make_Dodge', 'make_Ford', 'make_GMC', 'make_Honda',
                   'make_Hummer', 'make_INFINITI', 'make_Jeep', 'make_Land Rover', 'make_Lexus',
                   'make_Lincoln', 'make_MINI', 'make_Maserati', 'make_Mercedes-Benz', 'make_Nissan',
                   'make_Porsche', 'make_RAM', 'make_Toyota', 'make_Volkswagen', 'make_Volvo', 'trim_Base',
                   'trim_Custom', 'trim_Sport', 'trim_Winter', 'trim_untrimmed', 'fuel_type_Diesel',
                   'fuel_type_Electric', 'fuel_type_Gas', 'fuel_type_Hybrid'
                   ]]

    X = sm.add_constant(X)

    y = dataframe['price']

    # X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    #
    # # Build linear regression ensemble.
    # ensembleModel = BaggingRegressor(base_estimator=LinearRegression(), max_features=4,
    #                                  max_samples=0.5,
    #                                  n_estimators=10).fit(X_train, y_train)
    # evaluateModel(ensembleModel, X_test, y_test, "Ensemble")
    #
    # # Build stand alone linear regression model.
    # model = LinearRegression()
    # model.fit(X_train, y_train)
    # evaluateModel(model, X_test, y_test, "Linear Regression")
    #
    # return model

    feature_combo_list = []

    def evaluateModel(model, X_test, y_test, title, num_estimators, max_features):
        print("\n****** " + title)
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        # Store statistics and add to list.
        stats = {"type": title, "rmse": rmse,
                 "estimators": num_estimators, "features": max_features}
        feature_combo_list.append(stats)

    num_estimator_list = [420, 425]
    max_features_list = [13, 14, 15]

    for num_estimators in num_estimator_list:
        for max_features in max_features_list:
            # Create random split.
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

            # Build linear regression ensemble.
            ensembleModel = BaggingRegressor(base_estimator=LinearRegression(),
                                             max_features=max_features,
                                             max_samples=0.5,
                                             n_estimators=num_estimators).fit(X_train, y_train)
            evaluateModel(ensembleModel, X_test, y_test, "Ensemble",
                          num_estimators, max_features)

            # Build stand alone linear regression model.
            model = LinearRegression()
            model.fit(X_train, y_train)
            evaluateModel(model, X_test, y_test, "Linear Regression", None, None)

    # Build data frame with dictionary objects.
    dfStats = pd.DataFrame()
    print(dfStats)
    for combo in feature_combo_list:
        dfStats = dfStats.append(combo, ignore_index=True)

    # Sort and show all combinations.
    # Show all rows
    pd.set_option('display.max_rows', None)
    dfStats = dfStats.sort_values(by=['type', 'rmse'])
    print(dfStats)


def bagged_model_tuned(dataframe):
    X = dataframe[['year', 'm_body_type', 'm_transmission',
                   'body_type_convertible', 'body_type_coupe', 'body_type_hatchback',
                   'body_type_minivan', 'body_type_pickup', 'body_type_sedan', 'body_type_suv', 'body_type_wagon',
                   'transmission_Automatic', 'transmission_Manual', 'make_Audi', 'make_BMW', 'make_Bentley',
                   'make_Cadillac', 'make_Chevrolet', 'make_Dodge', 'make_Ford', 'make_GMC', 'make_Honda',
                   'make_Hummer', 'make_INFINITI', 'make_Jeep', 'make_Land Rover', 'make_Lexus',
                   'make_Lincoln', 'make_MINI', 'make_Maserati', 'make_Mercedes-Benz', 'make_Nissan',
                   'make_Porsche', 'make_RAM', 'make_Toyota', 'make_Volkswagen', 'make_Volvo', 'trim_Base',
                   'trim_Custom', 'trim_Sport', 'trim_Winter', 'trim_untrimmed', 'fuel_type_Diesel',
                   'fuel_type_Electric', 'fuel_type_Gas', 'fuel_type_Hybrid'
                   ]]

    y = dataframe['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Build linear regression ensemble.
    ensembleModel = BaggingRegressor(base_estimator=LinearRegression(), max_features=15,
                                     max_samples=0.5,
                                     n_estimators=420).fit(X_train, y_train)
    evaluateModel(ensembleModel, X_test, y_test, "Ensemble")

    # Build stand alone linear regression model.
    model = LinearRegression()
    model.fit(X_train, y_train)
    evaluateModel(model, X_test, y_test, "Linear Regression")

    Pkl_Filename = "bagged_model.pkl"

    with open(Pkl_Filename, 'wb') as file:
        pickle.dump(model, file)

    return model


def performSGD(X_train, X_test, y_train, y_test, scalerY):
    sgd = SGDRegressor(verbose=1)
    sgd.fit(X_train, y_train)
    print("\n***SGD=")
    predictions = sgd.predict(X_test)
    # print(predictions)

    y_test_unscaled = scalerY.inverse_transform(y_test)
    predictions_unscaled = scalerY.inverse_transform(predictions.reshape(-1, 1))
    # print(predictions_unscaled)

    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test_unscaled,
                                             predictions_unscaled)))

    return sgd


def gradient_descent_model(dataframe):
    X = dataframe[['year', 'm_body_type', 'm_transmission',
                   'body_type_convertible', 'body_type_coupe', 'body_type_hatchback',
                   'body_type_minivan', 'body_type_pickup', 'body_type_sedan', 'body_type_suv', 'body_type_wagon',
                   'transmission_Automatic', 'transmission_Manual', 'make_Audi', 'make_BMW', 'make_Bentley',
                   'make_Cadillac', 'make_Chevrolet', 'make_Dodge', 'make_Ford', 'make_GMC', 'make_Honda',
                   'make_Hummer', 'make_INFINITI', 'make_Jeep', 'make_Land Rover', 'make_Lexus',
                   'make_Lincoln', 'make_MINI', 'make_Maserati', 'make_Mercedes-Benz', 'make_Nissan',
                   'make_Porsche', 'make_RAM', 'make_Toyota', 'make_Volkswagen', 'make_Volvo', 'trim_Base',
                   'trim_Custom', 'trim_Sport', 'trim_Winter', 'trim_untrimmed', 'fuel_type_Diesel',
                   'fuel_type_Electric', 'fuel_type_Gas', 'fuel_type_Hybrid'
                   ]].values

    y = dataframe['price'].values

    scalerX = MinMaxScaler()
    scalerX.fit(X)
    x2Scaled = scalerX.transform(X)

    scalerY = MinMaxScaler()
    reshapedY = y.reshape(-1, 1)
    scalerY.fit(reshapedY)
    yScaled = scalerY.transform(reshapedY)
    X_train, X_test, y_train, y_test = train_test_split(x2Scaled, yScaled, test_size=0.2)
    model = performSGD(X_train, X_test, y_train, y_test, scalerY)

    Pkl_Filename = "Gradient_Descent_Model.pkl"

    with open(Pkl_Filename, 'wb') as file:
        pickle.dump(model, file)

    return model


def ridge_regression(X_train, X_test, y_train, y_test, alpha):
    # Fit the model
    ridgereg = Ridge(alpha=alpha, normalize=True)
    ridgereg.fit(X_train, y_train)
    y_pred = ridgereg.predict(X_test)
    # predictions = scalerY.inverse_transform(y_pred.reshape(-1,1))
    print("\n***Ridge Regression Coefficients ** alpha=" + str(alpha))
    print(ridgereg.intercept_)
    print(ridgereg.coef_)
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

    return ridgereg


def ridge_model(dataframe):
    X = dataframe[['year', 'm_body_type', 'm_transmission',
                   'body_type_convertible', 'body_type_coupe', 'body_type_hatchback',
                   'body_type_minivan', 'body_type_pickup', 'body_type_sedan', 'body_type_suv', 'body_type_wagon',
                   'transmission_Automatic', 'transmission_Manual', 'make_Audi', 'make_BMW', 'make_Bentley',
                   'make_Cadillac', 'make_Chevrolet', 'make_Dodge', 'make_Ford', 'make_GMC', 'make_Honda',
                   'make_Hummer', 'make_INFINITI', 'make_Jeep', 'make_Land Rover', 'make_Lexus',
                   'make_Lincoln', 'make_MINI', 'make_Maserati', 'make_Mercedes-Benz', 'make_Nissan',
                   'make_Porsche', 'make_RAM', 'make_Toyota', 'make_Volkswagen', 'make_Volvo', 'trim_Base',
                   'trim_Custom', 'trim_Sport', 'trim_Winter', 'trim_untrimmed', 'fuel_type_Diesel',
                   'fuel_type_Electric', 'fuel_type_Gas', 'fuel_type_Hybrid'
                   ]].values

    y = dataframe['price'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2)
    # alphaValues = [0, 0.5, 0.6, 0.9, 0.1]
    # for i in range(0, len(alphaValues)):
    #     ridge_regression(X_train, X_test, y_train, y_test,
    #                      alphaValues[i])

    alphaValues = 0
    model = ridge_regression(X_train, X_test, y_train, y_test,
                             alphaValues)

    Pkl_Filename = "ridge_regression_model.pkl"

    with open(Pkl_Filename, 'wb') as file:
        pickle.dump(model, file)

    return model


def performLassorRegression(X_train, X_test, y_train, y_test, alpha):
    lassoreg = Lasso(alpha=alpha, normalize=True, max_iter=1e5)
    lassoreg.fit(X_train, y_train)
    y_pred = lassoreg.predict(X_test)
    print("\n***Lasso Regression Coefficients ** alpha=" + str(alpha))
    print(lassoreg.intercept_)
    print(lassoreg.coef_)
    print('Root Mean Squared Error:',
          np.sqrt(metrics.mean_squared_error(y_test, y_pred)))
    return lassoreg


def lasso_regression_tuning(dataframe):
    X = dataframe[['year', 'm_body_type', 'm_transmission',
                   'body_type_convertible', 'body_type_coupe', 'body_type_hatchback',
                   'body_type_minivan', 'body_type_pickup', 'body_type_sedan', 'body_type_suv', 'body_type_wagon',
                   'transmission_Automatic', 'transmission_Manual', 'make_Audi', 'make_BMW', 'make_Bentley',
                   'make_Cadillac', 'make_Chevrolet', 'make_Dodge', 'make_Ford', 'make_GMC', 'make_Honda',
                   'make_Hummer', 'make_INFINITI', 'make_Jeep', 'make_Land Rover', 'make_Lexus',
                   'make_Lincoln', 'make_MINI', 'make_Maserati', 'make_Mercedes-Benz', 'make_Nissan',
                   'make_Porsche', 'make_RAM', 'make_Toyota', 'make_Volkswagen', 'make_Volvo', 'trim_Base',
                   'trim_Custom', 'trim_Sport', 'trim_Winter', 'trim_untrimmed', 'fuel_type_Diesel',
                   'fuel_type_Electric', 'fuel_type_Gas', 'fuel_type_Hybrid'
                   ]].values

    y = dataframe['price'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2)

    # alphaValues = [0, 0.1, 0.5, 1]
    # for i in range(0, len(alphaValues)):
    #     performLassorRegression(X_train, X_test, y_train, y_test,
    #                             alphaValues[i])

    alphaValues = 1

    model = performLassorRegression(X_train, X_test, y_train, y_test, alphaValues)

    Pkl_Filename = "lasso_model.pkl"

    with open(Pkl_Filename, 'wb') as file:
        pickle.dump(model, file)

    return model


def performElasticNetRegression(X_train, X_test, y_train, y_test, alpha, l1ratio, bestRMSE,
                                bestAlpha, bestL1Ratio):
    model = ElasticNet(alpha=alpha, l1_ratio=l1ratio)
    # fit model
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\n***ElasticNet Regression Coefficients ** alpha=" + str(alpha)
          + " l1ratio=" + str(l1ratio))
    rmse = np.sqrt(metrics.mean_squared_error(y_test, y_pred))
    print(model.intercept_)
    print(model.coef_)
    try:
        if (rmse < bestRMSE):
            bestRMSE = rmse
            bestAlpha = alpha
            bestL1Ratio = l1ratio
        print('Root Mean Squared Error:', rmse)
    except:
        print("rmse =" + str(rmse))

    return bestRMSE, bestAlpha, bestL1Ratio


def elastic_net_model_tuning(dataframe):
    bestRMSE = 10122

    X = dataframe[['year', 'm_body_type', 'm_transmission',
                   'body_type_convertible', 'body_type_coupe', 'body_type_hatchback',
                   'body_type_minivan', 'body_type_pickup', 'body_type_sedan', 'body_type_suv', 'body_type_wagon',
                   'transmission_Automatic', 'transmission_Manual', 'make_Audi', 'make_BMW', 'make_Bentley',
                   'make_Cadillac', 'make_Chevrolet', 'make_Dodge', 'make_Ford', 'make_GMC', 'make_Honda',
                   'make_Hummer', 'make_INFINITI', 'make_Jeep', 'make_Land Rover', 'make_Lexus',
                   'make_Lincoln', 'make_MINI', 'make_Maserati', 'make_Mercedes-Benz', 'make_Nissan',
                   'make_Porsche', 'make_RAM', 'make_Toyota', 'make_Volkswagen', 'make_Volvo', 'trim_Base',
                   'trim_Custom', 'trim_Sport', 'trim_Winter', 'trim_untrimmed', 'fuel_type_Diesel',
                   'fuel_type_Electric', 'fuel_type_Gas', 'fuel_type_Hybrid'
                   ]].values

    y = dataframe['price'].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    alphaValues = [0, 0.00001, 0.0001, 0.001, 0.01, 0.18]
    l1ratioValues = [0, 0.25, 0.5, 0.75, 1]
    bestAlpha = 0
    bestL1Ratio = 0

    for i in range(0, len(alphaValues)):
        for j in range(0, len(l1ratioValues)):
            bestRMSE, bestAlpha, bestL1Ratio = performElasticNetRegression(
                X_train, X_test, y_train, y_test,
                alphaValues[i], l1ratioValues[j], bestRMSE,
                bestAlpha, bestL1Ratio)

    print("Best RMSE " + str(bestRMSE) + " Best alpha: " + str(bestAlpha)

          + "  " + "Best l1 ratio: " + str(bestL1Ratio))


def getUnfitModels():
    models = list()
    models.append(Lasso(alpha=1, normalize=True, max_iter=1e5))
    models.append(RandomForestRegressor(n_estimators=25))
    models.append(ExtraTreesRegressor(n_estimators=35))
    return models


def evaluate_model_stack(y_test, predictions, model):
    mse = mean_squared_error(y_test, predictions)
    rmse = round(np.sqrt(mse), 3)
    print(" RMSE:" + str(rmse) + " " + model.__class__.__name__)


def fitBaseModels(X_train, y_train, X_test, models):
    dfPredictions = pd.DataFrame()

    # Fit base model and store its predictions in dataframe.
    for i in range(0, len(models)):
        models[i].fit(X_train, y_train)
        predictions = models[i].predict(X_test)
        colName = str(i)
        # Add base model predictions to column of data frame.
        dfPredictions[colName] = predictions
    return dfPredictions, models


def fitStackedModel(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model


def stacked_model_tuning(dataframe):
    X = dataframe[['year', 'm_body_type', 'm_transmission',
                   'body_type_convertible', 'body_type_coupe', 'body_type_hatchback',
                   'body_type_minivan', 'body_type_pickup', 'body_type_sedan', 'body_type_suv', 'body_type_wagon',
                   'transmission_Automatic', 'transmission_Manual', 'make_Audi', 'make_BMW', 'make_Bentley',
                   'make_Cadillac', 'make_Chevrolet', 'make_Dodge', 'make_Ford', 'make_GMC', 'make_Honda',
                   'make_Hummer', 'make_INFINITI', 'make_Jeep', 'make_Land Rover', 'make_Lexus',
                   'make_Lincoln', 'make_MINI', 'make_Maserati', 'make_Mercedes-Benz', 'make_Nissan',
                   'make_Porsche', 'make_RAM', 'make_Toyota', 'make_Volkswagen', 'make_Volvo', 'trim_Base',
                   'trim_Custom', 'trim_Sport', 'trim_Winter', 'trim_untrimmed', 'fuel_type_Diesel',
                   'fuel_type_Electric', 'fuel_type_Gas', 'fuel_type_Hybrid'
                   ]].values

    y = dataframe['price'].values

    # Split data into train, test and validation sets.
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.20)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.50)

    # Get base models.
    unfitModels = getUnfitModels()

    # Fit base and stacked models.
    dfPredictions, models = fitBaseModels(X_train, y_train, X_test, unfitModels)
    stackedModel = fitStackedModel(dfPredictions, y_test)

    # Evaluate base models with validation data.
    print("\n** Evaluate Base Models **")
    dfValidationPredictions = pd.DataFrame()
    for i in range(0, len(models)):
        predictions = models[i].predict(X_val)
        colName = str(i)
        dfValidationPredictions[colName] = predictions
        evaluate_model_stack(y_val, predictions, models[i])

    # Evaluate stacked model with validation data.
    stackedPredictions = stackedModel.predict(dfValidationPredictions)
    print("\n** Evaluate Stacked Model **")
    evaluate_model_stack(y_val, stackedPredictions, stackedModel)

    Pkl_Filename = "stacked_model.pkl"
    with open(Pkl_Filename, 'wb') as file:
        pickle.dump(stackedModel, file)

    return stackedModel


def random_forest_model_tuning(dataframe):
    X = dataframe[['year', 'm_body_type', 'm_transmission',
                   'body_type_convertible', 'body_type_coupe', 'body_type_hatchback',
                   'body_type_minivan', 'body_type_pickup', 'body_type_sedan', 'body_type_suv', 'body_type_wagon',
                   'transmission_Automatic', 'transmission_Manual', 'make_Audi', 'make_BMW', 'make_Bentley',
                   'make_Cadillac', 'make_Chevrolet', 'make_Dodge', 'make_Ford', 'make_GMC', 'make_Honda',
                   'make_Hummer', 'make_INFINITI', 'make_Jeep', 'make_Land Rover', 'make_Lexus',
                   'make_Lincoln', 'make_MINI', 'make_Maserati', 'make_Mercedes-Benz', 'make_Nissan',
                   'make_Porsche', 'make_RAM', 'make_Toyota', 'make_Volkswagen', 'make_Volvo', 'trim_Base',
                   'trim_Custom', 'trim_Sport', 'trim_Winter', 'trim_untrimmed', 'fuel_type_Diesel',
                   'fuel_type_Electric', 'fuel_type_Gas', 'fuel_type_Hybrid'
                   ]]

    X = sm.add_constant(X)

    y = dataframe['price']

    feature_combo_list = []

    def evaluateModel(model, X_test, y_test, title, num_estimators):
        print("\n****** " + title)
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        # Store statistics and add to list.
        stats = {"type": title, "rmse": rmse,
                 "estimators": num_estimators}
        feature_combo_list.append(stats)

    num_estimator_list = [22, 25, 27, 30]

    for num_estimators in num_estimator_list:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Build linear regression ensemble.
        ensembleModel = RandomForestRegressor(n_estimators=num_estimators).fit(X_train, y_train)
        evaluateModel(ensembleModel, X_test, y_test, "RandomForestRegressor",
                      num_estimators)

    # Build data frame with dictionary objects.
    dfStats = pd.DataFrame()
    print(dfStats)
    for combo in feature_combo_list:
        dfStats = dfStats.append(combo, ignore_index=True)

    # Sort and show all combinations.
    # Show all rows
    pd.set_option('display.max_rows', None)
    dfStats = dfStats.sort_values(by=['type', 'rmse'])
    print(dfStats)


def random_forest_model_tuned(dataframe):
    X = dataframe[['year', 'm_body_type', 'm_transmission',
                   'body_type_convertible', 'body_type_coupe', 'body_type_hatchback',
                   'body_type_minivan', 'body_type_pickup', 'body_type_sedan', 'body_type_suv', 'body_type_wagon',
                   'transmission_Automatic', 'transmission_Manual', 'make_Audi', 'make_BMW', 'make_Bentley',
                   'make_Cadillac', 'make_Chevrolet', 'make_Dodge', 'make_Ford', 'make_GMC', 'make_Honda',
                   'make_Hummer', 'make_INFINITI', 'make_Jeep', 'make_Land Rover', 'make_Lexus',
                   'make_Lincoln', 'make_MINI', 'make_Maserati', 'make_Mercedes-Benz', 'make_Nissan',
                   'make_Porsche', 'make_RAM', 'make_Toyota', 'make_Volkswagen', 'make_Volvo', 'trim_Base',
                   'trim_Custom', 'trim_Sport', 'trim_Winter', 'trim_untrimmed', 'fuel_type_Diesel',
                   'fuel_type_Electric', 'fuel_type_Gas', 'fuel_type_Hybrid'
                   ]]

    X = sm.add_constant(X)

    y = dataframe['price']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Build linear regression ensemble.
    model = RandomForestRegressor(n_estimators=25).fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print('******* Random Forest Regressor')
    print('Root Mean Squared Error:', rmse)

    Pkl_Filename = "random_forest_regression_model.pkl"
    with open(Pkl_Filename, 'wb') as file:
        pickle.dump(model, file)

    return model


def extra_tree_model_tuned(dataframe):
    X = dataframe[['year', 'm_body_type', 'm_transmission',
                   'body_type_convertible', 'body_type_coupe', 'body_type_hatchback',
                   'body_type_minivan', 'body_type_pickup', 'body_type_sedan', 'body_type_suv', 'body_type_wagon',
                   'transmission_Automatic', 'transmission_Manual', 'make_Audi', 'make_BMW', 'make_Bentley',
                   'make_Cadillac', 'make_Chevrolet', 'make_Dodge', 'make_Ford', 'make_GMC', 'make_Honda',
                   'make_Hummer', 'make_INFINITI', 'make_Jeep', 'make_Land Rover', 'make_Lexus',
                   'make_Lincoln', 'make_MINI', 'make_Maserati', 'make_Mercedes-Benz', 'make_Nissan',
                   'make_Porsche', 'make_RAM', 'make_Toyota', 'make_Volkswagen', 'make_Volvo', 'trim_Base',
                   'trim_Custom', 'trim_Sport', 'trim_Winter', 'trim_untrimmed', 'fuel_type_Diesel',
                   'fuel_type_Electric', 'fuel_type_Gas', 'fuel_type_Hybrid'
                   ]]

    X = sm.add_constant(X)

    y = dataframe['price']

    feature_combo_list = []

    num_estimator = 35

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    # Build linear regression ensemble.
    model = ExtraTreesRegressor(n_estimators=num_estimator).fit(X_train, y_train)

    predictions = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    print('******* Extra Trees  Regressor')
    print('Root Mean Squared Error:', rmse)

    Pkl_Filename = "extra_tree_regression_model.pkl"
    with open(Pkl_Filename, 'wb') as file:
        pickle.dump(model, file)

    return model


def extra_tree_model_tuning(dataframe):
    X = dataframe[['year', 'm_body_type', 'm_transmission',
                   'body_type_convertible', 'body_type_coupe', 'body_type_hatchback',
                   'body_type_minivan', 'body_type_pickup', 'body_type_sedan', 'body_type_suv', 'body_type_wagon',
                   'transmission_Automatic', 'transmission_Manual', 'make_Audi', 'make_BMW', 'make_Bentley',
                   'make_Cadillac', 'make_Chevrolet', 'make_Dodge', 'make_Ford', 'make_GMC', 'make_Honda',
                   'make_Hummer', 'make_INFINITI', 'make_Jeep', 'make_Land Rover', 'make_Lexus',
                   'make_Lincoln', 'make_MINI', 'make_Maserati', 'make_Mercedes-Benz', 'make_Nissan',
                   'make_Porsche', 'make_RAM', 'make_Toyota', 'make_Volkswagen', 'make_Volvo', 'trim_Base',
                   'trim_Custom', 'trim_Sport', 'trim_Winter', 'trim_untrimmed', 'fuel_type_Diesel',
                   'fuel_type_Electric', 'fuel_type_Gas', 'fuel_type_Hybrid'
                   ]]

    X = sm.add_constant(X)

    y = dataframe['price']

    feature_combo_list = []

    def evaluateModel(model, X_test, y_test, title, num_estimators):
        print("\n****** " + title)
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))

        # Store statistics and add to list.
        stats = {"type": title, "rmse": rmse,
                 "estimators": num_estimators}
        feature_combo_list.append(stats)

    num_estimator_list = [30, 35, 40]

    for num_estimators in num_estimator_list:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        # Build linear regression ensemble.
        ensembleModel = ExtraTreesRegressor(n_estimators=num_estimators).fit(X_train, y_train)
        evaluateModel(ensembleModel, X_test, y_test, "RandomForestRegressor",
                      num_estimators)

    # Build data frame with dictionary objects.
    dfStats = pd.DataFrame()
    print(dfStats)
    for combo in feature_combo_list:
        dfStats = dfStats.append(combo, ignore_index=True)

    # Sort and show all combinations.
    # Show all rows
    pd.set_option('display.max_rows', None)
    dfStats = dfStats.sort_values(by=['type', 'rmse'])
    print(dfStats)


def train_and_generate_models(df_cleaned):
    # un-optimized raw model
    # raw_model(df_cleaned)

    #############################
    #  Linear model
    #############################
    # optimized linear model
    # linear_model(df_cleaned)

    #############################
    #  Neural model Tuning
    #############################
    # neural_model_num(df_cleaned)
    # neural_model_optimizer(df_cleaned)
    # neural_model_learning_rate(df_cleaned)
    # neural_model_initializer_tuning(df_cleaned)
    # neural_model_activation_tuning(df_cleaned)
    # neural_model_another_layer_tuning(df_cleaned)

    #############################
    #  Final Tuned Neural model
    #############################
    # neural_model = neural_model_tuned(df_cleaned)

    #############################
    #  Bagged model
    #############################
    # bagged_model_tuning(df_cleaned)
    # bagged_model = bagged_model_tuned(df_cleaned)

    #############################
    #  ML model
    #############################
    # gsd_model = gradient_descent_model(df_cleaned)

    # ridge_regression_model = ridge_model(df_cleaned)
    # lasso_regression_model = lasso_regression_tuning(df_cleaned)
    # elastic_net_model_tuning(df_cleaned)

    # random_forest_model_tuning(df_cleaned)
    # random_forest_model = random_forest_model_tuned(df_cleaned)
    # extra_tree_model_tuning(df_cleaned)
    # extra_tree_model = extra_tree_model_tuned(df_cleaned)

    #############################
    #  Stacked model
    #############################
    stacked_model = stacked_model_tuning(df_cleaned)


def main():

    # #########################################################
    #
    #  DATA IMPORT
    #
    # #########################################################
    # PATH = "/Users/xindilu/Desktop/COMP 4948 Predictive Analytics/a2/"
    # CSV_DATA = "ca-used-car.csv"
    #
    # data = import_data(PATH, CSV_DATA)


    # #########################################################
    #
    #  DATA OVERVIEW + EDA
    #
    # #########################################################
    # # # data overview
    # # acquire_data_description(data)
    # display_missing_value_summary(data)
    # # view EDA
    # EDA(data)


    # #########################################################
    #
    #  DATA TREATMENTS
    #
    # #########################################################
    # # remove null prices - not helping price predictions
    # df = data[~data['price'].isnull()]
    # # print('\n\nCheck if there\'s null price:')
    # # print(df['price'].isnull().sum())
    # # print('\n')
    #
    # # remove unrelated info such as vin, zip, address
    # column_name = ['price', 'miles', 'year', 'make', 'trim', 'body_type',
    #                'transmission', 'fuel_type']
    # df = df[column_name]
    # # standardise data to keep data consistency
    # df_standard = standardise_body_types(df)
    # # print(df_standard.head(10))
    #
    # # imputing missing values
    # df_imputed = prepare_imputed_dataset(df_standard)
    # display_missing_value_summary(df_imputed)
    #
    # # create a cleaned df that do not contain the imputed columns
    # col_names = ['body_type', 'transmission', 'make', 'trim', 'fuel_type', 'miles']
    # df_cleaned = filter_out_imputed_cols(df_imputed, col_names)


    # #########################################################
    #
    #  MODEL TRAINING SECTION
    # RAN THE MODEL WILL OVERWRITE THE SAVED MODELS
    #
    # #########################################################
    # train_and_generate_models(df_cleaned)

    ##########################################################
    #
    #  MODEL EVALUATION SECTION
    #
    ##########################################################

    my_car_data = {'m_miles': 80000, 'year': 2012, 'm_body_type': 0, 'm_transmission': 0, 'body_type_convertible': 0,
                   'body_type_coupe': 0, 'body_type_hatchback': 0,
                   'body_type_minivan': 0, 'body_type_pickup': 0, 'body_type_sedan': 1, 'body_type_suv': 0,
                   'body_type_wagon': 0,
                   'transmission_Automatic': 1, 'transmission_Manual': 0, 'make_Audi': 0, 'make_BMW': 0,
                   'make_Bentley': 0,
                   'make_Cadillac': 0, 'make_Chevrolet': 0, 'make_Dodge': 0, 'make_Ford': 0, 'make_GMC': 0,
                   'make_Honda': 1,
                   'make_Hummer': 0, 'make_INFINITI': 0, 'make_Jeep': 0, 'make_Land Rover': 0, 'make_Lexus': 0,
                   'make_Lincoln': 0, 'make_MINI': 0, 'make_Maserati': 0, 'make_Mercedes-Benz': 0, 'make_Nissan': 0,
                   'make_Porsche': 0, 'make_RAM': 0, 'make_Toyota': 0, 'make_Volkswagen': 0, 'make_Volvo': 0,
                   'trim_Base': 0,
                   'trim_Custom': 1, 'trim_Sport': 0, 'trim_Winter': 0, 'trim_untrimmed': 0, 'fuel_type_Diesel': 0,
                   'fuel_type_Electric': 0, 'fuel_type_Gas': 1, 'fuel_type_Hybrid': 0}

    my_car = pd.DataFrame(my_car_data, index=[0])

    # load models
    stack_file = 'stacked_model.pkl'
    with open(stack_file, 'rb') as file:
        stack_model = pickle.load(file)

    lasso_file = 'lasso_model.pkl'
    with open(lasso_file, 'rb') as file:
        lasso_model = pickle.load(file)

    rf_file = 'random_forest_regression_model.pkl'
    with open(rf_file, 'rb') as file:
        rf_model = pickle.load(file)

    et_file = 'extra_tree_regression_model.pkl'
    with open(et_file, 'rb') as file:
        et_model = pickle.load(file)

    # make individual predictions
    lasso_data = my_car[['year', 'm_body_type', 'm_transmission',
                         'body_type_convertible', 'body_type_coupe', 'body_type_hatchback',
                         'body_type_minivan', 'body_type_pickup', 'body_type_sedan', 'body_type_suv', 'body_type_wagon',
                         'transmission_Automatic', 'transmission_Manual', 'make_Audi', 'make_BMW', 'make_Bentley',
                         'make_Cadillac', 'make_Chevrolet', 'make_Dodge', 'make_Ford', 'make_GMC', 'make_Honda',
                         'make_Hummer', 'make_INFINITI', 'make_Jeep', 'make_Land Rover', 'make_Lexus',
                         'make_Lincoln', 'make_MINI', 'make_Maserati', 'make_Mercedes-Benz', 'make_Nissan',
                         'make_Porsche', 'make_RAM', 'make_Toyota', 'make_Volkswagen', 'make_Volvo', 'trim_Base',
                         'trim_Custom', 'trim_Sport', 'trim_Winter', 'trim_untrimmed', 'fuel_type_Diesel',
                         'fuel_type_Electric', 'fuel_type_Gas', 'fuel_type_Hybrid'
                         ]].values
    lasso_result = lasso_model.predict(lasso_data)[0]
    print('*** lasso model prediction:')
    print(lasso_result)

    rf_result = rf_model.predict(my_car)[0]
    print('*** random forest model prediction:')
    print(rf_result)

    et_result = et_model.predict(my_car)[0]
    print('*** extra trees model prediction:')
    print(et_result)

    data_stack = {'0': lasso_result, '1': rf_result, '2': et_result}
    df_stack = pd.DataFrame(data_stack, index=[0])

    my_car_predicted_price = stack_model.predict(df_stack)[0]
    print('*** STACK model prediction:')
    print(my_car_predicted_price)


if __name__ == '__main__':
    main()
