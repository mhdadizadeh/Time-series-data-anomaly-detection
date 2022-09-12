import os
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.preprocessing import PolynomialFeatures
from statsmodels.tsa.seasonal import seasonal_decompose

input_path = os.environ.get('DATA_104_PATH') + '/test_data'
output_path = '/output'


class TimeSeriesAnomalyDetector:

    def __call__(self, df):

        main_df = df.copy()
        main_df['value'] += abs(main_df['value'].min()) + 1
        main_df['label'] = None


        main_df['predict_1'] = self.regression_outlier_model(main_df)
        main_df['predict_2'] = self.neighborhood_distance(main_df)
        main_df['predict_3'] = self.decomposition_model(main_df)

        main_df['label'] = main_df['predict_1'] | main_df['predict_2'] | main_df['predict_3'] 

        main_df['label'] = main_df['label'].replace(False, 0)
        main_df['label'] = main_df['label'].replace(True, 1)     

        
        return main_df[['timestamp', 'value', 'label']]


    def linear_detect_outliers(self, input_values):

        values = input_values.copy()

        Q1 = np.percentile(values, 23, interpolation = 'midpoint')  
        Q3 = np.percentile(values, 73, interpolation = 'midpoint')
        IQR = Q3 - Q1

        upper = values >= (Q3+1.5*IQR)
        lower = values <= (Q1-1.5*IQR)

        outliers = upper | lower 

        return outliers


    def decompose_detect_outliers(self, input_values):

        values = input_values.copy()

        Q1 = np.percentile(values, 4, interpolation = 'midpoint')  
        Q3 = np.percentile(values, 96, interpolation = 'midpoint')
        IQR = Q3 - Q1

        upper = values >= (Q3+1.5*IQR)
        lower = values <= (Q1-1.5*IQR)

        outliers = upper | lower 

        return outliers
      
    
    def neighbor_detect_outliers(self, input_values):
      
        values = input_values.copy()

        Q1 = np.percentile(values, 3, interpolation = 'midpoint')  
        Q3 = np.percentile(values, 97, interpolation = 'midpoint')
        IQR = Q3 - Q1

        upper = values >= (Q3+1.5*IQR)
        lower = values <= (Q1-1.5*IQR)

        outliers = upper | lower 

        return outliers
        

    def neighborhood_distance(self, input_df):
        
        df = input_df.copy()

        df['left_dist'] = (df['value'] - df['value'].shift(1)).abs()
        df['right_dist'] = (df['value'] - df['value'].shift(-1)).abs()
        df['neighbor_dist'] = df[['left_dist', 'right_dist']].min(axis=1)
        df['predict'] = self.neighbor_detect_outliers(df['neighbor_dist'])
        
        return df['predict']


    def regression_outlier_model(self, input_df):
    
        df = input_df.copy()
        no_slope_value = df[['value']] - self.Polynomial_line(df)
        return list(self.linear_detect_outliers(no_slope_value['value']))   
   
   
    def linear_regression(self, input_df):

        df = input_df.copy()
        regr = linear_model.LinearRegression()
        train_x = np.array(df[['timestamp']])
        train_y =  np.array(df[['value']])
        regr.fit (train_x, train_y)
        regression_line = (regr.coef_[0][0]*train_x + regr.intercept_[0])
        print('line: ',regression_line)

        return regression_line

    def Polynomial_line(self, input_df):

        df = input_df.copy()

        train_x = np.array(df[['timestamp']])
        train_y =  np.array(df[['value']])
        
        poly = PolynomialFeatures(degree=2)
        train_x_poly = poly.fit_transform(train_x)

        regr = linear_model.LinearRegression()
        train_y_ = regr.fit(train_x_poly, train_y)
        XX = np.arange(0.0, len(df.index))
        regression_line = (regr.intercept_[0]+ regr.coef_[0][1]*XX+ regr.coef_[0][2]*np.power(XX, 2))
        line_size = len(df.index)
        regression_line = np.reshape(regression_line, (line_size, 1))

        return regression_line


    def trend_line(self, input_df):

        initial_df = input_df.copy()
        df = initial_df[['timestamp', 'value']]
        df = df.set_index('timestamp')

        additive_decomposition = seasonal_decompose(df, model='additive', period=30)

        trend = additive_decomposition.trend
        trend[0:30] = trend.iloc[31]
        trend[-30:0] = trend.iloc[-31]
        trend = np.array(trend)
        line_size = len(trend)
        trend = np.reshape(trend, (line_size, 1))
      
        return trend


    def decomposition_model(self, input_df):

        initial_df = input_df.copy()
        temp_df = input_df.copy()
        df = initial_df[['timestamp', 'value']]
        df = df.set_index('timestamp')

        additive_decomposition = seasonal_decompose(df, model='additive', period=30)

        residual = additive_decomposition.resid
        residual = list(residual.fillna(residual.mean()))
        temp_df['value'] = residual

        return  self.decompose_detect_outliers(residual) 

          

anomaly_detector = TimeSeriesAnomalyDetector()

if __name__ == '__main__':

    filename_list = ['27.csv', '28.csv']
    for filename in os.listdir(input_path):
        input_df = pd.read_csv(os.path.join(input_path, filename))
        print(filename, len(input_df))
        result = anomaly_detector(input_df)
        result.to_csv(os.path.join(output_path, filename))
        print(f'item {filename} processed.')