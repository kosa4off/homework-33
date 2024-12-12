import dill
import pandas as pd
import os
from datetime import datetime

# Задаем локальный путь
#path = "C:/Users/veter/airflow_hw"
path = os.environ.get('PROJECT_PATH', '.')

# Считываем модель из файла pickle
with open(f'{path}/data/models/cars_pipe_202412112016.pkl', 'rb') as file:
    model = dill.load(file)

# Обходим все файлы json и делаем предсказания, объединяя в один датафрейм и сохраняя в .csv
def predict():
    df_pred = pd.DataFrame()
    for elem in os.listdir(f'{path}/data/test'):
        if elem.endswith('.json'):
            with open(f'{path}/data/test/{elem}', 'rb') as json_file:
                df = pd.read_json(json_file, orient='index').transpose()
                df_y = pd.DataFrame([[df['id'][0], model.predict(df)[0]]])
                df_pred = pd.concat([df_pred, df_y], axis=0)
    df_pred.columns = ['car_id', 'pred']
    df_pred.to_csv(f'{path}/data/predictions/preds_{datetime.now().strftime("%Y%m%d%H%M")}.csv', index=False)

    return df_pred

if __name__ == '__main__':
    predict()
