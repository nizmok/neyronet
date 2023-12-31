import numpy as np
import pandas as pd
import os

print('Hi Neyronet')

directory = 'C:/Users/Vitaly/YandexDisk/!Текучка/!!!Нейросети Питон Програмирование/Skynet/' # устанавливаю рабочую папку
                                # выбрал и скачал csv файл для анализа временных рядов, конкретно фьючерсы сбербанка
df = pd.read_csv('test.csv')    # считал файл в пандас

print(df.head(5))               # просматриваю первые 5 строк файла
'''
     <DATE>  <TIME>   <OPEN>   <HIGH>    <LOW>  <CLOSE>  <VOL>
0  20080109  103000  10401.0  10401.0  10250.0  10398.0    318
1  20080109  104500  10374.0  10450.0  10374.0  10411.0    235
2  20080109  110000  10420.0  10465.0  10400.0  10400.0    712
3  20080109  111500  10410.0  10411.0  10380.0  10380.0    157
4  20080109  113000  10380.0  10410.0  10349.0  10355.0    158
'''

print(df.dtypes)                # вывел на экран структуру данных, поля <DATE> и <TIME> преобразовались как целые числа
'''                             # остальные правильно
<DATE>       int64
<TIME>       int64
<OPEN>     float64
<HIGH>     float64
<LOW>      float64
<CLOSE>    float64
<VOL>        int64
dtype: object
'''
df['<DATE>'] = pd.to_datetime(df['<DATE>'], format='%Y%m%d')            # преобразую столбец <DATE> в формат datetime
print(df.head(5))
print(df.dtypes)

'''
<DATE>     datetime64[ns]
<TIME>              int64
<OPEN>            float64
<HIGH>            float64
<LOW>             float64
<CLOSE>           float64
<VOL>               int64
dtype: object
'''

df.to_csv('test2.csv')                                                   # сохраняю результат в другом файле




