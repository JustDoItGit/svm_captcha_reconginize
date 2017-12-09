from PIL import Image
import numpy as np
from sklearn.externals import joblib
from 测试模型.config import *

def tes1t1_data(file_path):
    img = Image.open(file_path)
    data = img.getdata()
    data = np.matrix(data) / 255
    data1 = data.tolist()
    data2 = []
    for i in data1:
        data2.extend(i)
    return [data2]


m = tes1t1_data(img_path)
svc_3 = joblib.load(model_path)
a = svc_3.predict(m)
print(a)