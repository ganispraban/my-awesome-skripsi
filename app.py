from email.mime import base
from pyexpat import model
from unittest import result
from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import os 
import math
import cv2
import pickle
from tensorflow.keras.models import load_model
from keras.preprocessing.image import img_to_array
from matplotlib import pyplot as plt

from skimage.feature import graycomatrix,graycoprops

from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)



def loadDataset(filename, trainingSet=[] ,trainingClass=[]):
    arr=[]
    names = ['sumValue','contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy','hue','value', 'saturaton','path','label']
    dataset = pd.read_csv(filename,names=names)
    X = dataset.iloc[:, :-2].values
    y = dataset.iloc[:, 10].values
    for i in range(1,len(dataset)):
        trainingSet.append(X[i])
        trainingClass.append(y[i])
    print(len(X))

def feature_extract(im,features=[]):
    slices=[]
    proList = ['contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy']
    #featlist = ['sumValue','contrast', 'dissimilarity', 'homogeneity', 'ASM', 'energy','hue', 'saturaton','value']
    properties =np.zeros(6)
    glcmMatrix = []
    

    img=cv2.imread(im,1)
    #print('Original Dimensions : ',img.shape)
    dim = (32, 32)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    #print('Resized Dimensions : ',resized.shape)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(resized, cv2.COLOR_BGR2HSV)
    
    lower_red = np.array([10,50,50])
    upper_red  = np.array([20,255,255])
    mask = cv2.inRange(hsv,lower_red,upper_red)
    res = cv2.bitwise_and(resized,resized,mask=mask)
    plt.imshow(res,cmap='Blues', interpolation = 'bicubic')
    plt.xticks([]),plt.yticks([])
    #plt.show()

    
    sum=0
    result=res.flatten()
    #print(len(result))
    for k in range(3072):
        sum+=result[k]^2
    #print(sum)
    sqsum=math.sqrt(sum)
    print(sqsum)
    features.append(sqsum)

    h,s,v = cv2.split(hsv)

    h_mean = np.mean(h)
    s_mean = np.mean(s)
    v_mean = np.mean(v)

    glcmMatrix = (graycomatrix(gray_image, [1], [0], levels=2 ** 8))
    #print(glcmMatrix)
        
    for j in range(0, len(proList)):
        properties[j] = (graycoprops(glcmMatrix, prop=proList[j]))
        features.append(properties[j])

    features.append(h_mean)
    features.append(s_mean)
    features.append(v_mean)
    

def knn(trainingSet ,trainingClass,features,k):
    
    row3=[]
    row2=[]
    row1=[]
    for training_instance,train_instance in zip(trainingSet,trainingClass):
        distance=pow((float(features[0]) - float(training_instance[0])), 2)
        dist=math.sqrt(distance)
        row_dict={'class':train_instance,'value':dist}
        row1.append(row_dict)
    #print(len(row1))
    row2=sorted(row1, key = lambda i: i['value'])
    #print(len(row2))
        
    for i in range(k):
        row3.append(row2[i])
    test_array=np.asarray(row3)
    #print(test_array)
    
    skala0_count=0
    skala4_count=0
    for j in test_array:
        if(j.get('class')=='1'):
            healthy_count+=1
        else:
            skala4_count+=1
            #late_count+=1

    print("The data for test data %s is: "%j.get('name'))
    print("Skala 0=",skala0_count)
    print("Skala 4 =",skala4_count)    
        
    if(skala0_count >= skala4_count):
        diagno="Skala Kerusakan 0, tidak ada bagian tanaman daun bawang yang rusak"
    else:
        row3=[]
        row2=[]
        row1=[]
        for training_instance,train_instance in zip(trainingSet,trainingClass):
            distance=0
            for x in range(1,9):
                #print(tes_instance[x])
                distance += pow((float(features[x]) - float(training_instance[x])), 2)
            dist=math.sqrt(distance)
            row_dict={'class':train_instance,'value':dist}
            row1.append(row_dict)
        row2=sorted(row1, key = lambda i: i['value'])
        
        for i in range(k):
            row3.append(row2[i])

        test_array=np.asarray(row3)
        #print(test_array)
    
        early_count=0
        late_count=0
        healthy_count=0
        for j in test_array:
            if(j.get('class')=='0'):
                skala0_count+=1
            elif(j.get('class')=='1'):
                skala1_count+=1
            elif(j.get('class')=='2'):
                skala2_count+=1
            elif(j.get('class')=='3'):
                skala3_count+=1
            elif(j.get('class')=='4'):
                skala4_count+=1    

        print("The data for test data %s is: "%j.get('name'))
        print("Skala 0 / Sehat=",skala0_count)
        print("Skala 1 / Kerusakan Ringan =",skala1_count)
        print("Skala 2 / Kerusakan Sedang =",skala2_count)
        print("Skala 3 / Kerusakan Berat =",skala3_count)
        print("Skala 4 / Kerusakan Sangat Berat =",skala4_count)
        
        
        if(skala0_count >= skala4_count):
            diagno="Early_Blight \n Remedies: \n # Burn or bag infected plant parts. Do NOT compost. \n# Drip irrigation and soaker hoses can be used to help keep the foliage dry."
        else:
            diagno="Late_Blight \n Remedies: # Monitor the field, remove and destroy infected leaves. # Treat organically with copper spray. # - Use chemical fungicides,the best of which for potato is chlorothalonil."
    return diagno
    os.remove('resized.jpg')



@app.route("/", methods = ['GET'])
def index():
    return render_template('index.html')

@app.route("/predict", methods = ['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(os.path.realpath('__file__'))
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        features=[]
        trainingSet=[];
        trainingClass=[];
        training_file = 'feature_dataset.csv'
        loadDataset(training_file,trainingSet,trainingClass)
        k = 10

        feature_extract(file_path,features)
        print(features)
        result = knn(trainingSet,trainingClass,features,k)

        print(result)
        return result

    return None

if __name__=='__main__':
    app.run(debug=True,port=5000)