import cv2
import os
import numpy as np

dataPath = 'D:\Documento\Cuarto Semestre\Semana Tec TC1001S.1\Aplicación\Data'
peoplelist= os.listdir (datapath)
print ("Lista de personas:", peoplelist)

lables=[]
faceData=[]
label=0

for nameDir in peoplelist:
    personPath= dataPath + '/' + nameDir
    print ('Leyendo las imagenes')

    for fileName in os.listdir(personPath):
        print ('Rostros:', nameDir + '/' +fileName)
        labels.append (label)
        faceData.append(cv2.imread (personPath+'/'+fileName,0))
        image= cv2.imread(personPath+'/'+fileName, 0)
        cv2.imshow( "image", image)
        cv2.waiKey(10)

    label= label +1

print ('labels',labels)
print ('Número de etiquetas 0: ' np.count_nonzero(np.array(labels)==0))
print ('Número de etiquetas 1: ' np.count_nonzero(np.array(labels)==1))

face_recognizer = cv2.face.EigenFaceRecognizer_create()

#Entrenando el reconocedor de rostros
print('Entrenando...')
face_recognizer.train(facesData, np.array(lables))

#Almacenamiendo el modelo obtenido
face_recognizer.write('modeloEigenFace.xml')
print ('Modelo almacenado...')

