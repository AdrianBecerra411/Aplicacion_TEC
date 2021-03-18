
import cv2
import os
import imutils
personName = "Adrian"
datapath = 'C:/Users/windw/Documents/proyecto_Tec/Aplicacion/Aplicacion_TEC/Reconocimiento_Facial/Data'
personPath=datapath+'/'+ personName
print(personPath)
if not os.path.exists(personPath):
	print('Carpeta Creada: ', personPath)
	os.makedirs(personPath)
#cap=cv2.VideoCapture('D:\Documento\Cuarto Semestre\Semana Tec TC1001S.1\Recocimiento Facial\Yo.mp4')
cap=cv2.VideoCapture(0)
faceClassif=cv2.CascadeClassifier(cv2.data.haarcascades+'haarcascade_frontalface_default.xml')
contador=0
while True:
	ret,frame=cap.read()
	print(frame.shape)
	if ret==False:break
	frame=imutils.resize(frame,width=640)
	gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
	auxFrame=frame.copy()
	faces=faceClassif.detectMultiScale(gray,1.3,5)
	for(x,y,w,h) in faces:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
		rostro=auxFrame[y:y+h,x:x+w]
		rostro=cv2.resize(rostro,(150,150),interpolation=cv2.INTER_CUBIC)
		cv2.imwrite(personPath+'/rostro_{}.jpg'.format(contador),rostro)
		contador=contador+1
	cv2.imshow('frame',frame)

	k=cv2.waitKey(1)
	if k==27 or contador>=300:
		break
cap.release()
cv2.destroyAllWindows()