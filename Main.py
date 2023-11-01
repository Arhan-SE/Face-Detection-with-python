import cv2
from random import randrange

data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


#img = cv2.imread('p1.jpg')
#r_img = cv2.resize(img, (400, 500), interpolation=cv2.INTER_LINEAR) 
vedio = cv2.VideoCapture(0)

while True:
    ret,frame=vedio.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    coordinates = data.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    

    for x,y,w,h in coordinates:
        cv2.rectangle(frame,(x,y),(x+w,x+h),randrange(100,255),randrange(100,255),randrange(100,255),2)

    cv2.imshow("p",frame) 
   

    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break
vedio.release()
cv2.destroyAllWindows()