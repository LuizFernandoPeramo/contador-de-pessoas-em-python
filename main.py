import numpy as np
import cv2

# achar o centro do retangulo do contorno
def center(x, y, w, h):
    x1 = int(w / 2)
    y1 = int(h / 2)
    cx = x + x1
    cy = y + y1
    return cx,cy


cap = cv2.VideoCapture('1.mp4')

fgbg = cv2.createBackgroundSubtractorMOG2()

detects = []

posL = 150
offset = 30

xy1 = (20, posL)
xy2 = (300, posL)


total = 0

up = 0
down = 0


while 1:
    ret, frame = cap.read()
    
    #passando a imagem para cinza
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #criando uma mascara 
    fgmask = fgbg.apply(gray)
    
    #tirando sobras da imagem e deixando a imagem binaria branca e preta
    retval, th = cv2.threshold(fgmask, 200, 255, cv2.THRESH_BINARY)
    
    #transformando a estrutura em uma matriz (5,5) em forma de elipse
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    
    #é usado  na remoção de ruído aparando os pixel para depois aumentar a imagem
    opening = cv2.morphologyEx(th, cv2.MORPH_OPEN, kernel, iterations = 2)
    
    #dilatar os pixel aparados no filtro anterior
    dilation = cv2.dilate(opening,kernel,iterations = 8)
    
    #é usado para fechar pequenos orifícios dentro dos objetos
    closing = cv2.morphologyEx(dilation, cv2.MORPH_CLOSE, kernel, iterations = 8)
    
    #desenha as linhas
    cv2.line(frame,xy1,xy2,(255,0,0),3)
    
    #linha para contar apenas quando passar da linha azul para a offset
    cv2.line(frame,(xy1[0],posL-offset),(xy2[0],posL-offset),(255,255,0),2)

    cv2.line(frame,(xy1[0],posL+offset),(xy2[0],posL+offset),(255,255,0),2)
    
    #destaca o contorno do objeto
    contours, hierarchy = cv2.findContours(closing,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    i = 0
    for cnt in contours:
        (x,y,w,h) = cv2.boundingRect(cnt)

        area = cv2.contourArea(cnt)
        
        #filtro para delimitar a area minima do objeto
        if int(area) > 3000 :
        
            centro = center(x, y, w, h)
            
            # texto do id
            cv2.putText(frame, str(i), (x+5, y+15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),2)
            #desenha um circulo no centro do retangulo
            cv2.circle(frame, centro, 4, (0, 0,255), -1)
            
            #desenha o retangulo em volta do contorno
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            if len(detects) <= i:
                detects.append([])
            if centro[1]> posL-offset and centro[1] < posL+offset:
                detects[i].append(centro)
            else:
                detects[i].clear()
            i += 1
            
            
    if i == 0:
        detects.clear()

    i = 0

    if len(contours) == 0:
        detects.clear()

    else:

        for detect in detects:
            for (c,l) in enumerate(detect):

                #subida
                if detect[c-1][1] < posL and l[1] > posL :
                    detect.clear()
                    up+=1
                    total+=1
                    cv2.line(frame,xy1,xy2,(0,255,0),5)
                    continue

                #descida
                if detect[c-1][1] > posL and l[1] < posL:
                    detect.clear()
                    down+=1
                    total+=1
                    cv2.line(frame,xy1,xy2,(0,0,255),5)
                    continue

                if c > 0:
                    cv2.line(frame,detect[c-1],l,(0,0,255),1)   
                    
    cv2.putText(frame, "TOTAL: "+str(total), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255),2)
    cv2.putText(frame, "SUBINDO: "+str(up), (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0),2)
    cv2.putText(frame, "DESCENDO: "+str(down), (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255),2)         
    
    #cv2.imshow("fgmask", fgmask)
    cv2.imshow("frame", frame)
    #cv2.imshow("gray", gray)
    #cv2.imshow("th", th)
    #cv2.imshow("opening", opening)
    #cv2.imshow("dilation", dilation)
    cv2.imshow("closing", closing)

    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()