from cmath import polar
from dataclasses import dataclass
import math
from  PIL import Image, ImageDraw
import random
from typing import List
import cv2
import numpy as np

# https://www.khanacademy.org/computer-programming/spin-off-of-tracing-a-magnetic-field/6635312282288128

@dataclass
class Polo:
    x : int  
    y : int  
    polarity : int      
    # polarity type:
    #       0 :  negative --- positive
    #       1 :  positive --- negative


    def __init__(self, x, y, polarity):
        self.x = x
        self.y = y
        self.polarity = polarity

@dataclass
class Magnet:
    pole :  List[Polo]
    angle : int
    strength : int
    radio : int
    # polaridad: int 
    fictice: int

    # def __init__(self,cx,cy,width,angle,strength, radio, polarity = 0 ):
    #     self.angle = angle
    #     self.x1 = cx 
    #     self.y1 = cy
    #     self.x2 = self.x1 + width*math.cos(angle*math.pi/180)
    #     self.y2 = self.y1 + width*math.sin(angle*math.pi/180)
        
    #     self.pole = []
    #     self.pole.append( Polo(self.x1, self.y1) )
    #     self.pole.append( Polo(self.x2, self.y2) )

    #     self.strength = strength 
    #     self.radio = radio
    #     # polarity type:
    #     #       0 :  negative --- positive
    #     #       1 :  positive --- negative
    #     #       2 :  negative --- negative
    #     #       3 :  positive --- positive
    #     self.polarity = polarity 

    #     self.fictice = False

    def __init__(self,p1, p2, strength, radio, polarity = 0 ):
        self.angle = math.atan2(p2.y-p1.y, p2.x-p1.x)* math.pi/180 + 90
        self.pole = []
        self.pole.append( p1 )
        self.pole.append( p2 )

        self.strength = strength 
        self.radio = radio
        self.fictice = False


class Magnetic_Field:
    canvas = None
    image = None
    bag_magnets = [] 
    bag_fictice_magnets = []
    size_line = 7
    width = None
    height = None

    def __init__(self, image ):
        self.image = image
        self.canvas = ImageDraw.Draw(self.image)
        self.width, self.height = image.size
        
        self.bag_magnets = []

    def __plot_poles(self, magnet ):
        colors = ['blue', "red"]
        
        for i in range(2):
            
            self.canvas.ellipse((magnet.pole[i].x-5, magnet.pole[i].y-5,
                                magnet.pole[i].x+5, magnet.pole[i].y+5),    
                                fill = colors[ magnet.pole[i].polarity ] , outline =colors[ magnet.pole[i].polarity ] )


        self.canvas.line( (magnet.pole[0].x, magnet.pole[0].y,
                                magnet.pole[1].x, magnet.pole[1].y  ), fill='gold', width=2)
    
    
        # self.canvas.ellipse((magnet.pole[0].x-magnet.radio,
        #                     magnet.pole[0].y-magnet.radio,
        #                     magnet.pole[1].x+magnet.radio,
        #                     magnet.pole[1].y+magnet.radio  ), outline ='gold')

    def __distance(self, polo, magnet ):
        dx = polo.x - magnet.pole[0].x
        dy = polo.y - magnet.pole[0].y
        r1 = math.sqrt(dx*dx + dy*dy)

        dx = polo.x - magnet.pole[1].x
        dy = polo.y - magnet.pole[1].y
        r2 = math.sqrt(dx*dx + dy*dy)

        r = int(math.sqrt(r1**2 +r2**2))//2

        if magnet.fictice and r > magnet.radio//2.05:
            r = 10000000

        return r, r1, r2
    #Vector de campo eléctrico como componentes separados (dx,dy),(dx,dy)
    def __pose_vector_field( self, x, y , magnet ): #este vector de la pose entre ambos polos como se observa en la simulacion
        dx = x - magnet.pole[0].x             #para los polos del primer iman?
        dy = y - magnet.pole[0].y
        theta1 = math.atan2(dy, dx) + math.pi/2 #devuelve los valores en radianes de dy y dx(derivadas)
        r1 = math.sqrt(dx*dx + dy*dy)           #devuelve la raiz cuadrada de la derivada de los puntos de x y y del primer dibujo 
        f1 = 1 if r1 == 0 else magnet.strength / r1 #

        dx = x - magnet.pole[1].x             #para los polos del segundo iman?
        dy = y - magnet.pole[1].y               
        theta2 = math.atan2(dy, dx) - math.pi/2 #devuelve los valores en radianes de dy y dx(derivadas)
        r2 = math.sqrt(dx*dx + dy*dy)           #devuelve la raiz cuadrada de la derivada de los puntos de x y y del primer dibujo 
        f2 = 1 if r2 == 0 else magnet.strength / r2

        mx = f1 * math.cos(theta1) + f2 * math.cos(theta2)
        my = f1 * math.sin(theta1) + f2 * math.sin(theta2)
        theta = math.atan2(my, mx)

        dx = math.cos(theta)*self.size_line
        dy = math.sin(theta)*self.size_line

        # TODO: update to ellipsoidal influence area
        r = int( math.sqrt((r1**2+r2**2))/2 )
        flag =  True if  r < magnet.radio  else 0

        return [dx, dy, theta, r, flag] #retorna el valor de dx,dy,theta,r,flag

    def __plot_magnet_field(self, magnet):#Dibujar las líneas de flujo con mapa de colores y estilos apropiados.
     
        # if magnet.fictice:
            for i in range(20,self.width,15):
                for j in range(20,self.height,15):
                    pose = self.__pose_vector_field( i, j, magnet )
                    if pose[4]:
                        g = 255 - pose[3]*1.5 if 255 - pose[3]*1.5 > 0 else 0
                        #g = 255
                        self.canvas.line( (i+pose[1] + random.randint(-2, 2), j-pose[0], i-pose[1],j+pose[0] + random.randint(-2, 2)),
                                            fill=(0,int(g),0), width=1 )
            self.__plot_poles(magnet)
        

    # # agrega una fuerza magentico si esta cerano a otro polo para hacer la representacion de ambos
    def __add_ficticial_magnet( self, pole, r1, r2, magnet, list_fictice_magnets ): #14 trae parametros de polo1.r1.r2.magnest,lista de magnets
        idx_min_dist = 0 if r1 < r2 else 1 #15 si r1 es menor que r2 idx_min_dist es igual a 0 sino igual a 1
        near_pole = magnet.pole[idx_min_dist] #16 near polo almacenara los mismo que magnetismo tiene en su posicion idx_min_dist

        r = math.sqrt( (near_pole.x - pole.x)**2 + (near_pole.y - pole.y)**2 )*1.0 #17 calcula la raiz del set de variables near pole (x - x)(near pole y - near pole y)

        fictional_magnet  = Magnet( Polo(0, 0,0), Polo(0, 0,1), #18 fictional_magnetico es igual a polo 
                                    magnet.strength, r, # 19 crea la fuerza magentica
                                    0 ) # TODO: fix polarity
        fictional_magnet.pole = [] #20 Declara la lista de magnetica de pole
        fictional_magnet.pole.append( near_pole ) # 21 abre la lista y le manda el polo mas cercano
        fictional_magnet.pole.append( pole ) #22 tambien le manda el polo actual en el que se encuentra 
        fictional_magnet.angle = math.atan2( pole.y - near_pole.y, pole.x - near_pole.x ) - math.pi/2 # 23 devuelve el valor en radianes de pole y, pole-ccercano y , pole x y pole cercano x y lo divide entre pi/ por los radianes
        fictional_magnet.fictice = True #24 declara que si hay un polo cercano 

        list_fictice_magnets.append( fictional_magnet ) # 25 agrega un polo y lo vueve a repetir como un ciclo en laa linea 161

    def __populate_bag_fictice_magnets(self): #8.- ficticios polos magneticos
        self.bag_fictice_magnets = [] #9.- bag_magnetes es una list del set de Magnetic_Field linea 69

        if len( self.bag_magnets ) > 1: #10.- la dividimos para corroborar que hay mas de 1 magnetismo 
            # check  distances between poles for add virtual magnets # en real time
            for i, magnet in enumerate(self.bag_magnets): #for en i,magnet enumerado en (self.bag_magentismo list)
                for j in range(i+1, len(self.bag_magnets)): #for wn j en el rango de (i +1 del for de arriba, con la separacion de la lista con len)
                    for idx in range(2): # 11.- proporciona una secuencia de pares en idx para la iteracion
                        pole = self.bag_magnets[j].pole[idx] #pole sera igual a la lista del set de Magnetic_Field linea 69
                        d, r1, r2 = self.__distance(pole, magnet)#hacemos la declaracion de 3 variables para self distance que hace el vector entre ambas lineas
                        if d < magnet.radio*1.4: # 12.- con los valores regresados de la funcion 120 __pose_vector_field 
                            self.__add_ficticial_magnet(pole, r1, r2, magnet, self.bag_fictice_magnets) #13.- trae la funtion donde agrega el campo magnetico __add_ficticial_magnet 161

    def __populate_field(self): #
        list_magnets = []
        list_magnets.extend( self.bag_fictice_magnets )
        list_magnets.extend( self.bag_magnets )

        if list_magnets: #for en time real como en handslarkmar? 
            for i in range(20,self.width,15):
                for j in range(20,self.height,15):
                    point = Polo(i,j,0) #point es igual a polo en el indice i,j,0 
                    distances = [ self.__distance(point, magnet )[0] for magnet in list_magnets ] #
                    min_idx = distances.index( min( distances ) )

                    pose = self.__pose_vector_field( i, j, list_magnets[min_idx] ) 
                    if pose[4]:
                        scale_color = 2.5 #if list_magnets[min_idx].fictice else 1.95 # 1.95
                        r = max(pose[3], pose[2])
                        g = max(255 - r*scale_color, 0)
                        # g = 255
                        if list_magnets[min_idx].fictice:
                            g = max(255 - r*1.5, 0)
                            if list_magnets[min_idx].pole[0].polarity == list_magnets[min_idx].pole[1].polarity:
                                # repeler
                                # g = 255
                                self.canvas.line( (i+pose[0] + random.randint(-2, 2), j-pose[1],
                                                    i-pose[0]*2, j+pose[1] + random.randint(-2, 2)),
                                                    fill=(int(g),0,0), width=2 )
                            else:
                                # atraer
                                self.canvas.line( (i+pose[1] + random.randint(-2, 2), j-pose[0],
                                                    i-pose[1],j+pose[0] + random.randint(-2, 2)),
                                                    fill=(0,0,int(g)), width=2)
                        else: # real magnets
                            self.canvas.line( (i+pose[1] + random.randint(-2, 2),
                                                j-pose[0], i-pose[1],j+pose[0] + random.randint(-2, 2)),
                                                fill=(0,int(g),0), width=1 )

        for magnet in self.bag_magnets:
            self.__plot_poles(magnet)

    def add_magnet(self, magnet ):
        self.bag_magnets.append( magnet )

    def plot_magnetic_field(self): #6.- ploteo se divide en 2 funciones mas

        self.__populate_bag_fictice_magnets() #7.-manda a traer la funcion   populate_bag 178
        self.__populate_field() #26.- despues de hacer la funcion anterior entra aqui y nos lleva a la linea 191

        return self.image

def in_same_block( pole1, pole2, mask ):
    num_divisions = 8
    count = 0
    
    offset = math.sqrt((pole2.x-pole1.x)**2 + (pole2.y-pole1.y)**2) // 8 
    x = pole2.x-pole1.x
    y = pole2.y-pole1.y

    n = math.sqrt(x*x + y*y)
    x = x/n
    y = y/n

    # image = cv2.circle(image, (pole1.x, pole1.y), 20, (255, 0, 90 ), 2)
    # image = cv2.circle(image, (pole2.x, pole2.y), 20, (255, 0, 90), 2)

    for d in range(1,8):
        nx = offset*d
        ny = offset*d
        if mask[int(pole1.y + y*ny)][int(pole1.x+ x*nx)] :
            # image = cv2.circle(image, (int(pole1.x+ x*nx), int(pole1.y + y*ny)), 20, (255, 90, 90), 2)
            count += 1
        # else: 
            # image = cv2.circle(image, (int(pole1.x+ x*nx), int(pole1.y + y*ny)), 20, (0, 0, 0), 2)

    return count > 6

def TUI_real_time():
    # define a video capture object
    vid = cv2.VideoCapture(0)
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 1000
    detector = cv2.SimpleBlobDetector_create(params) #---

    while(True):
        ret, frame = vid.read()
        #filro gausiano para el fondo negro y blanco------------------------------
        img = cv2.GaussianBlur(frame, (15, 15), 0)
        b,g,r = cv2.split(img)
        ret, mask = cv2.threshold(r,0,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
        cv2.imshow("mask", mask)

        keypoints = detector.detect(img)

        w, h = 1700, 800
        
        image = Image.new('RGB', (w,h), (0,0,0))

        field = Magnetic_Field( image )

        listPoles = []
        for point in keypoints:
            listPoles.append( Polo(int(point.pt[0]), int(point.pt[1]), 0) )

        # listPoles.sort(key=sortPolo)
        for idx in range(0,len(listPoles)-1):
            for idx_next in range(idx+1,len(listPoles)):
                if in_same_block( listPoles[idx], listPoles[idx_next], mask ) :
                    magnet = Magnet(listPoles[idx], listPoles[idx_next], 
                            strength=50, radio=200, polarity=0)
                    field.add_magnet(magnet)


        pil_image = field.plot_magnetic_field() #5.- manda a traer el ploteo magnetico field ->233 Magnetic_Field -->69
        open_cv_image = np.array(pil_image) 
        # Convert RGB to BGR 
        open_cv_image = open_cv_image[:, :, ::-1].copy() 
        #Muestra la mascara
        cv2.imshow("simulation", open_cv_image)

        blank = np.zeros((1, 1))
        blobs = cv2.drawKeypoints(img, keypoints, blank, (0, 0, 255),cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
        cv2.imshow("raw poles", blobs)

        # the 'q' button is set as the
        # quitting button you may use any
        # desired button of your choice
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()

def main():
    TUI_real_time()
    
    #Create an empty canvas
    # w, h = 1700, 800
    # image = Image.new('RGB', (w,h), (0,0,0))

    # field = Magnetic_Field( image )

    # magnet = Magnet(Polo(630,120, 0), Polo(760,150, 1), strength=50, radio=200, polarity=0)
    # field.add_magnet(magnet)
    # magnet = Magnet(Polo(890,120, 0), Polo(970,250, 1), strength=50, radio=200, polarity=1)
    # field.add_magnet(magnet)    
    # magnet = Magnet(Polo(700,300, 0), Polo(860,400, 1), strength=50, radio=200, polarity=0)
    # field.add_magnet(magnet)
    


    # pil_image = field.plot_magnetic_field()
    
    # image.show() 

if __name__ == "__main__":
    main()

# magnet_2 = Magnet(900,400,80,50, 200)
# magnet_3 = Magnet(0,0,300,50, 200)
# magnet_3.x1 = magnet.x2
# magnet_3.y1 = magnet.y2
# magnet_3.x2 = magnet_2.x1
# magnet_3.y2 = magnet_2.y1

# magnets = [  magnet, magnet_2, magnet_3 ]

#Create a pixel map in the window
# for i in range(20,1800,15):
#     for j in range(20,h,15):

#         dist = [ magnet.distance(i,j), magnet_2.distance(i,j) ] #, magnet_3.distance(i,j)  ]
#         min_idx = dist.index( min( dist ) )

#         pose = magnets[min_idx].pose_vector_field( i, j )
#         if pose[4]:
#             g = 255 - pose[3]*1.5 if 255 - pose[3]*1.5 > 0 else 0
#             #g = 255
#             canvas.line( (i+pose[1] + random.randint(-2, 2), j-pose[0], i-pose[1],j+pose[0] + random.randint(-2, 2)), fill=(0,int(g),0)  )

        # pose = pose_vector_field( i, j, magnet )
        # pose2 = pose_vector_field( i, j, magnet_2 )

        # if pose[4] and pose2[4]:
        #     g = 255 - pose[3]*2 if 255 - pose[3]*2 > 0 else 0
        #     g = 255
        #     if pose[3] < pose2[3]:
        #         canvas.line( (i+pose[1] + random.randint(-2, 2), j-pose[0], i-pose[1],j+pose[0]*2 + random.randint(-2, 2)), fill=(0,g,0)  )
        #     else :
        #         canvas.line( (i+pose2[1] + random.randint(-2, 2), j-pose2[0], i-pose2[1],j+pose2[0]*2 + random.randint(-2, 2)), fill=(0,g,0)  )
        # elif pose[4]:
        #     g = 255 - pose[3]*2 if 255 - pose[3]*2 > 0 else 0
        #     g = 255
        #     canvas.line( (i+pose[1] + random.randint(-2, 2), j-pose[0], i-pose[1],j+pose[0] + random.randint(-2, 2)), fill=(0,g,0)  )
        # elif pose2[4]:
        #     g = 255 - pose2[3]*2 if 255 - pose2[3]*2 > 0 else 0
        #     g = 255
        #     canvas.line( (i+pose2[1] + random.randint(-2, 2), j-pose2[0], i-pose2[1],j+pose2[0] + random.randint(-2, 2)), fill=(0,g,0)  )

# #Create a pixel map in the window
# for i in range(20,w,15):
#     for j in range(20,h,15):

#         pose = pose_vector_field( i, j, magnet )
#         if pose[4]:
#             g = 255 - pose[3] if 255 - pose[3] > 0 else 0
#             # g = 255
#             canvas.line( (i+pose[1] + random.randint(-2, 2), j-pose[0], i-pose[1],j+pose[0] + random.randint(-2, 2)), fill=(0,g,0)  )

# #Create a pixel map in the window
# for i in range(620,w+800,15):
#     for j in range(20,h,15):

#         pose = pose_vector_field( i, j, magnet_2 )
#         if pose[4]:
#             g = 255 - pose[3] if 255 - pose[3] > 0 else 0
#             # g = 255
#             canvas.line( (i+pose[1] + random.randint(-2, 2), j-pose[0], i-pose[1],j+pose[0] + random.randint(-2, 2)), fill=(0,0,g)  )



# magnet.plot_poles(canvas)
# magnet_2.plot_poles(canvas)
# magnet_3.draw(canvas)

