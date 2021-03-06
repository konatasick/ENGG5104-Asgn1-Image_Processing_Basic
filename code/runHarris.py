import scipy.io as sio
import numpy as np
import cv2
from matplotlib import pyplot as plt

def harrisdetector(image, k, t):
    # TODO Write harrisdetector function based on the illustration in specification.
    # Return corner points x-coordinates in result[0] and y-coordinates in result[1]
    I = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    I = I/255.0
    w = np.ones([2*k+1,2*k+1]) 
     
    Ix = cv2.filter2D(I,-1,np.array([[0, 0, 0],[0, -1, 1],[0, 0, 0]]))  
    Iy = cv2.filter2D(I,-1,np.array([[0, 0, 0],[0, -1, 0],[0, 1, 0]]))  

    Axx = cv2.filter2D(Ix**2,-1,w)  
    Axy = cv2.filter2D(Ix*Iy,-1,w)  
    Ayy = cv2.filter2D(Iy**2,-1,w)  
    
    H = np.size(I, 0)
    W = np.size(I, 1)
    
    ptr_x = []
    ptr_y = []
    for i in range(0, H):
        for j in range(0, W):                        
           A,B = np.linalg.eig(np.array([[Axx[i,j], Axy[i,j]], [Axy[i,j], Ayy[i,j]]]))
           #print A
           if A[0]>t and A[1]>t:
               ptr_y.append(i)
               ptr_x.append(j)    

    result = [ptr_x, ptr_y]
    return result

if __name__ == '__main__':
    k = 3       # change to your value
    t = 0.45   # change to your value

    I = cv2.imread('./misc/corner_gray.png')

    fr = harrisdetector(I, k, t)

    # Show input, OpenCV is BGR-channel, matplotlib is RGB-channel
    # Or: 
    #   cv2.imshow('output',out)
    #   cv2.waitKey(0)
    plt.imshow(I)
    
    # plot harris points overlaid on input image
    plt.scatter(x=fr[0], y=fr[1], c='r', s=40) 

    # show
    plt.show()							
