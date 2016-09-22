import scipy.io as sio
import numpy as np
import cv2
from matplotlib import pyplot as plt

def jpegCompress(image, quantmatrix):
    '''
        Compress(imagefile, quanmatrix simulates the lossy compression of 
        baseline JPEG, by quantizing the DCT coefficients in 8x8 blocks
    '''
    # Return compressed image in result
    
    H = np.size(image, 0)
    W = np.size(image, 1)

    # Number of 8x8 blocks in the height and width directions
    h8 = H / 8 + 1
    w8 = W / 8 + 1
    
    # TODO If not an integer number of blocks, pad it with zeros
    pad_h = h8*8-H
    pad_w = w8*8-W
    if (pad_h == 8):
        pad_h = 0
        h8 = h8-1
    if (pad_w == 8):
        pad_w = 0
        w8 = w8-1
        
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)/255.0
    I = np.pad(image, ((0, pad_h), (0, pad_w)), 'constant', constant_values=(0,0) )

    # TODO Separate the image into blocks, and compress the blocks via quantization DCT coefficients
    jpeg = []
    for h in range(0,h8):
        for w in range(0,w8):
            block = I[8*h:8*(h+1),8*w:8*(w+1)]
            dct = cv2.dct(block)
            B = np.rint(dct*255/quantmatrix)
            jpeg.append(B)

    # TODO Convert back from DCT domain to RGB image
    Ijpeg = np.zeros_like(I)
    for h in range(0,h8):
        for w in range(0,w8):
            iB = jpeg[h*w8 + w]
            idct = iB*quantmatrix
            iblock = cv2.idct(idct)
            Ijpeg[8*h:8*(h+1),8*w:8*(w+1)] = iblock
    if (pad_h != 0):
        Ijpeg = Ijpeg[0 : -pad_h, :]      
    if (pad_w != 0):
        Ijpeg = Ijpeg[:, 0 : -pad_w]   
    result = Ijpeg
    return result

if __name__ == '__main__':

    im = cv2.imread('./misc/lena_gray.bmp')
    im.astype('float')

    quantmatrix = sio.loadmat('./misc/quantmatrix.mat')['quantmatrix']

    out = jpegCompress(im, quantmatrix)

    # Show result, OpenCV is BGR-channel, matplotlib is RGB-channel
    # Or: 
    #   cv2.imshow('output',out)
    #   cv2.waitKey(0)

 

    plt.imshow(out,cmap = plt.cm.Greys_r)
    #plt.imshow(out)
    plt.show()

