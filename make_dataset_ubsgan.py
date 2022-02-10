import cv2
import numpy as np
from pip import main
import fnc as f
import os
import random
import matplotlib.pyplot as plt
import math

class Frame_YUV:

    def __init__(self,L,l,bit_per_pixel,tab_frame_yuv,tab_frame_rgb,n):
        self.L=L
        self.l=l
        self.bit_per_pixel=bit_per_pixel
        self.n=n
        self.tab_frame_yuv=tab_frame_yuv
        self.tab_frame_rgb=tab_frame_rgb
        

    def get_tab_frame_yuv(self):
        return(self.tab_frame_yuv)

    def get_tab_frame_rgb(self):
        return(self.tab_frame_rgb)

class Video_YUV_slice:
    def __init__(self,path_video,L,l,bit_per_pixel,nb_frame):
        self.path_video=path_video
        self.bit_per_pixel=bit_per_pixel
        self.L=L #height ou size_ver
        self.l=l #width ou size_hor
        # self.offset=offset
        self.nb_frame=nb_frame
        size_of_one_frame=L*l+2*np.round(L/2)*np.round(l/2)

        try :
            f=open(self.path_video,"rb")
            f.seek(0, os.SEEK_END)
            self.size_total = f.tell()
            self.nb_frame_total=self.size_total/(L*l+2*np.round(L/2)*np.round(l/2))

            self.offset=random.randint(0,self.nb_frame_total-self.nb_frame)
            f.seek(int(self.offset*size_of_one_frame),0)
            self.data=f.read(int(self.nb_frame*size_of_one_frame))
            

        except FileNotFoundError:
            print('cannot open', self.path_video)
            return(None)


        self.size_bytes=len(self.data)
        self.data_tab=np.array([self.data[i] for i in range(0,self.size_bytes)])

        
    def get_name(self):
        return(self.path_video)

    def get_nb_frame(self):
        return(self.nb_frame)

    def get_size_bytes(self):
        return(self.size_bytes)

    def get_frame(self,n):
        if n>self.nb_frame:
            print('Error : Total number of frame is ',int(self.nb_frame))
            print('Retry with n < ',int(self.nb_frame))
            return()

        size_luma=int(self.L*self.l)
        size_chroma=int(np.round(self.L/2)*np.round(self.l/2))
        size_frame=int(size_luma+2*size_chroma)

        # Get luma of frame number n
        luma=self.data_tab[size_frame*n:size_frame*n+size_luma].reshape(self.L,self.l)

        # Get chroma
        p_cr=int(size_frame*n+size_luma)
        cr=self.data_tab[p_cr:p_cr+size_chroma].reshape(int(np.round(self.L/2)),int(np.round(self.l/2)))
        cr_interpol= cv2.resize(cr,None, fx = 2, fy = 2, interpolation = cv2.INTER_NEAREST)

        # Get cb
        p_cb=int(size_frame*n+size_luma+size_chroma)
        cb=self.data_tab[p_cb:p_cb+size_chroma].reshape(int(np.round(self.L/2)),int(np.round(self.l/2)))
        cb_interpol= cv2.resize(cb,None, fx = 2, fy = 2, interpolation = cv2.INTER_NEAREST)

        #Reconstruction
        tab_frame_yuv=np.zeros((self.L,self.l,3))
        tab_frame_yuv[:,:,0]=luma
        tab_frame_yuv[:,:,1]=cr_interpol
        tab_frame_yuv[:,:,2]=cb_interpol
        tab_frame_rgb = cv2.cvtColor(tab_frame_yuv.astype('uint8'), cv2.COLOR_YCrCb2BGR)

        frame=Frame_YUV(self.L,self.l,self.bit_per_pixel,tab_frame_yuv,tab_frame_rgb,n)
        return(frame)

def norm(tab):
    return(255*(tab-np.min(tab))/(np.max(tab)-np.min(tab)))

def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr
########


if __name__ == "__main__":
    root='yuv'
    # name='RaceHorses_832x480_30.yuv'
    # l=480
    # L=832
    name='football_cif.yuv'
    l=288
    L=352
    
    vid=Video_YUV_slice(root+'/'+name,l,L,8,5)
    im_0=vid.get_frame(0).get_tab_frame_rgb()      
    im_1=vid.get_frame(1).get_tab_frame_rgb()
    im_2=vid.get_frame(2).get_tab_frame_rgb()
    im_3=vid.get_frame(3).get_tab_frame_rgb()
    im_4=vid.get_frame(4).get_tab_frame_rgb()

    M=np.array([[0.64,0,0.02,0.32],[0,0.48,0.84,0]])
    B=np.array([[1.22,0],[0,0.5],[0.16,0.88],[0.61,0]])
    S=np.zeros((4,l*L*3))
    S[0]=np.ravel(im_1)
    S[1]=np.ravel(im_2)
    S[2]=np.ravel(im_3)
    S[3]=np.ravel(im_4)
    A=np.round((1/1.32)*np.dot(M,S))
    print(np.min(A))
    print(np.max(A))

### envoie a l'encodeur décodeur

    output=1.32*np.dot(B,A)
    output=np.round(norm(output)).astype('int')
   
    fig=plt.figure()
    fig.add_subplot(2,4,1).set_title('Im1')
    plt.imshow(im_1)
    fig.add_subplot(2,4,2).set_title('Im2')
    plt.imshow(im_2)
    fig.add_subplot(2,4,3).set_title('Im3')
    plt.imshow(im_3)
    fig.add_subplot(2,4,4).set_title('Im3')
    plt.imshow(im_4)

    fig.add_subplot(2,4,5).set_title('Im1d')
    plt.imshow(output[0,:].reshape((l,L,3)))
    fig.add_subplot(2,4,6).set_title('Im2d')
    plt.imshow(output[1,:].reshape((l,L,3)))
    fig.add_subplot(2,4,7).set_title('Im3d')
    plt.imshow(output[2,:].reshape((l,L,3)))
    fig.add_subplot(2,4,8).set_title('Im3d')
    plt.imshow(output[3,:].reshape((l,L,3)))
    
    im_2_pred=output[1,:].reshape((l,L,3))
    psnr=(4*PSNR(im_2[:,:,0],im_2_pred[:,:,0])+2*PSNR(im_2[:,:,1],im_2_pred[:,:,1])+2*PSNR(im_2[:,:,2],im_2_pred[:,:,2]))/6
    print(psnr)
    plt.show()


