# -*- coding: utf-8 -*-
"""
Created on Sun Apr  9 14:35:23 2017

@author: vineeth
"""
import numpy as np
from PIL import Image
import os
from numpy import linalg 
from matplotlib import pyplot as plt

#initialising means and covariance 
mean_1 = [12.0,12.0,12.0]
mean_2 = [1.2,1.2,1.2]
mean_3 = [18.0,18.0,18.0]

cov_1 =  np.identity(3)
cov_2 =  np.identity(3)
cov_3 =  np.identity(3)

w1=1
w2=1
w3=1


#mean_1 = [11.606,13.872,17.2107]
#mean_2 = [4.916,5.345,6.513]
#mean_3 = [18.302,20.449,21.682]
#
#cov_1 = ([[ 4.88000761,  3.88996182,  1.22786459],
#       [ 3.88996182,  3.37655206,  1.57606076],
#       [ 1.22786459,  1.57606076,  2.63040758]])
#cov_2 = ([[ 18.71682332,  11.59976221,  10.58963548],
#       [ 11.59976221,   8.80288875,   8.9359862 ],
#       [ 10.58963548,   8.9359862 ,  11.08425438]])
#cov_3 = ([[ 10.37888216,   8.03069131,   4.02826505],
#       [  8.03069131,   6.37538741,   3.34129611],
#       [  4.02826505,   3.34129611,   2.28566275]])
#
#w1=0.26367909083161939
#w2=0.037206904461820754
#w3=0.69911400470658014


count =0
m =[]
n = []

def Guassian(x,mu,cov):                                             # calculating Guassinan, x=input,mu = mean,cov = covariance
    det_cov = (linalg.det(cov))**2
    cov = linalg.inv(cov)
    return np.exp(-np.inner(np.dot((x-mu),cov),(x-mu))/2)/(((2*np.pi)**2)*det_cov)


    

def covariance(a,b,c):                                #covariance, a= input, b = mean, c = posterior probablity
    tot = np.zeros(shape=(3,3))
    for i in range(len(a)):
        tot = tot + c[i]*np.outer((a[i]-b),(a[i]-b))
    return tot/sum(c)
    
    
    


im = Image.open('ski_image.jpg','r')                        #reading image file
image_out = Image.new(im.mode,im.size)                      #creating object for output file
pix = np.array(im.getdata())                                #extracting pixel values of image
pix_out = pix
pix = pix/10                                                #scaling data by a factor 10
p =  np.zeros(shape = (len(pix),3))
s=0 
while (1==1):
    s_prev = s
    s=0
    for i in range(len(pix)):
        tot = w1*Guassian(pix[i],mean_1,cov_1)+w2*Guassian(pix[i],mean_2,cov_2)+w3*Guassian(pix[i],mean_3,cov_3)
        s = s + np.log(tot)                                         #loglikelyhood value
        #calculating the responsibilities of each data point
        p[i] = [w1*Guassian(pix[i],mean_1,cov_1)/tot,w2*Guassian(pix[i],mean_2,cov_2)/tot,w3*Guassian(pix[i],mean_3,cov_3)/tot]
    
    temp = np.transpose(p)
    
    #updating the weights and means and covariance
    w1 = sum(temp[0])/len(temp[0])
    w2 = sum(temp[1])/len(temp[1])
    w3 = sum(temp[2])/len(temp[2])    
    
    mean1_prev = mean_1
    mean2_prev = mean_2
    mean2_prev = mean_3
    
    mean_1 = np.dot(temp[0],pix)/sum(temp[0])
    mean_2 = np.dot(temp[1],pix)/sum(temp[1])
    mean_3 = np.dot(temp[2],pix)/sum(temp[2])
    
    cov_1 = covariance(pix,mean_1,temp[0])
    cov_2 = covariance(pix,mean_2,temp[1])
    cov_3 = covariance(pix,mean_3,temp[2])
    
    count = count + 1
    m.append(s)
    n.append(count)
    #checkeing the change in loglikelyhood value and terminating when it less than set value    
    if(abs(s-s_prev) <1):                            
        break
#classifying the data into different clusters, and assigning a value equal to the mean of the respective cluster
for i in range (len(pix)):                 
    a = Guassian(pix[i],mean_1,cov_1)
    b = Guassian(pix[i],mean_2,cov_2)
    c = Guassian(pix[i],mean_3,cov_3)
    if max(a,b,c) == a:
        pix_out[i] = 10*mean_1
    elif max(a,b,c) == b:
        pix_out[i] = 10*mean_2
    elif max(a,b,c) == c:
        pix_out[i] = 10*mean_3

np.savetxt('loglikelyhood vs iterations.txt', np.c_[m,n], fmt='%0.8g', delimiter='    ', newline=os.linesep)
plt.plot(n,m)
plt.show()
image_out.putdata(tuple(map(tuple, pix_out)))
image_out.save('test_out.png')                          #saving the output imagefile
