import numpy as np
import cv2


inimg_slika_sa_kojom_se_poredi = cv2.imread("passat_main.jpg")
height = np.size(inimg_slika_sa_kojom_se_poredi, 0)
width = np.size(inimg_slika_sa_kojom_se_poredi, 1)
blockSize = (16, 16)
blockStride = (8, 8)
cellSize = (8, 8)
nbins = 9

inimg1_gray = cv2.cvtColor(inimg_slika_sa_kojom_se_poredi, cv2.COLOR_RGB2GRAY)

sum_distance = []
sum_all_distance = []
num_of_img = 16
Prefix = "trainingImgs/passat ("

for n in range(1, num_of_img, 1):
    s=str(n)
    inimg1 = cv2.imread( "trainingImgs/passat ("+s+").jpg")
    inimg1_b_gray = cv2.cvtColor(inimg1, cv2.COLOR_RGB2GRAY)

    for i in range(0, width - 64, 1):
        for j in range(0, height - 128, 24):
            roi1 = inimg1_gray[j: j + 128, i: i + 64]
            roi2 = inimg1_b_gray[j: j + 128, i: i + 64]
            cv2.imwrite("roi1.jpg", roi1)

            roi1a = roi1.size
            roi2a = roi2.size

            d = cv2.HOGDescriptor((64, 128), blockSize, blockStride, cellSize, nbins)
            d2 = cv2.HOGDescriptor((64, 128), blockSize, blockStride, cellSize, nbins)

            winStride = (8, 8)
            padding = (8, 8)
            locations = ((10, 20),)
            descriptorValues1 = d.compute(roi1)
            descriptorValues2 = d2.compute(roi2)
            A = descriptorValues1
            B = descriptorValues2

            C = A - B
            heightC = np.size(C, 0)
            widthC = np.size(C, 1)
            for x in range(0, widthC, 1):
                C = C[x] * C[x]

            np.sqrt(C, C)
            SumaM = sum(C)
            sum_distance.append(SumaM)

    average_dist = 0
    percent = (n / num_of_img) * 100.0
    ps = str(percent)
    
    s = "Processing" + ps + " % " 

    print(s) 

    for k in range(0, sum_distance.__len__(), 1):
        average_dist += sum_distance[k]

    average_dist /= sum_distance.__sizeof__()
    sum_all_distance.append(average_dist)
    del sum_distance[:]





min = 10000
rememberL = 0

for l in range(0, num_of_img - 1, 1):
    if(sum_all_distance[l] < min):
        min = sum_all_distance[l]
        rememberL = l + 1

f=str(rememberL)
cv2.imwrite("main.jpg", inimg_slika_sa_kojom_se_poredi)
cv2.imshow("1", inimg_slika_sa_kojom_se_poredi)
ucitaj = cv2.imread(Prefix + f + ").jpg")
cv2.imshow("2", ucitaj)
cv2.imwrite("najslicnija.jpg", ucitaj)

cv2.waitKey(0)
cv2.destroyWindow()

