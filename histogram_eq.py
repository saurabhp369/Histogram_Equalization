import numpy as np
import cv2
import matplotlib.pyplot as plt
import os
import glob

def plot_hist(intensity_count):
    intensity = np.arange(0,256)
    plt.stem(intensity, intensity_count , use_line_collection = 'True')
    plt.xlabel('intensity value')
    plt.ylabel('number of pixels')
    # plt.title('Histogram of the original image')
    plt.show()

def hist(image):
    bins = 256
    h = np.zeros(bins)
    for i in image:
        h[i] +=1
    return h

def cdf(count_matrix): 
    sum = count_matrix[0]
    cum_sum = [sum] 
    for i in range(0,len(count_matrix)-1):
        sum+=count_matrix[i+1]
        cum_sum.append(sum)
    return np.array(cum_sum)

def normalize(cdf):
    c = np.array(cdf)
    return ((c - c.min()) * 255) / (c.max() - c.min())

def hist_equalization(channel):
    y_histogram = hist(channel)
    # plot_hist(y_histogram)
    cum_dist = cdf(y_histogram)
    y_norm = np.int32(normalize(cum_dist))
    y_hist_equal = y_norm[channel]
    y_equal_hist = hist(y_hist_equal)
    # plot_hist(y_equal_hist)
    return y_hist_equal

def main():
    image_files = sorted(glob.glob('*.png'))
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    choice = int(input('Enter the choice 1: Histogram Equalization 2:AHE '))
    if choice ==1:
        print('Performing Histogram Equalization')
        video = cv2.VideoWriter('HE.mp4',fourcc, 5, (1224, 370))
        for name in image_files:
            img = cv2.imread(name)
            # converting RGB image to YCbCr color space for histogram equalization 
            ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
            y_flatten = ycrcb_img[:, :, 0].flatten()
            equalized_channel = hist_equalization(y_flatten)
            ycrcb_img[:,:,0] = equalized_channel.reshape(ycrcb_img[:,:,0].shape)
            final_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCR_CB2BGR)
            video.write(final_img)
    else:
        print('Performing Adaptive Histogram Equalization')
        video = cv2.VideoWriter('AHE.mp4', fourcc, 5, (1224, 320))
        for name in image_files:
            
            img = cv2.imread(name)
            # converting RGB image to YCbCr color space for histogram equalization 
            ycrcb_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)
            ycrcb_img = cv2.resize(ycrcb_img, (1224,320))
            # splitting the image into 8*8 kernels
            sliced = np.split(ycrcb_img[:,:,0],8,axis=0)
            blocks = [np.split(img_slice,8,axis=1) for img_slice in sliced]
            for i in range(len(blocks)):
                for j in range(len(blocks[0])):
                    # print(blocks[i][j])
                    y_block_flatten = blocks[i][j].flatten()
                    equalized_block = hist_equalization(y_block_flatten)
                    blocks[i][j] = equalized_block.reshape((40,153))

            equalized_y = np.block(blocks)
            ycrcb_img[:,:,0]= np.float32(equalized_y)
            final_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
            video.write(final_img)
            # cv2.imshow('frame',final_img)
            # cv2.waitKey(0)

    print('Video saved')
    video.release()


if __name__ == '__main__':
    main()