#Steven Vuong - COMP 204
#260928068

import skimage.io as io
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel
import matplotlib.pyplot as plt
    
# This function is provided to you. You will need to call it.
# You should not need to modify it.
def seedfill(im, seed_row, seed_col, fill_color,bckg):
    """
    im: The image on which to perform the seedfill algorithm
    seed_row and seed_col: position of the seed pixel
    fill_color: Color for the fill
    bckg: Color of the background, to be filled
    Returns: Number of pixels filled
    Behavior: Modifies image by performing seedfill
    """
    size=0  # keep track of patch size
    n_row, n_col = im.shape
    front={(seed_row,seed_col)}  # initial front
    while len(front)>0:
        r, c = front.pop()  # remove an element from front
        if im[r, c]==bckg: 
            im[r, c]=fill_color  # color the pixel
            size+=1
            # look at all neighbors
            for i in range(max(0,r-1), min(n_row,r+2)):
                for j in range(max(0,c-1),min(n_col,c+2)):
                    # if background, add to front
                    if im[i,j]==bckg and\
                       (i,j) not in front:
                        front.add((i,j))
    return size


# QUESTION 4
def fill_cells(edge_image):
    """
    Args:
        edge_image: A black-and-white image, with black background and
                    white edges
    Returns: A new image where each close region is filled with a different
             grayscale value
    """
    filled_image = edge_image.copy()      #image that we're changing 
    seedfill(filled_image, 0, 0, 0.1, 0)  #changing the black background to a dark gray using the seedfill algorithm
    
    nb_regions_found_so_far = 0    #factor used to change shade of gray
    #nested for loops : iterating over all the pixels of the image 
    for row in range(filled_image.shape[0]):
        for col in range(filled_image.shape[1]):
            if np.array_equal(filled_image[row,col],0): #check if the pixel is black or not
                nb_regions_found_so_far +=1             #increment the number of regions found so far by 1 because we found 1
                #use the seedfill algorith to fill the inside of the cell by a shade of gray 
                seedfill(filled_image, row, col, 0.5+0.001*nb_regions_found_so_far , 0)  
    return filled_image #return the image with filled cells 


# QUESTION 5
def classify_cells(original_image, labeled_image, \
                   min_size=1000, max_size=5000, \
                   infected_grayscale=0.5, min_infected_percentage=0.02):
    """
    Args:
        original_image: A graytone image
        labeled_image: A graytone image, with each closed region colored
                       with a different grayscal value
        min_size, max_size: The min and max size of a region to be called a cell
        infected_grayscale: Maximum grayscale value for a pixel to be called infected
        min_infected_percentage: Smallest fraction of dark pixels needed to call a cell infected
    Returns: A tuple of two sets, containing the grayscale values of cells 
             that are infected and not infected
    """
    set_of_all_grayscale_values = set([])                            #build a set of all grayscale values 
    #nested for loops : iterate over each pixel of the labeled image
    for r in range(labeled_image.shape[0]):                          
        for c in range(labeled_image.shape[1]):                     
            if 0.5 <= labeled_image[r,c] < 1.0:                      #checking if the pixel is part of a cell, 
                set_of_all_grayscale_values.add(labeled_image[r,c])  #adding the value, it's a set so no duplicate
    infected = set([])        #empty set to add grayscale values of infected cells
    not_infected = set([])    #empty set to add grayscale values of non-fected cells 
    #iterate over each grayscale values, aka each region colored in a different shade of gray 
    for gray_value in set_of_all_grayscale_values:    
        nb_dark_pixels = 0     #to increment if there's a dark pixel
        nb_light_pixels = 0    #to increment if there's a light pixel
        #nested for loops : iterate over each pixel of the labeled image
        for r in range(labeled_image.shape[0]):                      
            for c in range(labeled_image.shape[1]):                 
                # checking if the pixel value of the current pixel is the same as the gray_value that we're currently iterating on  
                if labeled_image[r,c] == gray_value:
                    if original_image[r,c] <= infected_grayscale:    #checking if the pixel grayscale value in the original image is below 0.5 
                        nb_dark_pixels +=1                           #if it is increment by 1 the dark pixel counter
                    elif original_image[r,c] >= infected_grayscale:  #checking if the pixel grayscale value in the original image is above 0.5 
                        nb_light_pixels +=1                          #if it is increment by 1 the light pixel counter
        total_pixels = nb_dark_pixels + nb_light_pixels              #count the total number of pixel with that grayscale value 
        if min_size < total_pixels < max_size:                       #check if the pixels with the gray_value are part of a cell
            threshold = total_pixels*min_infected_percentage         #calculate the 2% of pixels
            if nb_dark_pixels > threshold :                          #check if there are at least 2% of the cells that are dark in the original grayscale
                infected.add(gray_value)                             #add the grayscale value to the infected set 
            else :                                                   
                not_infected.add(gray_value)                         #add the grayscale value to hte not_infected                                     
    return (infected, not_infected)                                  #return the tuple of 2 sets


# QUESTION 6
def annotate_image(color_image, labeled_image, infected, not_infected):
    """
    Args:
        color_image: A color image
        labeled_image: A graytone image, with each closed region colored
                       with a different grayscal value
        infected: A set of graytone values of infected cells
        not_infected: A set of graytone values of non-infcted cells
    Returns: A color image, with infected cells highlighted in red
             and non-infected cells highlighted in green
    """    
    highlighted_image = color_image.copy()                         #making a copy of the original image
   #nested for loops : iterating over the whole image
    for row in range(highlighted_image.shape[0]):
        for col in range(highlighted_image.shape[1]):
            if labeled_image[row,col] in infected:                 #check if grayscale value is part of infected grayscale value set
                #nested for loops :looking at the 8 surrounding pixels
                for r in range(row-1,row+2):          
                    for c in range(col-1, col+2):  
                        if labeled_image[r,c] == 1.0 :             #check if pixel is white
                           highlighted_image[row,col] = (255,0,0)  #change that pixel from white to red

            if labeled_image[row,col] in not_infected :            #check if grayscale value is part of not_fected grayscale value set
                #nested for loops : looking at the 8 surrounding pixels
                 for r in range(row-1,row+2):   
                    for c in range(col-1, col+2):  
                        if labeled_image[r,c] == 1.0 :             #check if pixel is white
                           highlighted_image[row,col] = (0,255,0)  #change that pixel from white to red
   
    return highlighted_image                                       #return the properly highlighted image

if __name__ == "__main__":  # do not remove this line   
    
    # QUESTION 1: WRITE YOUR CODE HERE 
    og_image = io.imread("malaria-1.jpg")           #accessing the image from folder
    gray_image = rgb2gray(og_image)                 #converting the image into a grayscale image
    edge = sobel(gray_image)                        #using sobel algorithm go get the edges of the image
    io.imsave("Q1_Sobel.jpg",edge)                  #saving the image
       
    # QUESTION 2: WRITE YOUR CODE HERE
    black_white = np.where(edge>0.05, 1.0, 0.0)     #if edginess is > than 0.05, change to white, if <0.05 change to black 
    io.imsave("Q2_Sobel_T_0.05.jpg",black_white)    
    
    
    # QUESTION 3: WRITE YOUR CODE HERE
    n_row, n_col, foo = og_image.shape   #get nb of rows, columns and dimension
    #nested for loops : iterate over all the pixels in the image except the edges 
    clean_image = black_white.copy()                #create a copy of the image to modify
    for row in range(1,n_row-1):       
        for col in range(1,n_col-1):  
            if np.array(gray_image[row,col]<0.5):   #check if the original graytone pixel value is below 0.5
                #nested for loop: looking at the 8 surrounding pixels
                for r in range(row-1,row+2):      
                    for c in range(col-1, col+2):                      
                        clean_image[r,c] = 0        #change pixel to a black  
 
    io.imsave("Q3_Sobel_T0.05_clean.jpg",clean_image) 
    
    # QUESTION 4: WRITE YOUR CODE CALLING THE FILL_CELLS FUNCTION HERE
    labeled_image = fill_cells(clean_image)    
    io.imsave("Q4_Sobel_T0.05_clean_filled.jpg",labeled_image)
    
    
    # QUESTION 5: WRITE YOUR CODE CALLING THE CLASSIFY_CELLS FUNCTION HERE
    sets_of_grayscale_values = classify_cells(gray_image, labeled_image, \
                   min_size=1000, max_size=5000, \
                   infected_grayscale=0.5, min_infected_percentage=0.02)
    
    # QUESTION 6: WRITE YOUR CODE CALLING THE ANNOTATE_IMAGE FUNCTION HERE
 
    annotated_image = annotate_image(og_image, labeled_image, \
                                     sets_of_grayscale_values[0], \
                                     sets_of_grayscale_values[1])
    io.imsave("Q6_annotated.jpg",annotated_image)
    
    