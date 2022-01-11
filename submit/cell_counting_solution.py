import skimage.io as io
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import sobel
    
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
    filled_image=edge_image.copy()
    n_regions_found_so_far=0
    # start by filling the background to dark gray, from pixel (0,0)
    s=seedfill(filled_image, 0 ,0, 0.1,0)
    for i in range(filled_image.shape[0]):
        for j in range(filled_image.shape[1]):
            # if pixel is black, seedfill from here
            if filled_image[i,j]==0:
                col = 0.5+0.001*n_regions_found_so_far
                seedfill(filled_image, i ,j, col,0)
                n_regions_found_so_far+=1
                print(n_regions_found_so_far)
    return filled_image


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
    n_row, n_col = original_image.shape
    # Build a set of all grayscale values in the labeled image
    grayscales = {labeled_image[i,j] for i in range(n_row) for j in range(n_col)}
    infected = set()  # will keep track of grayscale values associated to infected cells
    not_infected = set() # will keep track of grayscale values associated to infected cells
    for gray in grayscales:
        if gray>=0.5 and gray<1:  # skip grayscale values corresponding to background or to edges
            # count the number of dark and light pixels
            # belonging to that label
            n_dark=0
            n_clear=0
            for i in range(n_row):
                for j in range(n_col):
                    if labeled_image[i,j]==gray:
                        if original_image[i,j]<=infected_grayscale:
                            n_dark+=1
                        else: 
                            n_clear+=1
            #determine is region is cell, infected or not
            region_size = n_dark+n_clear
            if region_size>=min_size and region_size<=max_size:
                if n_dark/region_size>=min_infected_percentage:
                    infected.add(gray)
                else:
                    not_infected.add(gray)
    return (infected, not_infected)


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
    n_row, n_col = labeled_image.shape
    annotated_image=color_image.copy()
    for i in range(n_row):
        for j in range(n_col):
            max_neighbor = np.max(labeled_image[max(0,i-1):min(n_row,i+2),max(0,j-1):min(n_col,j+2)])
            if max_neighbor==1.0:
                if labeled_image[i,j] in infected:
                    annotated_image[i,j]=(255,0,0)
                if labeled_image[i,j] in not_infected:
                    annotated_image[i,j]=(0,255,0)
    return annotated_image

if __name__ == "__main__":  # do not remove this line   
    
    # QUESTION 1: WRITE YOUR CODE HERE
    image = io.imread("malaria-1.jpg")
    print(image.shape)
    image_graytone = rgb2gray(image)
    image_sobel = sobel(image_graytone)
    
    io.imsave("Q1_Sobel.jpg",image_sobel)
    
    # QUESTION 2: WRITE YOUR CODE HERE
    image_sobel_T005=np.where(image_sobel>=0.05,1.0, 0.0)
    io.imsave("Q2_Sobel_T0.05.jpg",image_sobel_T005)
    
    # QUESTION 3: WRITE YOUR CODE HERE
    n_row, n_col = image_sobel_T005.shape
    sobel_clean = image_sobel_T005.copy()
    for i in range(n_row):
        for j in range(n_col):
            if np.min(image_graytone[max(0,i-1):min(n_row,i+2),max(0,j-1):min(n_col,j+2)])<0.5:
                sobel_clean[i,j]=0
    io.imsave("Q3_Sobel_T0.05_clean.jpg",sobel_clean)
    
    # QUESTION 4: WRITE YOUR CODE CALLING THE FILL_CELLS FUNCTION HERE
    image_filled=fill_cells(sobel_clean)
    io.imsave("Q4_Sobel_T_0.05_clean_filled.jpg",image_filled)
    
    # QUESTION 5: WRITE YOUR CODE CALLING THE CLASSIFY_CELLS FUNCTION HERE
    infected, not_infected = classify_cells(image_graytone, image_filled)
    print(infected)
    print(not_infected)
    
    # QUESTION 6: WRITE YOUR CODE CALLING THE ANNOTATE_IMAGE FUNCTION HERE
    annotated_image = annotate_image(image, image_filled,infected, not_infected)
    io.imsave("Q6_annotated.jpg",annotated_image)
    

