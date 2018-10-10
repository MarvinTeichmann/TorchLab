import sys
import numpy as np
#import cv2
#sys.path.append('/home/ibu6429/shared_file_system/code/math_functions/')
from algebra import Algebra

def extractEquirectangular(wrap,source_image,result_image,euler_angles):
	extractEquirectangular_quick(wrap,source_image,result_image,Algebra.rotation_matrix(euler_angles))

def extractEquirectangular_quick(wrap, source_image, result_image,R):

        # First dimension: Yaw
        # First dimension: Roll
        # Third dimenstion: Pitch


        Algebra.test_polar_to_polar()
        min_theta = np.pi * 4
        max_theta = -np.pi * 4;
        min_psi = np.pi * 4;
        max_psi = -np.pi * 4;

        result_image[:,:,0]=255
        width=result_image.shape[1]
        height=result_image.shape[0]

        row,col = np.mgrid[0:height,0:width]
        polar_point=np.zeros((height,width,3))
        polar_point [:,:,0]=1.0
        polar_point[:,:,1] = (row / (1.0 * height))*np.pi
        polar_point[:,:,2]= ((col - width // 2) / (0.5 * width)) *np.pi

        max_1=np.max(np.max(polar_point[:,:,1]))
        min_1=np.min(np.min(polar_point[:,:,1]))
        abs_max_2=np.max(np.max(np.abs(polar_point[:,:,2])))

        #print('max 1 min 1 absmax2 ' +str(max_1)+ ' '+str(min_1)+' ' +str(abs_max_2))
        #assert(max_1 <= np.pi) and(min_1>=0)and(abs_max_2<=np.pi) #disabled for speed

        plane_point = Algebra.polar_to_cartesian_array(polar_point)

        # assert( np.max(np.max(np.abs(Algebra.magnitude_array(plane_point)-polar_point[:,:,0]))) < 0.0001) #disabled for speed

        plane_rot=Algebra.rotate_array(R,plane_point)

        #assert(np.max(np.max(np.abs(Algebra.magnitude_array(plane_point) - Algebra.magnitude_array(plane_rot)))) < 0.0001) #disbled for speed

        eq_row,eq_col=Algebra.cartesian_to_polar_quantised_array(wrap,plane_rot,width,height)


        #assert ((np.min(np.min(eq_row))>=0) and (np.max(np.max(eq_row))<height)and (np.min(np.min(eq_col))>=0)  and (np.max(np.max(eq_row))<width)) #disabled for speed

        result_image[row, col, :] = source_image[ eq_row ,eq_col, :]

