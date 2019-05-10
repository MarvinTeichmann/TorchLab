try:
    import cv2
except:
    ImportError
import os
import numpy as np
from matplotlib import pyplot as plt

from functools import reduce
#//obtaining image plane rotation matrix
#double[,] rot = PixelOperations.multiply(PixelOperations.multiply(PixelOperations.NewRotateAroundZ(rotation[2]), PixelOperations.NewRotateAroundY(rotation[1])), PixelOperations.NewRotateAroundX(rotation[0]));

def euler_to_mat(z=0,y=0,x=0):
    Ms = []
    if z:
        cosz = np.cos(z)
        sinz = np.sin(z)
        Ms.append(np.array(
                [[cosz, -sinz, 0],
                 [sinz, cosz, 0],
                 [0, 0, 1]]))
    if y:
        cosy = np.cos(y)
        siny = np.sin(y)
        Ms.append(np.array(
                [[cosy, 0, siny],
                 [0, 1, 0],
                 [-siny, 0, cosy]]))
    if x:
        cosx = np.cos(x)
        sinx = np.sin(x)
        Ms.append(np.array(
                [[1, 0, 0],
                 [0, cosx, -sinx],
                 [0, sinx, cosx]]))
    if Ms:
        return reduce(np.dot, Ms[::-1])
    return np.eye(3)


def cartesian_to_polar(x,y, z):
    #/https://en.wikipedia.org/wiki/Spherical_coordinate_system
    # print('x shap e'+str(x.shape))
    r = np.sqrt(x[:,:] * x[:,:] + y[:,:] * y[:,:] + z[:,:] * z[:,:])
    latitude_theta = np.arccos(z / r) # 0 .. pi
    longitude_psi = np.arctan2(y, x) #; //-pi/2..pi/2
    result = np.zeros((x.shape[0],x.shape[1],3))
    result [:,:,0] = r[:,:]
    result [:,:,1]= latitude_theta[:,:]
    result [:,:,2]= longitude_psi[:,:]
    return result

def cartesian_to_polar_scalar(x,y, z):
        #/https://en.wikipedia.org/wiki/Spherical_coordinate_system
        r = np.sqrt(x * x + y * y + z * z)
        latitude_theta = np.arccos(z / r) # 0 .. pi
        longitude_psi = np.arctan2(y, x) #; //-pi/2..pi/2
        return np.array([r,latitude_theta,longitude_psi])

#        public static double[] polar_to_cartesian(double r, double latitude_theta, double longitude_psi)
#            //https://en.wikipedia.org/wiki/Spherical_coordinate_system
#            double x = r * Math.Sin(latitude_theta) * Math.Cos(longitude_psi);
#            double y = r * Math.Sin(latitude_theta) * Math.Sin(longitude_psi);
#            double z = r * Math.Cos(latitude_theta);
#            return new double[] { x, y, z };

def equirectangular_crop_slow(source,params):
    wrap = params['wrap']
    plane_f=params['plane_f']
    HFoV =params['HFoV']
    VFoV = params['VFoV']
    rot = params['R']
    plane_w = (2*plane_f)*np.tan(HFoV/2.0)
    plane_h = (2 * plane_f) * np.tan(VFoV / 2.0)
    H_res = params['H_res']
    W_res = params['W_res']
    result=np.zeros((H_res,W_res,3))

    H_res = result.shape[0]
    W_res = result.shape[1]

    H_res = result.shape[0]
    W_res = result.shape[1]
    H_source=source.shape[0]
    W_source=source.shape[1]
    for row in range(H_res):
        for col in range(W_res):
                #//choosing a point on an image plane
                        plane_point = np.array([ ((row - H_res / 2) / (1.0 * H_res)) * plane_h, ((col - W_res / 2) / (1.0 * W_res)) * plane_w, plane_f ])#,1.0])
                        middle_point = np.array([ 0, 0, plane_f] ) #, 1.0 ])
                        vect1 = (plane_point[0] - middle_point[0]) * middle_point[0] + (plane_point[1] - middle_point[1]) * middle_point[1] + (plane_point[2] - middle_point[2]) * middle_point[2]
                        assert(np.abs(vect1) < 0.0001)
                        if (np.linalg.norm(plane_point) <= 1.0):
                            #//obtaining rotated point on a sphere
                            plane_rot = np.matmul(rot,plane_point)
                            assert (np.abs(np.linalg.norm(plane_point) - np.linalg.norm(plane_rot)) < 0.0001)
                            # //If the plane point is within a sphere then proceed
                            r_scale = 1.0 / np.linalg.norm(plane_rot)
                            #//projecting a point on a sphere
                            sp_rot = plane_rot*r_scale
                            assert (np.abs(np.linalg.norm(sp_rot)-1) < 0.0001)
                            mid_rot = np.matmul (rot, middle_point)
                            assert (np.abs(np.linalg.norm(middle_point) - np.linalg.norm(mid_rot)) < 0.0001)
                            dif_point = plane_point-middle_point
                            dif_point_rot = plane_rot - mid_rot
                            assert (np.abs(np.linalg.norm(dif_point) - np.linalg.norm(dif_point_rot)) < 0.0001)
                            #//Console.WriteLine(plane_rot[0]+ " "+plane_rot[1]+ " " +plane_rot[2]+ " | " +mid_rot[0]+" "+mid_rot[1]+ " "+mid_rot[2]);
                            vect = (plane_rot[0]- mid_rot[0]) * mid_rot[0] + ( plane_rot[1]- mid_rot[1]) * mid_rot[1] + (plane_rot[2]- mid_rot[2]) * mid_rot[2]
                            assert (np.abs(vect) < 0.0001)
                            #//Checking if the point is still on a sphere
                            assert (np.abs(1 - np.linalg.norm(sp_rot)) < 0.0001)                          
                            #//Obtaining polar coordinates
                            polar = cartesian_to_polar_scalar(sp_rot[0], sp_rot[1], sp_rot[2]);
                            latitude_theta = polar[1] #; //0..pi
                            longitude_psi = polar[2] #;//-pi..pi
                            if (np.isnan(longitude_psi) == False):
                                #//obtaining row in equirectangular image corresponding to latitude
                                eq_row = np.int32(np.round(((latitude_theta) / np.pi) * H_source)) #check for np rounding problems
                                # //obtaining row in equirectangular image corresponding to longitude
                            eq_col = np.int32(np.round(((longitude_psi + np.pi) / (2 * np.pi)) * W_source))
                            if(wrap==True):
                                    eq_row = (eq_row + H_source) % H_source
                                    eq_col = (eq_col + W_source) % W_source
                            if ((eq_row >= 0) and (eq_col >= 0) and (eq_row < H_source) and (eq_col < W_source)):
                            #{   //If projected point is within an image then proceed to copy the right pixel value from equirectangular image
                                result[row, col,:] = source[eq_row, eq_col, :]
                            else:
                                    assert False, "Should not be stepping here!"
                                    #//If projected point is not within an image then fill with blue pixel for debugging
                                    #im2_pdataa[im2_pixeldata.Stride * row + col * 3 + 0] = 255;
                        else:
                            assert False, "Should not be stepping here!"
                            #//If the plane point is not within a sphere then fill it with green pixels for debugging
                            result[row,col,2] = 255
    return result
def equirectangular_crop_id_image(source,params):
    wrap = params['wrap']  
    plane_f=params['plane_f']
    HFoV =params['HFoV'] 
    VFoV = params['VFoV']
    rot = params['R']
    plane_w = (2*plane_f)*np.tan(HFoV/2.0)
    plane_h = (2 * plane_f) * np.tan(VFoV / 2.0)
    H_res = params['H_res']
    W_res = params['W_res']
    H_source=source.shape[0]
    W_source=source.shape[1]
    row,col = np.meshgrid(np.arange(H_res), np.arange(W_res), sparse=False, indexing='ij') 
    #print('row shape '+str(row.shape))
    assert(row[3,4]==3 and col[3,4]==4)
    #exit(1)
    plane_point =   np.zeros((H_res,W_res,3))
    plane_point [:,:,0] =  ((row - H_res / 2) / (1.0 * H_res)) * plane_h
    plane_point [:,:,1]=  ((col - W_res / 2) / (1.0 * W_res)) * plane_w
    plane_point [:,:,2]=  plane_f 
    #print('plane point '+str(plane_point))
    #print('plane point hsape '+str(plane_point.shape))
    #exit(1)
    middle_point = plane_point*0+np.reshape(np.array([ 0, 0, plane_f] ),(1,1,3)) #, 1.0 ])
    #print('middle point shape '+str(middle_point.shape))
    #exit(1)
    vect1 = (plane_point[:,:,0] - middle_point[:,:,0]) * middle_point[:,:,0] + (plane_point[:,:,1] - middle_point[:,:,1]) * middle_point[:,:,1] + (plane_point[:,:,2] - middle_point[:,:,2]) * middle_point[:,:,2]
    assert(np.sum(np.abs(vect1[:,:]) >= 0.0001)==0)
    #exit(1)
    mask_out_of_sphere = (np.linalg.norm(plane_point,axis=-1) > 1.0)
    assert(np.sum(mask_out_of_sphere)==0)
    #exit(1)
    #//obtaining rotated point on a sphere
    plane_rot = np.matmul(np.reshape(rot,(1,1,3,3)),np.reshape(plane_point,(H_res,W_res,3,1)))[:,:,:,0]
    #print('plane rot shape '+str(plane_rot.shape))
    #exit(1)
    assert np.sum((np.abs(np.linalg.norm(plane_point) - np.linalg.norm(plane_rot)) >= 0.0001))==0
    # //If the plane point is within a sphere then proceed
    r_scale = np.reshape(1.0 / np.linalg.norm(plane_rot,axis=-1),(H_res,W_res,1))
    #print('r_shale shape '+str(r_scale.shape))
    #//projecting a point on a sphere
    sp_rot = plane_rot*r_scale
    assert (np.sum(np.abs(np.linalg.norm(sp_rot,axis=-1)-1) >= 0.0001)==0)
    mid_rot = np.matmul (np.reshape(rot,(1,1,3,3)),np.reshape( middle_point,(H_res,W_res,3,1)))[:,:,:,0]
    #print('mid rot shape '+str(mid_rot.shape))
    assert (np.sum((np.abs(np.linalg.norm(middle_point,axis=-1) - np.linalg.norm(mid_rot,axis=-1)) >= 0.0001))==0)
    dif_point = plane_point-middle_point
    dif_point_rot = plane_rot - mid_rot
    assert (np.sum((np.abs(np.linalg.norm(dif_point,axis=-1) - np.linalg.norm(dif_point_rot,axis=-1)) >= 0.0001))==0)
    #//Console.WriteLine(plane_rot[0]+ " "+plane_rot[1]+ " " +plane_rot[2]+ " | " +mid_rot[0]+" "+mid_rot[1]+ " "+mid_rot[2]);
    vect = (plane_rot[:,:,0]- mid_rot[:,:,0]) * mid_rot[:,:,0] + ( plane_rot[:,:,1]- mid_rot[:,:,1]) * mid_rot[:,:,1] + (plane_rot[:,:,2]- mid_rot[:,:,2]) * mid_rot[:,:,2]
    #print('vect shape '+str(vect.shape))
    assert np.sum((np.abs(vect) >= 0.0001)==0)
    #//Checking if the point is still on a sphere
    assert (np.sum((np.abs(1 - np.linalg.norm(sp_rot,axis=-1))>= 0.0001))==0)
    #exit(1)
    #//Obtaining polar coordinates
    polar = cartesian_to_polar(sp_rot[:,:,0], sp_rot[:,:,1], sp_rot[:,:,2])
    #print('polar shape '+str(polar.shape))
    #exit(1)
    latitude_theta = polar[:,:,1] #; //0..pi
    longitude_psi = polar[:,:,2] #;//-pi..pi
    assert (np.sum(np.isnan(longitude_psi))==0)
    #//obtaining row in equirectangular image corresponding to latitude
    eq_row = np.int32(np.round(((latitude_theta) / np.pi) * H_source)) #check for np rounding problems
    #print('eq roqw'+str(eq_row))
    #exit(1)
    # //obtaining row in equirectangular image corresponding to longitude
    eq_col = np.int32(np.round(((longitude_psi + np.pi) / (2 * np.pi)) * W_source))
    if(wrap==True):
        eq_row = (eq_row + H_source) % H_source
        eq_col = (eq_col + W_source) % W_source
    assert (np.min(eq_row)>=0)
    assert (np.min(eq_col) >= 0)
    assert (np.max(eq_row) < H_source)
    assert (np.max(eq_col) < W_source)
    id_image = np.int32(eq_row)*W_source+np.int32(eq_col)
    return id_image

def equirectangular_crop(source,params):
    id_image=equirectangular_crop_id_image(source,params)
    eq_row = id_image // np.int32(source.shape[1])
    eq_col = id_image % np.int32(source.shape[1])
    result = source[eq_row, eq_col, :]
    return result


if __name__ == "__main__":
    params={
         'batch_size':1,
         'wrap':True,
         'H_res': 512,
         'W_res': 1024,
         'plane_f':0.05,
         'HFoV':(58.1599 / 360) * np.pi * 2*1.5,
         'HFoV_range':[0.8,2.5],
         'VFoV': (34.8592 / 360) *np.pi * 2*1.5,
         'VFoV_range':[0.8,2.5],
         'R': euler_to_mat(z=-np.pi/2,y=0,x=np.pi/2)  #range for all [0..2pi), 
        }
    fname='R0010094_20170622125256_er_f_00008010.png'
    img = cv2.imread(fname)
    for i in range(100):
        print('i = 0 '+str(i))      
        temp_params = params.copy()
        # temp_params['HFoV']=((i/100.0)*(temp_params['HFoV_range'][1]-temp_params['HFoV_range'][0]))+temp_params['HFoV_range'][0]
        # temp_params['VFoV']=((i/100.0)*(temp_params['VFoV_range'][1]-temp_params['VFoV_range'][0]))+temp_params['VFoV_range'][0]
        temp_params['R']=euler_to_mat(z=0.0,y=(i/100.0)*np.pi*2,x=0)
        print(temp_params['HFoV'])
        #temp_params['VFoV']=temp_params['VFoV']*(1+(np.random.rand()-0.5)*2*2.0)
        #temp_params['plane_f']=temp_params['plane_f']*(1+(np.random.rand()-0.5)*2*0.3)
        #result_slow=equirectangular_crop_slow(img,temp_params)
        result=equirectangular_crop(img,temp_params)
        print('result shape '+str(result.shape))
        #assert( np.sum(np.abs(result-result_slow))<0.00001)
        cv2.imwrite('./test/'+fname.split('.')[0]+'_C_'+str(i).zfill(6)+'.png',result)
    

