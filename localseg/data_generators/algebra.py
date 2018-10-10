import os
import math
import numpy as np


class Algebra:
    @staticmethod
    def cartesian_to_polar(x, y, z):
        #//https://en.wikipedia.org/wiki/Spherical_coordinate_system
        r = np.sqrt(x * x + y * y + z * z)
        latitude_theta = np.arccos(z / r)  # //0..pi
        longitude_psi = np.arctan2(y, x)  # //-pi/2..pi/2
        return np.array([r, latitude_theta, longitude_psi])

    @staticmethod
    def cartesian_to_polar_array(xyz):
        rtp = np.zeros(xyz.shape)
        rtp[:, :, 0] = np.sqrt(xyz[:, :, 0] * xyz[:, :, 0] +
                               xyz[:, :, 1] * xyz[:, :, 1] + xyz[:, :, 2] * xyz[:, :, 2])
        rtp[:, :, 1] = np.arccos(xyz[:, :, 2] / rtp[:, :, 0])  # //0..pi
        rtp[:, :, 2] = np.arctan2(xyz[:, :, 1], xyz[:, :, 0])  # //-pi/2..pi/2
        return rtp

    @staticmethod
    def cartesian_to_polar_array_with_F(xyz, F):
        rtp = np.zeros(xyz.shape)
        rtp[:, :, 0] = np.sqrt(xyz[:, :, 0] * xyz[:, :, 0] + xyz[:, :, 1]
                               * xyz[:, :, 1] + xyz[:, :, 2] * xyz[:, :, 2])*F
        rtp[:, :, 1] = np.arccos(xyz[:, :, 2] / (rtp[:, :, 0]*F))  # //0..pi
        rtp[:, :, 2] = np.arctan2(xyz[:, :, 1], xyz[:, :, 0])  # //-pi/2..pi/2
        return rtp

    @staticmethod
    def polar_to_cartesian(r, latitude_theta, longitude_psi):
        x = r * np.sin(latitude_theta) * np.cos(longitude_psi)
        y = r * np.sin(latitude_theta) * np.sin(longitude_psi)
        z = r * np.cos(latitude_theta)
        return np.array([x, y, z])

    @staticmethod
    def polar_to_cartesian_array(polar):
        xyz = np.zeros(polar.shape)

        xyz[:, :, 0] = polar[:, :, 0] * \
            np.sin(polar[:, :, 1]) * np.cos(polar[:, :, 2])
        xyz[:, :, 1] = polar[:, :, 0] * \
            np.sin(polar[:, :, 1]) * np.sin(polar[:, :, 2])
        xyz[:, :, 2] = polar[:, :, 0] * np.cos(polar[:, :, 1])
        return xyz

    @staticmethod
    def NewRotateAroundX(radians):
        matrix = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [
                          0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
        matrix[0, 0] = np.cos(radians)
        matrix[0, 1] = np.sin(radians)
        matrix[1, 0] = -(np.sin(radians))
        matrix[1, 1] = np.cos(radians)
        matrix[2, 2] = 1
        return matrix

    @staticmethod
    def NewRotateAroundY(radians):
        matrix = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [
                          0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

        matrix[0, 0] = np.cos(radians)
        matrix[0, 2] = -(np.sin(radians))
        matrix[2, 0] = np.sin(radians)
        matrix[2, 2] = np.cos(radians)
        matrix[1, 1] = 1
        return matrix

    @staticmethod
    def NewRotateAroundZ(radians):
        matrix = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [
                          0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])

        matrix[0, 0] = 1
        matrix[1, 1] = np.cos(radians)
        matrix[1, 2] = np.sin(radians)
        matrix[2, 1] = -(np.sin(radians))
        matrix[2, 2] = np.cos(radians)
        return matrix

    @staticmethod
    def rotate_array(R, points):
        points_rot = np.zeros((points.shape[0], points.shape[1], 3))
        points_rot[:, :, 0] = R[0, 0]*points[:, :, 0] + \
            R[0, 1]*points[:, :, 1]+R[0, 2]*points[:, :, 2]
        points_rot[:, :, 1] = R[1, 0]*points[:, :, 0] + \
            R[1, 1]*points[:, :, 1]+R[1, 2]*points[:, :, 2]
        points_rot[:, :, 2] = R[2, 0]*points[:, :, 0] + \
            R[2, 1]*points[:, :, 1]+R[2, 2]*points[:, :, 2]
        return points_rot

    @staticmethod
    def multiply(a, b):
        matrix = np.array([[0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [
                          0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        for i in range(0, 4):
            for j in range(0, 4):
                for k in range(0, 4):
                    matrix[i, j] += a[i, k] * b[k, j]
        return matrix

    @staticmethod
    def rotation_matrix(euler_angles):
        R = Algebra.multiply(Algebra.multiply(Algebra.NewRotateAroundZ(
            euler_angles[2]), Algebra.NewRotateAroundY(euler_angles[1])), Algebra.NewRotateAroundX(euler_angles[0]))
        return R

    @staticmethod
    def cartesian_to_polar_quantised_array(wrap, cartesian_points, width, height):
        polar_points = Algebra.cartesian_to_polar_array(cartesian_points)

        latitude_theta = polar_points[:, :, 1]  # ; //0..pi
        longitude_psi = polar_points[:, :, 2]  # ;//-pi..pi

        eq_row = (
            np.round(((latitude_theta[:, :]) / np.pi) * height)).astype(np.int32)
        eq_col = (np.round(
            ((longitude_psi[:, :] + np.pi) / (2 * np.pi)) * width)).astype(np.int32)

        if (wrap == True):
            eq_row = (eq_row + height) % height
            eq_col = (eq_col + width) % width
        else:
            eq_row[eq_row[:, :] >= height] = height-1
            eq_col[eq_col[:, :] >= width] = width-1
        return eq_row, eq_col

    @staticmethod
    def cartesian_to_polar_quantised_array_with_F(wrap, cartesian_points, width, height, F):
        polar_points = Algebra.cartesian_to_polar_array_with_F(
            cartesian_points, F)

        latitude_theta = polar_points[:, :, 1]  # ; //0..pi
        longitude_psi = polar_points[:, :, 2]  # ;//-pi..pi

        eq_row = (
            np.round(((latitude_theta[:, :]) / np.pi) * height)).astype(np.int32)
        eq_col = (np.round(
            ((longitude_psi[:, :] + np.pi) / (2 * np.pi)) * width)).astype(np.int32)

        if (wrap == True):
            eq_row = (eq_row % height + height) % height
            eq_col = (eq_col % width + width) % width
        else:
            eq_row[eq_row[:, :] >= height] = height-1
            eq_col[eq_col[:, :] >= width] = width-1
        return eq_row, eq_col

    @staticmethod
    def test_polar_to_polar():
        N = 100
        for theta in range(1, N):
            for psi in range(-N, N):
                ipolar = np.array(
                    [1, (theta / (N*1.0)) * (np.pi), (psi / (N*1.0)) * (np.pi)])

                cart = Algebra.polar_to_cartesian(
                    ipolar[0], ipolar[1], ipolar[2])

                polar = Algebra.cartesian_to_polar(cart[0], cart[1], cart[2])

                dist = (polar[0] - ipolar[0]) * (polar[0] - ipolar[0]) + (polar[1] - ipolar[1]) * (
                    polar[1] - ipolar[1]) + (polar[2] - ipolar[2]) * (polar[2] - ipolar[2])

                if (dist >= 0.001):
                    print("------------------")
                    print(str(ipolar))
                    print(str(cart))
                    print(str(polar))
                assert(dist < 0.001), " Failing polar to polar conversion"

    @staticmethod
    def magnitude(vector):
        return np.sqrt(vector[0] * vector[0] + vector[1] * vector[1] + vector[2] * vector[2])

    @staticmethod
    def magnitude_array(vector):
        return np.sqrt(vector[:, :, 0] * vector[:, :, 0] + vector[:, :, 1] * vector[:, :, 1] + vector[:, :, 2] * vector[:, :, 2])
