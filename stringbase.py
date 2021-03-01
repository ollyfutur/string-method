# -*- coding: utf-8 -*-
"""
Created on Mon Jun 15 14:30:18 2015

@author: languin

This script aims to find the most probable transition path between two bassins of a free energy landscape with String Method.
"""

import numpy as np
from scipy.interpolate import splprep,splev,RectBivariateSpline
import subprocess
from scipy.linalg import norm

class Landscape:
    def __init__(self,x,y,z,gradient=None):
        self.shape = np.array(z).shape
        self.x = np.array(x)
        self.y = np.array(y)
        self.dx = x[1] - x[0]
        self.dy = y[1] - y[0]
        self.z = np.array(z)
        self._f_z = RectBivariateSpline(self.y,self.x,self.z)
        if gradient is None:
            self.der_y,self.der_x = np.gradient(self.z)
        else: 
            self.der_x,self.der_y = gradient
        self._f_der_x = RectBivariateSpline(self.y,self.x,self.der_x)
        self._f_der_y = RectBivariateSpline(self.y,self.x,self.der_y)
    
    @classmethod
    def from_plumed(cls,datfile):
        data = np.loadtxt(datfile)
        x = np.sort(np.unique(data[:,0]))
        y = np.sort(np.unique(data[:,1]))
        z = data[:,2].reshape(len(y),len(x))
        der_x = data[:,3].reshape(len(y),len(x))
        der_y = data[:,4].reshape(len(y),len(x))
        return cls(x,y,z,(der_x,der_y))

    def compute_gradient(self):
        self.der_x,self.der_y = np.gradient(self.z)

class String:
    def __init__(self,xdata,ydata):
        """
        Parameters
        ----------
        xdata,ydata : array-like
            Coordinates corresponding to both string ends. Must have the same length.
        """ 
        if len(xdata) == len(ydata):
            self.N = len(xdata)
            self.xdata = np.array(xdata)
            self.ydata = np.array(ydata)
        else:
            raise ValueError('xdata and ydata must have the same length.')
        
    @classmethod
    def from_interpolation(cls,x1,y1,x2,y2,N):
        """
        Initializes string from linear interpolation.
        
        Parameters
        ----------
        x1,y1,x2,y2 : floats
            Coordinates of string ends.
        N : int
            Total number of nodes.
        """
        xdata = [x1 + (x2 - x1)*n/(N-1) for n in range(N)]
        ydata = [y1 + (y2 - y1)*n/(N-1) for n in range(N)]
        return cls(xdata,ydata)
    
    def get_pmf(self,landscape):
        """
        Returns a tupple of free energy values along the string.
        """
        pmf = landscape._f_z.ev(self.ydata,self.xdata)
        return pmf
    
    def reparameterize(self, N=None):
    
        """
        Reparameterize the string using cubic spline interpolation 
        and equal arc distances between nodes.
        Parameters
        ----------
        N : int, optional
            Number of nodes. Otherwise, keeps the current number.
        """
        if N is not None:
            self.N = N
    
        tck,u = splprep([self.xdata,self.ydata], s=0)
        out = splev(np.linspace(0,1,self.N), tck)

        self.xdata,self.ydata = out
    
    def random_walk(self, sigmax=1., sigmay=1.):
        """
        Displaces each string node using a random gaussian vector. 
        Parameters
        ----------
        sigmax,sigmay : floats (optional)
            standard deviations along x and y (landscape units)
        """
        deltax = np.random.normal(scale=sigmax, size=self.N)
        deltay = np.random.normal(scale=sigmay, size=self.N)
        self.xdata += deltax
        self.ydata += deltay
   
    def drift(self, landscape, factor):
        """
        Displaces the string nodes down the landscape gradient.
        Parameters
        ----------
        landscape : landscape object
        factor : float
            multiplicative factor for gradient descent
        """
        deltax = factor*landscape._f_der_x.ev(self.ydata,self.xdata)
        deltay = factor*landscape._f_der_y.ev(self.ydata,self.xdata)
        
        self.xdata -= deltax
        self.ydata -= deltay
        
    def untangle(self):
        """
        This function prevents the string to fold and tangle up. 
        If two nonadjacent nodes of the string are closer than the 
        average inter-nodes distance, the nodes inbetween are simply 
        removed (loop excision).
        """    
        cutoff = np.mean(
                    np.sqrt(np.diff(self.xdata)**2 + np.diff(self.ydata)**2)
                    )

        i = 0
        while i < self.N:
            for j in range(i+2, self.N):
                if norm([self.xdata[i] - self.xdata[j],
                         self.ydata[i] - self.ydata[j]]) < cutoff:
                    new_xdata = np.concatenate([self.xdata[0:i+1],
                                                self.xdata[j:]])
                    new_ydata = np.concatenate([self.ydata[0:i+1],
                                                self.ydata[j:]])
                    #print('Deleted nodes : from ', i, ' to ', j)
                    self.xdata = new_xdata
                    self.ydata = new_ydata
                    self.N = self.N - (j - i + 1)
                    break
                else:
                    pass
            i += 1


