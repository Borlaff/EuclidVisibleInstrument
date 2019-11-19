#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 17:38:42 2019

@author: rvavrek

Modified by Alejandro S. Borlaff - May 2019
"""

"""

Euclid VIS simulator
"""


from astropy.io import fits
import matplotlib.pyplot as plt
#import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
#from matplotlib.ticker import LinearLocator
import numpy as np
#from matplotlib.colors import LogNorm
from scipy import interpolate
#from scipy import ogrid, sin, mgrid, ndimage, array
import sys
import os
import math
import warnings
import tars
import multiprocessing
from tqdm import tqdm
from scipy.stats import sigmaclip
from astropy.modeling import models, fitting
from GIT.VISsim.sources import stellarNumberCounts
from GIT.VISsim.support import cosmicrays as cr
from GIT.VISsim.support import logger as lg
from GIT.VISsim.support import files as fileIO
from scipy import ndimage

# https://www.python-course.eu/matplotlib_multiple_figures.php

saveDir = '/home/borlaff/ESA/Euclid/SKYSIM/'



#%%
def multivariate_gaussian(pos, mu, Sigma):
    """Return the multivariate Gaussian distribution on array pos.

    pos is an array constructed by packing the meshed arrays of variables
    x_1, x_2, x_3, ..., x_k into its _last_ dimension.

    """

    n = mu.shape[0]
    Sigma_det = np.linalg.det(Sigma)
    Sigma_inv = np.linalg.inv(Sigma)
    N = np.sqrt((2*np.pi)**n * Sigma_det)
    # This einsum call calculates (x-mu)T.Sigma-1.(x-mu) in a vectorized
    # way across all the input variables.
    fac = np.einsum('...k,kl,...l->...', pos-mu, Sigma_inv, pos-mu)

    return np.exp(-fac / 2) / N

#%%
def visPsfRead(psfFileName,verbose=0):
    
    # Read PSF file
    hdulist = fits.open(psfFileName)
    hdulist.info()
    hdulist[0].header
    psf = hdulist[0].data
    hdulist.close()
    psfOver = hdulist[0].header["OVERSAMP"]
    
    return psf,psfOver

#%%
def visPsfInterpolate(Z,psfOver,x1PeakSource,y1PeakSource,verbose=0):
    
    #
    N0 = Z.shape[0]
    if Z.shape[0] != Z.shape[1]: print("Error")
    
    x0Grid = np.arange(-(N0/2),N0/2,1) +0.5 
    y0Grid = np.arange(-(N0/2),N0/2,1) +0.5 
        
    x0D = psfOver * (int(x1PeakSource)+0.5-x1PeakSource)
    y0D = psfOver * (int(y1PeakSource)+0.5-y1PeakSource)
    
    if int(N0 % psfOver) == 0 :
        N1 = N0 / psfOver
    else:
        N1 = int(N0 / psfOver) + 1
             
    x1Grid = np.empty([int(N1)])
    y1Grid = np.empty([int(N1)])
    
    pMultipliers = np.arange(0,(N1/2),1)
    nMultipliers = np.arange((N1/2),0,-1)
    
    x1Grid[int(N1/2):int(N1)] = (0.5 + x0D) + (pMultipliers * psfOver)
    x1Grid[0:int(N1/2)]  = (0.5 + x0D) - (nMultipliers * psfOver)
    
    y1Grid[int(N1/2):int(N1)] = (0.5 + y0D) + (pMultipliers * psfOver)
    y1Grid[0:int(N1/2)]  = (0.5 + y0D) - (nMultipliers * psfOver)
    
    
    f = interpolate.interp2d(x0Grid, y0Grid, Z, kind='cubic')
    
    ZZ = f(x1Grid, y1Grid)
    
    return ZZ

#%%
def visPsfInsert(Z,ZZZ,psfOversampling,xStarPositions,yStarPositions,magnitudes,xsize,ysize,xyMargin,magzero,exptime,verbose=0):
    
    #
    counter_fail = 0
    
    for i in range(len(xStarPositions)):
            
        ZZ = visPsfInterpolate(Z,psfOversampling,xStarPositions[i],yStarPositions[i])
        
        maxpos = np.unravel_index(ZZ.argmax(), ZZ.shape)
        xRefPos = int(xStarPositions[i]) - maxpos[0]
        yRefPos = int(yStarPositions[i]) - maxpos[1]
        xStartPos = xRefPos+int(xyMargin/2)
        yStartPos = yRefPos+int(xyMargin/2)
        
        ZZnorm = ZZ / ZZ.sum()
        
        ZZstarTotalADU  = (10.0**(-0.4 * magnitudes[i])) * magzero * exptime * ZZnorm 
        
        #print(magnitudes[i], np.sum(ZZstarTotalADU), np.max(ZZstarTotalADU),np.min(ZZstarTotalADU),np.max(ZZnorm))
        
        verbose=0
        if verbose:
            print(i)
            print("ZZ shape", ZZ.shape)
            print(maxpos,xStarPositions[i],yStarPositions[i])
            print(xRefPos,yRefPos)
            print(xRefPos+maxpos[0],yRefPos+maxpos[1])
            print(xStartPos,xStartPos+ZZ.shape[0],yStartPos,yStartPos+ZZ.shape[1])
        
        try:
            ZZZ[xStartPos:xStartPos+ZZ.shape[0]+0,yStartPos:yStartPos+ZZ.shape[1]+0] = ZZZ[xStartPos:xStartPos+ZZ.shape[0]+0,yStartPos:yStartPos+ZZ.shape[1]+0] + ZZstarTotalADU
        except:
            counter_fail =+ counter_fail
            

    xyOffset = int(xyMargin/2)    
    ZZZZ = ZZZ[xyOffset:xyOffset+xsize,xyOffset:xyOffset+ysize]

    if verbose: 
        print("MAXPOS", np.unravel_index(ZZZZ.argmax(), ZZZZ.shape))
        print('Failed to insert star ', counter_fail, ' times')
    
    return ZZZZ

#%%
def visGaussianPsfCreate(verbose=0):
    
    #
    N0 = 60
    X = np.arange(-(N0/2),N0/2,1)
    Y = np.arange(-(N0/2),N0/2,1)   
    X, Y = np.meshgrid(X, Y)
    
    # Mean vector and covariance matrix
    mu = np.array([0., 0.])
    #Sigma = np.array([[ 12. , -2.5], [-10.5,  7.5]])
    Sigma = np.array([[ 16. , 16.], [-16. , 16.]])
    
    # Pack X and Y into a single 3-dimensional array
    pos = np.empty(X.shape + (2,))
    pos[:, :, 0] = X
    pos[:, :, 1] = Y
    
    # Create test image
    Z = multivariate_gaussian(pos, mu, Sigma)

    if verbose:
        # Create a surface plot and projected filled contour plot under it.
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot_surface(X, Y, Z, rstride=3, cstride=3, linewidth=1, antialiased=True, cmap=cm.viridis)
        
        cset = ax.contourf(X, Y, Z, zdir='z', offset=-0.15, cmap=cm.viridis)
        
        # Adjust the limits, ticks and view angle
        ax.set_zlim(0.,0.03)
        ax.set_zticks(np.linspace(0,0.03,0.01))
        ax.view_init(27, -21)
        
    return Z

#%%
def getInputStarCatalogue(magMin,magMax,magBinsize,lon,lat,area,xsize,ysize,nStars,modelStars='BS',verbose=0):
    
    #
    if modelStars == 'BS':
        
        # Model stellar densities per area
        Nvconst = stellarNumberCounts.integratedCountsVband()
        print(Nvconst)
        magnitudes = np.arange(magMin, magMax+(magBinsize/2), magBinsize)
            
        starPerBin = np.empty(len(magnitudes))
        starDensitiesPerSqDegPerMagBin = np.zeros([2,len(magnitudes)-1])
        
        ind = -1
        prev = stellarNumberCounts.bahcallSoneira(magnitudes[0]-magBinsize, lon, lat, Nvconst)
        
        for m in magnitudes:

            ind = ind+1
            n = stellarNumberCounts.bahcallSoneira(m, lon, lat, Nvconst)
            
            N = n - prev 
            
            starPerBin[ind] = N * area
            
            if ind > 0:
                starDensitiesPerSqDegPerMagBin[0,ind-1] = m - (magBinsize/2)
                starDensitiesPerSqDegPerMagBin[1,ind-1] = N
            
            prev = n
                        
        
        # Simulate catalogue
        Nmax = stellarNumberCounts.bahcallSoneira(magMax, lon, lat, Nvconst)
        Nmin = stellarNumberCounts.bahcallSoneira(magMin, lon, lat, Nvconst)
        
        Nsim = int((Nmax-Nmin)*area)
        
        catalogueInitial = np.zeros([3,Nsim+200])
        ind = 0
        indAllsource = -1
    
        for i in tqdm(range(len(magnitudes)-1)):

            ind = ind + 1
            if np.floor(starPerBin[ind]) > 0:
                
                candidates = 0
                
                while candidates < int(starPerBin[ind]):
                      
                    
                    xMonteCarlo = np.random.rand()+magnitudes[ind-1]
                    yMonteCarlo = np.random.rand()*magnitudes[ind]
                    
            
                    if yMonteCarlo <= np.interp(xMonteCarlo, magnitudes, starPerBin):
                       
                        candidates = candidates + 1
                        xStarPosition = (np.random.rand()*xsize)
                        yStarPosition = (np.random.rand()*ysize)
                        
                        indAllsource = indAllsource + 1
                        catalogueInitial[:,indAllsource] = [xStarPosition,yStarPosition,xMonteCarlo]
        
        
        catalogueInitial = catalogueInitial[:,0:indAllsource]
        
        catalogueFinal = np.zeros([3,Nsim+200])
        ind = -1
        
        for i in range(indAllsource):
        
            mag = catalogueInitial[2,i]
            if (mag >= magMin) and (mag <= magMax):
                ind = ind+1
                catalogueFinal[:,ind] = catalogueInitial[:,i]
            
        catalogueFinal = catalogueFinal[:,0:ind]
    
        if verbose:
            print('Check: ',Nsim,np.sum(starPerBin),indAllsource,ind)

        
    if modelStars == 'Uniform':
        
        catalogueFinal = np.zeros([3,nStars])
        #xStarPositions = (np.random.rand(nStars)*xsize) #This variables do nothing here, they are overwritten
        #yStarPositions = (np.random.rand(nStars)*ysize) #This variables do nothing here, they are overwritten
        magnitudes     = (np.random.rand(nStars)*(magMax-magMin))+magMin
        
        xStarPositions = np.floor(np.random.rand(nStars)*xsize)+0.5
        yStarPositions = np.floor(np.random.rand(nStars)*ysize)+0.5
        
        
        catalogueFinal[0,:] = xStarPositions
        catalogueFinal[1,:] = yStarPositions
        catalogueFinal[2,:] = magnitudes
        
        starDensitiesPerSqDegPerMagBin = np.zeros([2,1])
        
    return catalogueFinal, starDensitiesPerSqDegPerMagBin



#%%
def visStarsInsert(Z,psfOversampling,starCatalogue,xsize,ysize,xyMargin,magzero,exptime,verbose=0):
    
    #
    ZZZ = np.empty([xsize+xyMargin,ysize+xyMargin])
    #ZZZ[:,:] = 0.0001 # Ugly workaround for displaying
    
    print(starCatalogue)
    xStarPositions = starCatalogue[0,:]
    yStarPositions = starCatalogue[1,:]
    magnitudes     = starCatalogue[2,:]
    
    ZZZZ = visPsfInsert(Z,ZZZ,psfOversampling,xStarPositions,yStarPositions,magnitudes,xsize,ysize,xyMargin,magzero,exptime,verbose=verbose)
    
    return ZZZZ


#%%
def visApplyPoissonNoise(ZZZZ):
     """
     Add Poisson noise to the image.
     """
     
     # THIS IS INCORRECT!!! noisePoission as ALREADY image+noise!  
     ZZZZ = np.random.poisson(ZZZZ).astype(np.float64)
        
     return ZZZZ

#%%
def visApplyDarkCurrent(ZZZZ,dark,exptime):
    """
    Apply dark current. Scales the dark with the exposure time.

    """

    compValue = exptime * dark
    ZZZZ = ZZZZ + compValue

    return ZZZZ

#%%
def visCorrectDarkCurrent(ZZZZ,dark,exptime):
    """
    Apply dark current. Scales the dark with the exposure time.

    """

    compValue = exptime * dark
    ZZZZ = ZZZZ - compValue

    return ZZZZ


#%%
def visApplyBias(ZZZZ,bias):
    """
    Apply bias current.

    """

    compValue = bias
    ZZZZ = ZZZZ + compValue

    return ZZZZ


#%%
def visApplyGap(ZZZZ):
    """
    Apply gap between sectors of detector.

    """

    ZZZZ[2066:2070,:] = 0

    return ZZZZ


#%%
def visCorrectBias(ZZZZ,bias):
    """
    Apply bias current.

    """

    compValue = bias
    ZZZZ = ZZZZ - compValue

    return ZZZZ


def visAddCosmicRays(ZZZZ, fractionCR):
    #set up logger
    log = lg.setUpLogger('VISsim.log')
    #test section
    crImage = ZZZZ[:,:]*0
    #cosmic ray instance
    cosmics = cr.cosmicrays(log, crImage)
    #add cosmic rays up to the covering fraction
    CCD_cr = cosmics.addUpToFraction(fractionCR, limit=None, verbose=True)
    return(ZZZZ + CCD_cr)


#%%
def visApplyFlatField(ZZZZ,flat_name, ext=0):
    """
    Apply multiplicative flat field across the detector, based on the input flat_name file.

    """

    compValue = fits.open(flat_name)[ext].data
    ZZZZ = ZZZZ*compValue

    return ZZZZ


#%%
def visApplyCosmicBackground(ZZZZ,cosmic_bkgd,exptime):
    """
    Apply dark the cosmic background. Scales the background with the exposure time.

    """

    #add background
    compValue = exptime * cosmic_bkgd
    ZZZZ = ZZZZ + compValue

    return ZZZZ

#%%
def visApplyScatteredLight(ZZZZ,scattered_light,exptime):
    """
    Adds spatially uniform scattered light to the image.

    """

    #add background
    compValue = exptime * scattered_light
    ZZZZ = ZZZZ + compValue

    return ZZZZ

#%%
def visApplyReadoutNoise(ZZZZ,readout):
    """
    Applies readout noise to the image being constructed.
    The noise is drawn from a Normal (Gaussian) distribution with average=0.0 and std=readout noise.
    """
    compValue = np.random.normal(loc=0.0, scale=readout, size=ZZZZ.shape)
    ZZZZ = ZZZZ + compValue

    return ZZZZ

#%%

#%%
def visElectrons2Photons(ZZZZ,qe):
    """
    Convert from electrons to ADUs using the value read from the configuration file.
    """
    
    ZZZZ = ZZZZ.astype(np.float64)/qe
    ZZZZ[ZZZZ < 0] = 0

    return ZZZZ

#%%
def visPhotons2Electrons(ZZZZ,qe):
    """
    Convert from electrons to ADUs using the value read from the configuration file.
    """
    
    ZZZZ = ZZZZ.astype(np.float64)*qe
    ZZZZ[ZZZZ < 0] = 0

    return ZZZZ


def visElectrons2ADU(ZZZZ,e_adu):
    """
    Convert from electrons to ADUs using the value read from the configuration file.
    """
    
    ZZZZ = ZZZZ.astype(np.float64) / e_adu
    ZZZZ[ZZZZ < 0] = 0

    return ZZZZ

#%%
def visDiscretise(ZZZZ,max=2**16-1):
    """
    Converts a floating point image array (self.image) to an integer array with max values
    defined by the argument max.

    :param max: maximum value the the integer array may contain [default 65k]
    :type max: float

    """
    
    #avoid negative numbers in case bias level was not added
    #self.image[self.image < 0.0] = 0.
    #cut of the values larger than max

    ZZZZ[ZZZZ > max] = max
    ZZZZ = np.rint(ZZZZ).astype(np.int)
    ZZZZ[ZZZZ < 0] = 0
    #ZZZZ = ZZZZ.astype("float")
    #ZZZZ = ZZZZ * 0.0005
        
    return ZZZZ

#%%
def visPsfSaturation(Z,psfOversampling,magMin,magMax,magBinsize,magzero,exptime,zeroOffset,verbose=0):
    
    #
    magnitudes = np.arange(magMin, magMax+(magBinsize/2), magBinsize)
    
    xStarPos = 0.5
    yStarPos = 0.5
    
    pixelsAtSaturation = np.zeros([2,len(magnitudes)-1])
   
    ind = -1
        
    for m in magnitudes:
            
        ind = ind+1
        
        ZZs = visPsfInterpolate(Z,psfOversampling,xStarPos,yStarPos)
        ZZnorm = ZZs / ZZs.sum()
        
        magnitude = m - (magBinsize/2)
        
        ZZstarTotalElectrons  = (10.0**(-0.4 * magnitude)) * magzero * exptime * ZZnorm 
        ZZs = visElectrons2ADU(ZZstarTotalElectrons,e_adu) + zeroOffset
        ZZs = visDiscretise(ZZs)
        subs = np.where(ZZs >= 2**16-1)
        Nsaturated = len(subs[0])
        
        if ind > 0:
           pixelsAtSaturation[0,ind-1] = magnitude
           pixelsAtSaturation[1,ind-1] = Nsaturated
           print(magnitude,np.min(ZZs),np.max(ZZs),Nsaturated)
                
        if magnitude == 10.5:
            ZZs[subs[0],subs[1]] = 2**16-1
#            imageplot   = plt.imshow(ZZs,norm=LogNorm())
#            #imageplot   = plt.imshow(ZZs)
#            plt.show()
             
            print(np.sum(ZZs))
            fig = plt.figure()
            ax = Axes3D(fig)
            X = np.arange(0, ZZs.shape[0], 1)
            Y = np.arange(0, ZZs.shape[1], 1)
            X, Y = np.meshgrid(X, Y)
            ax.plot_surface(X, Y, ZZs, rstride=1, cstride=1, cmap='hot')
            plt.show()

       
        
    if verbose: 
        print("Done...")
    
    return pixelsAtSaturation

#%%
def visApplyBlooming(img,FWC=2.E5):
    
    """
    Adapted from script by Ruymann
    Apply bleeding along the CCD columns if the number of electrons in a pixel exceeds the full-well capacity.

    Bleeding is modelled in the parallel direction only, because the CCD273s are assumed not to bleed in
    serial direction.

    :return: None
    """

    
    bimg = img.copy() 
    size = np.shape(bimg)
 
    #loop over each column, as bleeding is modelled column-wise
    
    for c in range(size[0]):
        column = bimg[c,:]
        
        _sum = 0.
        for j, value in enumerate(column):
            #first round - from bottom to top (need to half the bleeding)
            overload = value - FWC
            if overload > 0.:
                   overload /= 2.
                   bimg[j, i] -= overload
                   _sum += overload
            elif sum > 0.:
                   if -overload > _sum:
                       overload = -_sum
                   bimg[j, i] -= overload
                   _sum += overload
    
        for i, column in enumerate(bimg.T):
            _sum = 0.
            for j, value in enumerate(column[::-1]):
                #second round - from top to bottom (bleeding was half'd already, so now full)
                overload = value - FWC
                if overload > 0.:
                    bimg[-j-1, i] -= overload
                    _sum += overload
                elif _sum > 0.:
                    if -overload > _sum:
                        overload = -_sum
                    bimg[-j-1, i] -= overload
                    _sum += overload
        
    return bimg



#%%
def pixelsImpacted(magnitude, magzero=15861729325.3279, fullwellcapacity=200000, exptime=565, pixelFractions=(0.65, 0.4, 0.35, 0.18, 0.09, 0.05),
                   star=False, lookup=True):
    """

    This potentially overestimates because does not consider the fact that bleeding is along the column
    and hence some saturated pixels may be double counted.
    """
    if lookup:
        #use a lookup table
        data = [(0, 311609),
                (1, 251766),
                (2, 181504),
                (3, 119165),
                (4, 75173),
                (5, 46298),
                (6, 28439),
                (7, 18181),
                (8, 12491),
                (9, 7552),
                (10, 4246),
                (11, 1652),
                (12, 636),
                (13, 247),
                (14, 93),
                (15, 29),
                (16, 8),
                (17, 2),
                (18, 1),
                (19, 0),
                (20, 0)]
        data.sort()

        pos = bisect.bisect_left(data, (magnitude - 0.99,))
        return data[pos][1]
    else:
        #try to calculate

        zp = 2.5 * np.log10(magzero)
        fw = fullwellcapacity

        electrons = 10**(-.4*(magnitude - zp)) * exptime

        mask = 0
        for x in pixelFractions:
            mask += np.round(electrons * x / fw - 0.4)  #0.4 as we don't want to mask if say 175k pixels...
            print(np.round(electrons * x / fw),mask)

        if star:
            mask += (20*20)

        if mask > 2000**2:
            mask = 2000**2

        return mask

#%%


#%%

def visPsfFit(psf,psfOver):
    #
    # Fit undersampled PSF with 2D Gaussian

    maxpos       = np.unravel_index(psf.argmax(), psf.shape)
    psf_under    = visPsfInterpolate(psf,psfOver,maxpos[0],maxpos[1],verbose=0)
    maxpos_under = np.unravel_index(psf_under.argmax(), psf_under.shape)
    p_init       = models.Gaussian2D(amplitude=np.max(psf_under), x_mean=maxpos_under[0], y_mean=maxpos_under[1], x_stddev=3, y_stddev=3)
    fit_p        = fitting.LevMarLSQFitter()
    psfshape     = psf_under.shape
    y, x         = np.mgrid[:psfshape[0], :psfshape[1]]
    
    with warnings.catch_warnings():
        # Ignore model linearity warning from the fitter
        warnings.simplefilter('ignore')
        model = fit_p(p_init, x, y, psf_under)
            
    return model

#%%
    
def fitCentroidGauss2D(catalogueIN,visImage,modelFit,centroidInputError,centroidBiasX,centroidBiasy):
    #
    
    visImage     = visImage - np.median(visImage)
    fit_p        = fitting.LevMarLSQFitter()
    catalogueOUT = np.zeros([2,catalogueIN.shape[1]])
    
    for i in range(catalogueIN.shape[1]):
        
        angle = np.random.rand(1) * 360
        centroidErrorX = centroidInputError * np.cos(angle * np.pi / 180)
        centroidErrorY = centroidInputError * np.sin(angle * np.pi / 180)
        
        #centroidErrorX = np.random.rand(1) * centroidInputError / np.sqrt(2)
        #centroidErrorY = np.random.rand(1) * centroidInputError / np.sqrt(2)
        
        
        try:
            starCrop  = visImage[np.int(np.floor(catalogueIN[0,i])-10) : np.int(np.floor(catalogueIN[0,i])+10), np.int(np.floor(catalogueIN[1,i])-10): np.int(np.floor(catalogueIN[1,i])+10)]
            maxpos    = np.unravel_index(starCrop.argmax(), starCrop.shape)
            p_init    = models.Gaussian2D(amplitude=np.max(starCrop), x_mean=maxpos[0]+centroidErrorX, y_mean=maxpos[1]+centroidErrorY, x_stddev=modelFit.x_stddev.value, y_stddev=modelFit.y_stddev.value, theta=modelFit.theta.value)
            sh        = starCrop.shape
            y, x      = np.mgrid[:sh[0], :sh[1]]
            
            with warnings.catch_warnings():
                # Ignore model linearity warning from the fitter
                warnings.simplefilter('ignore')
                model = fit_p(p_init, x, y, starCrop)
        
            catalogueOUT[0,i] = model.x_mean.value+np.floor(catalogueIN[0,i])-9.5 - centroidBiasX 
            catalogueOUT[1,i] = model.y_mean.value+np.floor(catalogueIN[1,i])-9.5 - centroidBiasY
            
        except:
            
            catalogueOUT[0,i] = np.nan
            catalogueOUT[1,i] = np.nan


    return catalogueOUT

#%%
    
def centroidError(catalogueIN, catalogueOUT):
    #
       
    errorOUT = np.zeros([3,catalogueIN.shape[1]])
    
    for i in range(catalogueIN.shape[1]):
        
        errorOUT[2,i] = math.hypot(catalogueIN[0,i] - catalogueOUT[0,i], catalogueIN[1,i] - catalogueOUT[1,i])
        
        #print(catalogueIN[0,i]-catalogueOUT[0,i], catalogueIN[1,i]-catalogueOUT[1,i], errorOUT[2,i])
        
        errorOUT[0,i] = catalogueIN[0,i]-catalogueOUT[0,i]
        errorOUT[1,i] = catalogueIN[1,i]-catalogueOUT[1,i]
        
    return errorOUT


#%%

def centroidStatistics(centroidDiffs):
    #
    
    sel_cd = np.isfinite(centroidDiffs[0,:])

    x_diffs = centroidDiffs[0,sel_cd]
    y_diffs = centroidDiffs[1,sel_cd]
    d_diffs = centroidDiffs[2,sel_cd]

    x_diffs_s, low, upp = sigmaclip(x_diffs, 5, 5)
    y_diffs_s, low, upp = sigmaclip(y_diffs, 5, 5)
    d_diffs_s, low, upp = sigmaclip(d_diffs, 5, 5)


    return np.mean(x_diffs_s), np.std(x_diffs_s), np.mean(y_diffs_s), np.std(y_diffs_s), np.mean(d_diffs_s), np.std(d_diffs_s)
        


# Main
def createVisSim(outname):
    #%%
    # Euclid mission configuration
    
    #ccdx=int(opts.xCCD),
    #ccdy=int(opts.yCCD),
    psfoversampling=6.0
    ccdxgap=1.643
    ccdygap=8.116
    xsize=int(6000)
    ysize=int(6000)
    prescanx=50
    ovrscanx=20

    ccdxsize=int(4096)
    ccdysize=int(4136)

    fullwellcapacity=200000
    dark=0.001 # e-/s 
    readout=4.5
    bias=200.0
#    cosmic_bkgd=0.182758225257 # e-/s Variable for function
    scattered_light=2.96e-2 # e-/s Variable for function
    cosmic_bkgd=0.3006107 # e-/s Variable for function
    scattered_light_sd=scattered_light*0.05 # e-/s Variable for function
    cosmic_bkgd_sd=cosmic_bkgd*0.05 # e-/s Variable for function

#    scattered_light=0 # e-/s Variable for function

    e_adu=3.1
    qe = 0.75
    #magzero=15861729325.3279
    #magzero=15182880871.225231
    magzero = 18593523364. # In e-. This zp to have m = -2.5 log10(ADU) + 24.445, like the OUVIS Sims 
    exposures=1
    exptime=565.0
    rdose=8.0e9
    ra=223.0
    dec=45.0
    injection=45000.0
    ghostCutoff=22.0
    ghostRatio=5.e-5
    fractionCR = 1.4
    pixscale = 0.101 #arcsec per 12μm pixel
    fovx = xsize*pixscale #0.72 # deg
    fovy = ysize*pixscale #0.79 # deg 
    zeroOffset = 2500

              
    # Simulated objects
    nStars   = 5  # Variable for function
    modelStars = 'Uniform' #'BS' # 'Uniform'
    lon = 180.
    lat = 30.
    magMin = 20.00 # Variable for function
    magMax = 20.00 # Variable for function
    magBinsize = 0.1

    #
    centroidBiasX = 0.150
    centroidBiasY = 0.07    
    centroidInputError = 1.2 #0.2

    # Simulator configuration
    xyMargin = 100
    CCDs     = 1
    area = ((fovx * fovy) / 36.) * CCDs

    #MAGZEROP=               24.445# / zero-point                                     
    #PHOTIRMS=                0.078# / mag dispersion RMS     

    # Environment configuration
    # psfFileName = '/lhome/aserrano/EMDB/EUC_VIS_PSF-WFE-AFFINEDFT-SC3_20161108T210137.4Z_01.00.fits'
    # Running tests with Osiris PSF from GTC - 6 spikes 
    psfFileName = saveDir + '/psf_osiris_r_crop_norm.fits'
    flat_name = saveDir + '/flat_res_ccd00.fits'
    #%%


    psf,psfOver = visPsfRead(psfFileName,verbose=0)
    starCatalogue, starDensities = getInputStarCatalogue(magMin,magMax,magBinsize,lon,lat,area,xsize,ysize,nStars,modelStars=modelStars,verbose=1)
    visCCD      = visStarsInsert(psf,psfOver,starCatalogue,xsize,ysize,xyMargin,magzero,exptime,verbose=1)
    # Here we crop the visCCD array
    #visCCD      = np.zeros([6000,6000])
    visCCD      = visCCD[int(ysize/2-ccdysize/2):int(ysize/2+ccdysize/2), int(xsize/2-ccdxsize/2):int(xsize/2+ccdxsize/2)] # Move to a function

    #visCCD      = visApplyCosmicBackground(visCCD, cosmic_bkgd, exptime)  # Verificar fotones background. (This includes CIB, ISM and Zody?)  
    ##visCCD      = visApplyReadoutNoise(visCCD,cosmic_bkgd_sd) # Adding 5% random gaussian noise to the sky background
    #visCCD      = visApplyScatteredLight(visCCD, scattered_light, exptime)  # Adding 5% random gaussian noise to the scattered light # Verificar y añadir fotones scattered light
    # ADD SCATTERED LIGHT GRADIENTS
    ##visCCD      = visApplyReadoutNoise(visCCD,scattered_light_sd)
    #visCCD      = visElectrons2Photons(visCCD,qe)
    #visCCD      = visDiscretise(visCCD)
    #visCCD      = visApplyPoissonNoise(visCCD)
    #visCCD      = visDiscretise(visCCD)
    ############ THE LIGHT ENTERS THE TELESCOPE  ################################
    #visCCD      = visApplyFlatField(visCCD, flat_name, ext=1)     # Apply flat field
    #visCCD      = visPhotons2Electrons(visCCD,qe)
    #visCCD      = visAddCosmicRays(visCCD, fractionCR)
    #visCCD      = visApplyDarkCurrent(visCCD, dark, exptime)        # Apply Dark current
    #visCCD      = visApplyBias(visCCD,bias)
    #visCCD      = visDiscretise(visCCD)
    #visCCD      = visApplyReadoutNoise(visCCD,readout)
    #visCCD      = visDiscretise(visCCD)
   ########## NOW FULLY FORMED RAW FILE ###########################
    #visCCD      = visCorrectBias(visCCD,bias)
    #visCCD      = visCorrectDarkCurrent(visCCD,dark,exptime)
    visCCD      = visElectrons2ADU(visCCD,e_adu)
    visCCD      = visApplyGap(visCCD)
     
    hdu = fits.PrimaryHDU(visCCD)
    if os.path.exists(outname):
        os.remove(outname)

    hdu.header["EMAGZERO"] = magzero
    hdu.header["PSF_FILE"] = psfFileName
    hdu.header["FLATFILE"] = flat_name
    hdu.header["CCDXGAP"] = ccdxgap
    hdu.header["CCDYGAP"] = ccdygap
    hdu.header["PRESCANX"] = prescanx
    hdu.header["OVRSCANX"] = ovrscanx
    hdu.header["FWELLCAP"] = fullwellcapacity
    hdu.header["DARK"] = dark
    hdu.header["READOUT"] = readout
    hdu.header["BIAS"] = bias
    hdu.header["COSMICBK"] = cosmic_bkgd
    hdu.header["COSMICBK_SD"] = cosmic_bkgd_sd
    hdu.header["SCTRLGHT"] = scattered_light
    hdu.header["SCTRLGHT_SD"] = scattered_light_sd 
    hdu.header["E2ADU"] = e_adu
    hdu.header["QE"] = qe
    hdu.header["NEXP"] = exposures
    hdu.header["E2ADU"] = e_adu
    hdu.header["EXPTIME"] = exptime
    hdu.header["RDOSE"] = rdose
    hdu.header["E2ADU"] = e_adu
    hdu.header["RA"] = ra
    hdu.header["DEC"] = dec
    hdu.header["INJECTION"] = injection
    hdu.header["GHOSTCUT"] = ghostCutoff
    hdu.header["GHOTSRATIO"] = ghostRatio
    hdu.header["CR_FRAC"] = fractionCR
    hdu.header["PIXSCALE"] = pixscale
    hdu.header["ZERO_OFF"] = zeroOffset
    hdu.header["NSTARS"] = nStars
    hdu.header["STAR_MODEL"] = modelStars
    hdu.header["LON"] = lon
    hdu.header["LAT"] = lat
    hdu.header["MAGMIN"] = magMin
    hdu.header["MAGMAX"] = magMax
    hdu.header["MAGBINSIZE"] = magBinsize
    hdu.header["CENBIASX"] = centroidBiasX
    hdu.header["CENBIASY"] = centroidBiasY
    hdu.header["CENERROR"] = centroidInputError
    hdu.header["XYMARGIN"] = xyMargin
    hdu.header["CCDS"] = CCDs

    hdu.writeto(outname)

    return(outname)


######################################
                                     #
version = "v11"                       #
nsimul=1                           #
os.system("rm " + saveDir + "simVis_" + version + "_*.fits")                                     #
######################################


tars.normalize_frame([saveDir + "/flat_res_ccd00.fits"],1) 
#nproc = multiprocessing.cpu_count() - 2
#pool = multiprocessing.Pool(processes=nproc)
outname_list = []
for j in range(nsimul):
    outname_list.append("simVis_" + version + "_" + str(j).zfill(5) + ".fits")

print(outname_list)
#for _ in tqdm(pool.starmap(createVisSim, zip(outname_list)), total=len(outname_list)):
#    pass
#pool.terminate()    

masked_list = []       

for i in tqdm(range(nsimul)):
    print(outname_list[i])
    masked_name = outname_list[i].replace(".fits","_mask.fits")
    if not os.path.exists(outname_list[i]):
        createVisSim(outname_list[i])
    else:
        #print("Jumping: " + outname_list[i])
        os.system("rm " + outname_list[i])
        createVisSim(outname_list[i])

    #if not os.path.exists(masked_name):
    #    tars.mask_fits(outname_list[i], 0)
    #else:
    #    print("Jumping: " + outname_list[i])
    #masked_list.append(masked_name)
    #tars.normalize_frame([masked_name],1)                                      

outname_final = saveDir + "test_masked_" + version + ".fits"

tars.bootima(fits_list=outname_list, ext=0, nsimul=1, outname=outname_final, clean=True, verbose=False, mode="median")
#tars.bootima(fits_list=masked_list, ext=1, nsimul=3, outname="test_masked_v0.fits", clean=True, verbose=False,mode="median")                                                                                                                             
tars.normalize_frame([saveDir + "test_masked_"+version+".fits"],1)                                                                                                                              
tars.execute_cmd(cmd_text="astarithmetic " + outname_final + " " + saveDir + "/flat_res_ccd00.fits -h1 -h1 /", verbose=False)
