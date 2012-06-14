"""
Generating Objects
==================

This script provides a class that can be used to generate objects such as galaxies.

:requires: PyRAF
:requires: PyFITS
:requires: NumPy

:author: Sami-Matias Niemi
:contact: smn2@mssl.ucl.ac.uk

:version: 0.1
"""
from pyraf import iraf
from iraf import artdata
import numpy as np
import pyfits as pf
from support import logger as lg
import os, datetime


class generateFakeData():
    """

    """
    def __init__(self, log, **kwargs):
        """

        """
        self.log = log
        self.settings = dict(dynrange=1.e5,
                             gain=3.5,
                             magzero=25.58,
                             exptime=565.0,
                             rdnoise=4.5,
                             background=97.0,
                             xdim=4096,
                             ydim=4132,
                             star='gaussian',
                             beta=2.5,
                             radius=0.18,
                             ar=1.0,
                             pa=0.0,
                             poisson=iraf.yes,
                             egalmix=0.4,
                             output='image.fits')
        self.settings.update(kwargs)
        #self._createEmptyImage()


    def _createEmptyImage(self, unsigned16bit=True):
        """

        :param unsigned16bit: whether to scale the data using bzero=32768
        :type unsigned16bit: bool
        """
        self.image = np.zeros((self.settings['ydim'], self.settings['xdim']))

        if os.path.isfile(self.settings['output']):
            os.remove(self.settings['output'])

        #create a new FITS file, using HDUList instance
        ofd = pf.HDUList(pf.PrimaryHDU())

        #new image HDU
        hdu = pf.ImageHDU(data=self.image)

        #convert to unsigned 16bit int if requested
        if unsigned16bit:
            hdu.scale('int16', '', bzero=32768)
            hdu.header.add_history('Scaled to unsigned 16bit integer!')

        #update and verify the header
        hdu.header.add_history('If questions, please contact Sami-Matias Niemi (smn2 at mssl.ucl.ac.uk).')
        hdu.header.add_history('This file has been created with the VISsim Python Package at %s' % datetime.datetime.isoformat(datetime.datetime.now()))
        hdu.verify('fix')

        ofd.append(hdu)

        #write the actual file
        ofd.writeto(self.settings['output'])
        self.log.info('Wrote %s' % self.settings['output'])



    def createStarlist(self, nstars=20, output='stars.dat'):
        """
        Generates an ascii file with uniform random x and y positions.
        The magnitudes of stars are taken from an isotropic and homogeneous power-law distribution.

        :param nstars: number of stars to include
        :type nstars: int
        :param output: name of the output ascii file
        :type output: str
        """
        self.log.info('Generating a list of stars; including %i stars to %s' % (nstars, output))
        if os.path.isfile(output):
            os.remove(output)
        iraf.starlist(output, nstars, xmax=self.settings['xdim'], ymax=self.settings['ydim'],
                      minmag=-1, maxmag=10)


    def createGalaxylist(self, ngalaxies=50, output='galaxies.dat'):
        """
        Generates an ascii file with uniform random x and y positions.
        The magnitudes of galaxies are taken from an isotropic and homogeneous power-law distribution.

        :param ngalaxies: number of galaxies to include
        :type ngalaxies: int
        :param output: name of the output ascii file
        :type output: str
        """
        self.log.info('Generating a list of galaxies; including %i galaxies to %s' % (ngalaxies, output))
        if os.path.isfile(output):
            os.remove(output)
        iraf.gallist(output, ngalaxies, xmax=self.settings['xdim'], ymax=self.settings['ydim'],
                     egalmix=self.settings['egalmix'], maxmag=15.0, minmag=7)


    def addObjects(self, inputlist='galaxies.dat'):
        """
        Add objects from inputlist to the output image.

        :param inputlist: name of the input list
        :type inputlist: str

        """
        self.log.info('Adding objects from %s' % inputlist)
        iraf.artdata.dynrange = self.settings['dynrange']
        iraf.mkobjects(self.settings['output'],
                       output='',
                        ncols=self.settings['xdim'],
                        nlines=self.settings['ydim'],
                        background=self.settings['background'],
                        objects=inputlist,
                        xoffset=0.0,
                        yoffset=0.0,
                        star=self.settings['star'],
                        radius=self.settings['radius'],
                        beta=self.settings['beta'],
                        ar=self.settings['ar'],
                        pa=self.settings['pa'],
                        distance=1.0,
                        exptime=self.settings['exptime'],
                        magzero=self.settings['magzero'],
                        gain=self.settings['gain'],
                        rdnoise=self.settings['rdnoise'],
                        poisson=self.settings['poisson'],
                        seed=2,
                        comments=iraf.yes)


    def runAll(self):
        """

        """
        self.createStarlist()
        self.createGalaxylist()
        self.addObjects(inputlist='stars.dat')
        self.addObjects()

        for key, value in self.settings.iteritems():
            self.log.info('%s = %s' % (key, value))



if __name__ == '__main__':
    log = lg.setUpLogger('generateGalaxies.log')
    log.info('Starting to create fake galaxies')

    fakedata = generateFakeData(log)
    fakedata.runAll()

    #no noise or background
    settings = dict(rdnoise=0.0, background=0.0, output='nonoise.fits', poisson=iraf.no)
    fakedata = generateFakeData(log, **settings)
    fakedata.runAll()

    log.info('All done...\n\n\n')