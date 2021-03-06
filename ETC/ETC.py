"""
Calculating Exposure Times and Limiting Magnitude
=================================================

This file provides a simple functions to calculate exposure times or limiting magnitudes.

:requires: NumPy
:requires: SciPy
:requires: matplotlib

:version: 0.4

:author: Sami-Matias Niemi
:contact: s.niemi@ucl.ac.uk
"""
import matplotlib
matplotlib.rc('text', usetex=True)
matplotlib.rcParams['font.size'] = 17
matplotlib.rc('xtick', labelsize=14)
matplotlib.rc('axes', linewidth=1.1)
matplotlib.rcParams['legend.fontsize'] = 11
matplotlib.rcParams['legend.handlelength'] = 3
matplotlib.rcParams['xtick.major.size'] = 5
matplotlib.rcParams['ytick.major.size'] = 5
import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import math, datetime
from support.VISinstrumentModel import VISinformation


def exposureTime(info, magnitude, snr=10.0, exposures=3, fudge=0.7, galaxy=True, diginoise=False):
    """
    Returns the exposure time for a given magnitude.

    :param info: information describing the instrument
    :type info: dict
    :param magnitude: the magnitude of the object
    :type magnitude: float or ndarray
    :param snr: signal-to-noise ratio required [default = 10.0].
    :type snr: float
    :param exposures: number of exposures that the object is present in [default = 3]
    :type exposures: int
    :param fudge: the fudge parameter to which to use to scale the snr to SExtractor required [default = 0.7]
    :type fudge: float
    :param galaxy: whether the exposure time should be calculated for an average galaxy or a star.
                   If galaxy=True then the fraction of flux within an aperture is lower than in case of a point source.
    :type galaxy: boolean
    :param diginoise: if the readout noise is undersampled or poorly resolved then the effective readout noise
                      should be used [default = False]
    :type diginoise: boolean

    :return: exposure time [seconds]
    :rtype: float or ndarray
    """
    snr /= fudge

    if galaxy:
        flux_in_aperture = 10**(-0.4*(magnitude - info['zeropoint'])) * info['galaxy_fraction']
    else:
        flux_in_aperture = 10**(-0.4*(magnitude - info['zeropoint'])) * info['star_fraction']

    sky = 10**(-0.4*(info['sky_background'] - info['zeropoint'])) * (info['pixel_size']**2)
    zodiacal = 10**(-0.4*(info['zodiacal'] - info['zeropoint'])) * (info['pixel_size']**2)
    instrument = 0.2 * zodiacal  #20% of zodiacal background

    tmp = flux_in_aperture + (sky + instrument + info['dark'])*info['aperture_size']

    first = snr**2 * tmp * exposures

    if diginoise:
        root = np.sqrt(first**2 + (2*flux_in_aperture*exposures*snr*(info['readnoise']**2 +
                                                                     (info['gain']/2.)**2) * info['aperture_size']))
    else:
        root = np.sqrt(first**2 + (2*flux_in_aperture*exposures*snr*info['readnoise'])**2 *
                                  exposures*info['aperture_size'])

    div = 2*flux_in_aperture**2*exposures**2

    return (first + root)/div


def limitingMagnitude(info, exp=565, snr=10.0, exposures=3, fudge=0.7, galaxy=True, diginoise=False):
    """
    Calculates the limiting magnitude for a given exposure time, number of exposures and minimum signal-to-noise
    level to be reached.

    :param info: instrumental information such as zeropoint and background
    :type info: dict
    :param exp: exposure time [seconds]
    :type exp: float or ndarray
    :param snr: the minimum signal-to-noise ratio to be reached.
                .. Note:: This is couple to the fudge parameter: snr_use = snr / fudge
    :type snr: float or ndarray
    :param exposures: number of exposures [default = 3]
    :type exposures: int
    :param fudge: a fudge factor to divide the given signal-to-noise ratio with to reach to the required snr.
                  This is mostly due to the fact that SExtractor may require a higher snr than what calculated
                  otherwise.
    :type fudge: float
    :param galaxy: whether the exposure time should be calculated for an average galaxy or a star.
                   If galaxy=True then the fraction of flux within an aperture is lower than in case of a point source.
    :type galaxy: boolean
    :param diginoise: if the readout noise is undersampled or poorly resolved then the effective readout noise
                      should be used [default = False]
    :type diginoise: boolean

    :return: limiting magnitude given the input information
    :rtype: float or ndarray
    """
    snr /= fudge
    totalexp = exposures*exp

    sky = 10**(-0.4*(info['sky_background'] - info['zeropoint'])) * (info['pixel_size']**2)
    zodiacal = 10**(-0.4*(info['zodiacal'] - info['zeropoint'])) * (info['pixel_size']**2)
    instrument = 0.2 * zodiacal  #20% of zodiacal background

    if diginoise:
        tmp = 4*((sky+instrument+info['dark']) * info['aperture_size'] * totalexp + exposures *
                 (info['readnoise']**2 + (info['gain']/2.)**2 ) * info['aperture_size'])
    else:
        tmp = 4*((sky+instrument+info['dark']) * info['aperture_size'] * totalexp + exposures *
                  info['readnoise']**2 * info['aperture_size'])

    root = np.sqrt(snr**2 + tmp)

    if galaxy:
        lg = 2.5*np.log10(((0.5*(snr**2 + snr*root))/totalexp)/info['galaxy_fraction'])
    else:
        lg = 2.5*np.log10(((0.5*(snr**2 + snr*root))/totalexp)/info['star_fraction'])

    out = info['zeropoint'] - lg

    return out


def SNR(info, magnitude=24.5, exptime=565.0, exposures=3, galaxy=True, background=True, diginoise=False):
    """
    Calculates the signal-to-noise ratio for an object of a given magnitude in a given exposure time and a
    number of exposures.

    :param info: instrumental information such as zeropoint and background
    :type info: dict
    :param magnitude: input magnitude of an object(s)
    :type magnitude: float or ndarray
    :param exptime: exposure time [seconds]
    :type exptime: float
    :param exposures: number of exposures [default = 3]
    :type exposures: int
    :param galaxy: whether the exposure time should be calculated for an average galaxy or a star.
                   If galaxy=True then the fraction of flux within an aperture is lower than in case of a point source.
    :type galaxy: boolean
    :param background: whether to include background from sky, instrument, and dark current [default=True]
    :type background: boolean
    :param diginoise: if the readout noise is undersampled or poorly resolved then the effective readout noise
                      should be used [default = False]
    :type diginoise: boolean

    :return: signal-to-noise ratio
    :rtype: float or ndarray
    """
    if galaxy:
        flux_in_aperture = 10**(-0.4*(magnitude - info['zeropoint'])) * info['galaxy_fraction']
    else:
        flux_in_aperture = 10**(-0.4*(magnitude - info['zeropoint'])) * info['star_fraction']

    sky = 10**(-0.4*(info['sky_background'] - info['zeropoint'])) * (info['pixel_size']**2)
    zodiacal = 10**(-0.4*(info['zodiacal'] - info['zeropoint'])) * (info['pixel_size']**2)
    instrument = 0.2 * zodiacal  #20% of zodiacal background
    bgr = (sky + instrument + info['dark']) * info['aperture_size'] * exptime

    #print bgr / info['aperture_size']
    #print info['zeropoint'], sky, zodiacal, instrument, bgr

    if not background:
        bgr = 0.0

    nom = flux_in_aperture * exptime
    if diginoise:
        denom = np.sqrt(nom + bgr + (info['readnoise']**2 + (info['gain']/2.)**2) * info['aperture_size'])
    else:
        denom = np.sqrt(nom + bgr + info['readnoise']**2 * info['aperture_size'])

    return nom / denom * np.sqrt(exposures)


def SNRproptoPeak(info, exptime=565.0, exposures=1, diginoise=False, server=False):
    """
    Calculates the relation between the signal-to-noise ratio and the electrons in the peak pixel.

    :param info: instrumental information such as zeropoint and background
    :type info: dict
    :param exptime: exposure time [seconds]
    :type exptime: float
    :param exposures: number of exposures [default = 1]
    :type exposures: int
    :param diginoise: if the readout noise is undersampled or poorly resolved then the effective readout noise
                      should be used [default = False]
    :type diginoise: bool
    :param server: whether to save the figure in a PNG or PDF format (the former can be used together with the WWW server)
    :type server: bool

    :return: signal-to-noise ratio
    :rtype: float or ndarray
    """
    sky = 10**(-0.4*(info['sky_background'] - info['zeropoint'])) * (info['pixel_size']**2)
    zodiacal = 10**(-0.4*(info['zodiacal'] - info['zeropoint'])) * (info['pixel_size']**2)
    instrument = 0.2 * zodiacal  #20% of zodiacal background
    bgr = (sky + instrument + info['dark']) * info['aperture_size'] * exptime

    snr = []
    peak = []
    for magnitude in np.arange(10, 27, 1)[::-1]:
        nom = 10**(-0.4*(magnitude - info['zeropoint'])) * info['star_fraction'] * exptime
        peak_pixel = nom * info['peak_fraction']

        if diginoise:
            denom = np.sqrt(nom + bgr + (info['readnoise']**2 + (info['gain']/2.)**2) * info['aperture_size'])
        else:
            denom = np.sqrt(nom + bgr + (info['readnoise']**2 * info['aperture_size']))

        result = nom / denom * np.sqrt(exposures)
        snr.append(result)
        peak.append(peak_pixel)

    peak = np.asarray(peak)
    snr = np.asanyarray(snr)

    f = interpolate.interp1d(snr, peak, kind='cubic')
    sn = np.asarray([5, 10, 50, 100, 200, 500, 1000])
    vals = f(sn)

    txt = '%s' % datetime.datetime.isoformat(datetime.datetime.now())
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('VIS Nominal System PSF: Peak Pixel')
    plt.text(0.83, 1.12, txt, ha='left', va='top', fontsize=9, transform=ax.transAxes, alpha=0.2)

    ax.loglog(peak, snr, 'bo-', label=r'SNR $\propto$ counts')

    ax.axvline(x=220000, c='r', ls='--', label=r'Saturation $> 220000 e^{-}$')
    ax.fill_between(np.linspace(220000, np.max(peak), 3), 1e5, 1.0, facecolor='red', alpha=0.08, label=r'Saturation $> 220000 e^{-}$')

    i = 0
    for val, s in zip(vals, sn):
        plt.text(0.05, 0.9-i, r'snr = %i, peak = %.1f $e^{-}$' % (s, val), fontsize=10, transform=ax.transAxes)
        i += 0.03

    ax.set_ylabel('Signal-to-Noise Ratio [nominal 565s]')
    ax.set_xlabel(r'Counts in the Peak Pixel $[e^{-}]$')

    ax.set_xlim(xmax=np.max(peak))

    plt.legend(shadow=True, fancybox=True, loc='lower right')
    if server:
        plt.savefig('delete.png')
    else:
        plt.savefig('Peak.pdf')
    plt.close()


def galaxyDetection(info, magnitude=24.5, exposures=3, exptime=565):
    """
    Can be used to study the ghost contribution to the galaxy detection.

    :param info:
    :param magnitude: object detection magnitude limit
    :param exposures: number of exposures to be combined
    :param exptime: individual exposure time

    :return: number of electrons tolerated in the ghost
    """
    req1 = 10./0.7
    req2 = 10.

    print info['zeropoint'], info['galaxy_fraction'], info['sky_background'], info['sky_low']
    print info['pixel_size'], info['dark'], info['readnoise']

    flux_in_aperture = 10**(-0.4*(magnitude - info['zeropoint'])) * info['galaxy_fraction']

    sky = 10**(-0.4*(info['sky_background'] - info['zeropoint'])) * (info['pixel_size']**2)
    instrument = 0.2 * 10**(-0.4*(info['sky_low'] - info['zeropoint'])) * (info['pixel_size']**2)
    print instrument
    ghostlevels = np.linspace(0, 200., num=20)

    results = []
    for ghost in ghostlevels:
        bgr = (sky + instrument + info['dark']) * info['aperture_size'] * exptime
        ghostContribution = ghost * info['aperture_size']
        bgr += ghostContribution

        nom = flux_in_aperture * exptime
        denom = np.sqrt(nom + bgr + info['readnoise']**2 * info['aperture_size'])

        snr = nom / denom * np.sqrt(exposures)
        results.append(snr)

    #find the crossing
    results = np.asarray(results)
    f = interpolate.interp1d(ghostlevels, results, kind='cubic')
    x = np.linspace(ghostlevels.min(), ghostlevels.max(), num=1000)
    vals = f(x)
    msk1 = vals >= req1
    msk2 = vals >= req2
    maxn1 = np.max(x[msk1])
    maxn2 = np.max(x[msk2])
    print 'Ghost requirements:'
    print maxn1, maxn2

    #plot number of electrons vs SNR
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title('Impact of Ghost Image to Galaxy Detection')

    ax.plot(ghostlevels, results, 'bo', label=r'SNR within the Aperture')
    ax.plot(x, vals, '-', c='b')

    ax.axhline(y=req1, c='r', ls='--', label=r'Requirement')
    ax.axhline(y=req2, c='g', ls='-.')
    #ax.axhline(y=req2, c='r', ls='-.', label=r'No Margin')

    plt.text(100, req1-0.2, r'To meet the SNR requirement ghost contribution must be $\leq %.2f e^{-}$ per pixel' % (maxn1),
             fontsize=11, ha='center', va='center')

    ax.set_ylabel(r'Signal-to-Noise Ratio')
    ax.set_xlabel(r'Ghost Contribution $[e^{-}]$')

    ax.set_xlim(-0.01, 200.01)

    plt.legend(shadow=True, fancybox=True, loc='best', numpoints=1)
    plt.savefig('ObjectDetection.pdf')
    plt.close()

    return maxn1, maxn2


if __name__ == '__main__':
    magnitude = 24.5
    exptime = 565.0

    info = VISinformation()
    print 'VIS zero point = %f' % info['zeropoint']

    exp = exposureTime(info, magnitude, exposures=1)
    limit = limitingMagnitude(info, exp=exptime)
    snrS = SNR(info, magnitude=magnitude, exptime=exptime, exposures=1, galaxy=False)
    snrG = SNR(info, magnitude=magnitude, exptime=exptime, exposures=3)

    print 'Exposure time required to reach SNR=10/14.29 for a %.2f magnitude galaxy is %.1f' % (magnitude, exp)
    print 'SNR=%f for %.2fmag star if exposure time is %.2f seconds' % (snrS, magnitude, exptime)
    print 'SNR=%f for %.2fmag galaxy if three exposures of %.2f seconds' % (snrG, magnitude, exptime)
    print 'Limiting magnitude of a galaxy in three %.2f second exposures is %.2f' % (exptime, limit)

    #SNRproptoPeak(info)

    galaxyDetection(info, magnitude=magnitude, exptime=exptime)
