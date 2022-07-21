from __future__ import print_function, division

import sys
from astroML.py3k_compat import HTTPError
import numpy as np
from astroML.datasets import fetch_sdss_spectrum
from astroML.dimensionality import iterative_pca
import pandas as pd
from linetools.spectra.xspectrum1d import XSpectrum1D
import astropy.units as units
from astroquery.sdss import SDSS
from astropy import coordinates as coords
from scipy.interpolate import interp1d
import scipy

import os

from astroML.datasets.tools import get_data_home, download_with_progress_bar,\
    SDSSfits, sdss_fits_url, sdss_fits_filename

#BASED ON CODE FROM https://github.com/stephenportillo/SDSS-VAE/blob/master/compute_sdss_pca.py


def fetch_sdss_spectrum1(plate, mjd, fiber, data_home="/cosma5/data/durham/dc-will10/astroMLdata",
                        download_if_missing=True,
                        cache_to_disk=True):
    """Fetch an SDSS spectrum from the Data Archive Server
    Parameters
    ----------
    plate: integer
        plate number of desired spectrum
    mjd: integer
        mean julian date of desired spectrum
    fiber: integer
        fiber number of desired spectrum
    Other Parameters
    ----------------
    data_home: string (optional)
        directory in which to cache downloaded fits files.  If not
        specified, it will be set to ~/astroML_data.
    download_if_missing: boolean (default = True)
        download the fits file if it is not cached locally.
    cache_to_disk: boolean (default = True)
        cache downloaded file to data_home.
    Returns
    -------
    spec: :class:`astroML.tools.SDSSfits` object
        An object wrapper for the fits data
    """
    data_home = get_data_home(data_home)

    target_url = sdss_fits_url(plate, mjd, fiber)
    target_file = os.path.join(data_home, 'SDSSspec', '%04i' % plate,
                               sdss_fits_filename(plate, mjd, fiber))

    if not os.path.exists(target_file):
        if not download_if_missing:
            raise IOError("SDSS colors training data not found")

        buf = download_with_progress_bar(target_url, return_buffer=True)

        if cache_to_disk:
            print("caching to %s" % target_file)
            if not os.path.exists(os.path.dirname(target_file)):
                os.makedirs(os.path.dirname(target_file))
            fhandler = open(target_file, 'wb')
            fhandler.write(buf.read())
            buf.seek(0)
    else:
        buf = target_file

    return SDSSfits(buf), target_file


data1 = pd.read_csv("/cosma5/data/durham/dc-will10/FullGalList.csv")

def fetch_and_shift_spectra(n_spectra,
                            outfile,
                            wavemin,
                            wavemax,
                            newvpix,
                            zlim=(0, 0.7)):
    """
    This function queries CAS for matching spectra, and then downloads
    them
    """
    # First set up the new wavelength bins for the spectra
    wavemin, wavemax = wavemin, wavemax
    newvpix = newvpix
    npix = np.log10(wavemax/wavemin) / np.log10(1.0 + newvpix/299792.458)
    npix = np.int(npix)
    newwave = wavemin*(1.0+newvpix/299792.458)**np.arange(npix)
    Nlam = len(newwave)
    
    plate = data1["plate"]
    mjd = data1["mjd"]
    fiber = data1["fiberid"]
    ra = data1["ra"]
    dec = data1["dec"]
    #objid = data1["objid"]

    # Set up arrays to hold information gathered from the spectra
    spec_cln = np.zeros(n_spectra, dtype=np.int32)
    lineindex_cln = np.zeros(n_spectra, dtype=np.int32)

    log_NII_Ha = np.zeros(n_spectra, dtype=np.float32)
    log_OIII_Hb = np.zeros(n_spectra, dtype=np.float32)

    z = np.zeros(n_spectra, dtype=np.float32)
    zerr = np.zeros(n_spectra, dtype=np.float32)
    spectra = np.zeros((n_spectra, Nlam), dtype=np.float32)
    mask = np.zeros((n_spectra, Nlam), dtype=np.bool)
    specerr = np.zeros((n_spectra, Nlam), dtype=np.float32)

    # also save plate, mjd, fiber, ra, dec to allow reference to SDSS data
    plates = np.zeros(n_spectra, dtype=np.int32)
    mjds = np.zeros(n_spectra, dtype=np.int32)
    fibers = np.zeros(n_spectra, dtype=np.int32)
    ras = np.zeros(n_spectra, dtype = np.float32)
    decs = np.zeros(n_spectra, dtype = np.float32)
    objids = np.zeros(n_spectra, dtype = np.int64)
    normfactors = []

    # Now download all the needed spectra, and resample to a common
    #  wavelength bin.
    #n_spectra = len(plate)
    num_skipped = 0
    # changed counter and loop so that skipped spectra do not create gaps in arrays
    j = 0
  

    for i in range(n_spectra):
        sys.stdout.write(' %i / %i spectra\r' % (i + 1, n_spectra))
        sys.stdout.flush()
        try:
            spec, fileext = fetch_sdss_spectrum1(plate[i], mjd[i], fiber[i], data_home="/cosma5/data/durham/dc-will10/astroMLdata")
            spec1 = XSpectrum1D.from_file(fileext) 
            xid = SDSS.query_specobj(plate = plate[i], mjd = mjd[i], fiberID = fiber[i], cache = False)
        except HTTPError:
            num_skipped += 1
            print("%i, %i, %i not found" % (plate[i], mjd[i], fiber[i]))
            continue

        #Series of error checks

        try:    
            newspec = spec1.rebin(newwave*units.AA,do_sig = True, grow_bad_sig=True)
        except ValueError:
            num_skipped += 1
            print("Spectrum did not have good pixels")
            continue
            
        try:
            objids[j] = int(xid["objid"].data)

        except KeyError:
            num_skipped += 1
            print("No available Object ID")

        if spec.z < zlim[0] or spec.z > zlim[1]:
            num_skipped += 1
            print("%i, %i, %i outside redshift range" % (plate[i], mjd[i], fiber[i]))
            continue

        if np.all(newspec.flux == 0):
            num_skipped += 1
            print("%i, %i, %i is all zero" % (plate[i], mjd[i], fiber[i]))
            continue

        if spec.spec_cln < 2 or spec.spec_cln > 3:
            num_skipped += 1
            print("%i, %i, %i is not a galaxy spectrum" % (plate[i], mjd[i], fiber[i]))
            continue

        spec_cln[j] = spec.spec_cln

        lineindex_cln[j], (log_NII_Ha[j], log_OIII_Hb[j])\
            = spec.lineratio_index()

        z[j] = spec.z
        zerr[j] = spec.zerr

        spectra[j] = newspec.flux
        #mask[j] = newspec.compute_mask(0.5, 5)
        #assert((mask[j] == 0).any())
        specerr[j] = newspec.sig

        plates[j] = plate[i]
        mjds[j] = mjd[i]
        fibers[j] = fiber[i]
        ras[j] = ra[i]
        decs[j] = dec[i]
        objids[j] = int(xid["objid"].data)

        j += 1
    sys.stdout.write('\n')
    N = j
    #The following finds the skyline range of indices in the spectra and predicts then assigns new values using scipy interpolation.
    #This code then also normalises the spectra by L2 method to make the flux of all spectra comparable, then saves the norm factor
    #This factor is saved in order to allow for the spectra to be reverted back to their original fluxes later.
    for i in range(N):
        origspec = spectra[i]
        origerrs = specerr[i]
        #testspec = np.delete(spectra[i], [766,767,768,769,770])
        #testerrs = np.delete(specerr[i], [766,767,768,769,770])
        #lmbtest = np.delete(newwave, [766,767,768,769,770])
        #f = interp1d(lmbtest, testspec)
        #g = interp1d(lmbtest, testerrs)
        #origspec = f(newwave)
        #besterrs = g(newwave)
        contspec = scipy.ndimage.median_filter(origspec, size=5)
        conterrs = scipy.ndimage.median_filter(specerr[i], size=5)
        for n in range(766, 771):
            origspec[n] = contspec[n]
            origerrs[n] = conterrs[n]
            
        norm = np.sqrt(np.sum(spectra[i]**2))
        normfactors.append(norm)
        spectra[i] = (origspec)
        specerr[i] = origerrs

    
    print("   %i spectra skipped" % num_skipped)
    print("   %i spectra processed" % N)
    print("saving to %s" % outfile)
    print(i)

    np.savez(outfile,
             spectra=spectra[:N],
             norms = np.array(normfactors),
             #mask=mask[:N],
             spec_err=specerr[:N],
             wavelengths = newwave,
             spec_cln=spec_cln[:N],
             lineindex_cln=lineindex_cln[:N],
             log_NII_Ha=log_NII_Ha[:N],
             log_OIII_Hb=log_OIII_Hb[:N],
             z=z[:N],
             zerr=zerr[:N],
             plate=plates[:N],
             mjd=mjds[:N],
             fiber=fibers[:N],
             ra=ras[:N],
             dec = decs[:N],
             objid = objids[:N])
    

fetch_and_shift_spectra(110000, '/cosma5/data/durham/dc-will10/spec80new6.npz', 3800, 9200, 150, zlim = (0.05,0.1))