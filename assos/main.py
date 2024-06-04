import tdpy
from tdpy.util import summgene

import h5py


import time as timemodu

import scipy.signal
from scipy import interpolate

import os, sys, datetime, fnmatch

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#from astroquery.mast import Catalogs
#import astroquery
#
#import astropy
#from astropy.wcs import WCS
#from astropy import units as u
#from astropy.io import fits
#import astropy.time
#from astropy.coordinates import SkyCoord

import numpy as np
import scipy

import time as modutime

import astropy
import astropy.convolution

import tdpy
import aspendos
import chalcedon
import nicomedia
from tdpy import summgene



'''
Library to forward-model imaging data
'''

def plot_anim(gdat, cntp, strgvarb, cmap='Greys_r', strgtitlbase='', boolresi=False, indxsideyposoffs=0, indxsidexposoffs=0):
    
    vmin = np.amin(cntp)
    vmax = np.amax(cntp)
    if boolresi:
        vmax = max(abs(vmax), abs(vmin))
        vmin = -vmax
    
    for t in gdat.indxtime:
        strgtitl = strgtitlbase + ', JD = %d' % gdat.time[t]
        path = gdat.pathdata + '%s_%s_%05d.pdf' % (strgvarb, gdat.strgcntp, t)
        plot_imag(gdat, cntp[:, :, t], path=path, strgvarb=strgvarb, cmap=cmap, strgtitl=strgtitl, \
                                        indxsideyposoffs=indxsideyposoffs, indxsidexposoffs=indxsidexposoffs, boolresi=boolresi, vmin=vmin, vmax=vmax)
    os.system('convert -density 300 -delay 10 %s%s_%s_*.pdf %s%s_%s.gif' % (gdat.pathdata, strgvarb, gdat.strgcntp, gdat.pathdata, strgvarb, gdat.strgcntp))
    ### delete the frame plots
    path = gdat.pathdata + '%s_%s_*.pdf' % (strgvarb, gdat.strgcntp)
    #os.system('rm %s' % path)


def plot_imag(gdat, cntp, strgvarb, path=None, cmap=None, indxsideyposoffs=0, indxsidexposoffs=0, \
                    strgtitl='', boolresi=False, xposoffs=None, yposoffs=None, indxpixlcolr=None, vmin=None, vmax=None):
    
    if cmap == None:
        if boolresi:
            cmap = 'RdBu'
        else:
            cmap = 'Greys_r'
    
    if vmin is None or vmax is None:
        vmax = np.amax(cntp)
        vmin = np.amin(cntp)
        if boolresi:
            vmax = max(abs(vmax), abs(vmin))
        vmin = -vmax
    
    if gdat.cntpscaltype == 'asnh':
        cntp = np.arcsinh(cntp)
        vmin = np.arcsinh(vmin)
        vmax = np.arcsinh(vmax)

    figr, axis = plt.subplots(figsize=(8, 6))
    objtimag = axis.imshow(cntp, origin='lower', interpolation='nearest', cmap=cmap, vmin=vmin, vmax=vmax)
    
    if indxpixlcolr is not None:
        temp = np.zeros_like(cntp).flatten()
        temp[indxpixlcolr[-1]] = 1.
        temp = temp.reshape((gdat.numbside, gdat.numbside))
        alph = np.zeros_like(cntp).flatten()
        alph[indxpixlcolr[-1]] = 1.
        alph = alph.reshape((gdat.numbside, gdat.numbside))
        alph = np.copy(temp)
        axis.imshow(temp, origin='lower', interpolation='nearest', alpha=0.5)
    
    # overplot catalog
    plot_catl(gdat, axis, indxsideyposoffs=indxsideyposoffs, indxsidexposoffs=indxsidexposoffs)
    
    # make color bar
    cax = figr.add_axes([0.83, 0.1, 0.03, 0.8])
    cbar = figr.colorbar(objtimag, cax=cax)
    if gdat.cntpscaltype == 'asnh':
        tick = cbar.get_ticks()
        tick = np.sinh(tick)
        labl = ['%d' % int(tick[k]) for k in range(len(tick))]
        cbar.set_ticklabels(labl)

    if path is None:
        path = gdat.pathimag + '%s_%s.pdf' % (strgvarb, gdat.strgcntp)
    print('Writing to %s...' % path)
    #plt.tight_layout()
    plt.savefig(path)
    plt.close()
    

def init( \
         listisec=None, \
         listicam=None, \
         listiccd=None, \
         extrtype='qlop', \
         targtype='slen', \
         pathfile=None, \
         datatype='obsd', \
         rasctarg=None, \
         decltarg=None, \
         labltarg=None, \
         strgtarg=None, \
         numbside=None, \
         **args \
        ):

    
    # inputs:
    # 1) TIC IDs
    # 2) One sector, One Cam, One CCD

    # preliminary setup
    # construct the global object 
    gdat = tdpy.util.gdatstrt()
    #for attr, valu in locals().iteritems():
    #    if '__' not in attr and attr != 'gdat':
    #        setattr(gdat, attr, valu)
    
    # copy all provided inputs to the global object
    #for strg, valu in args.iteritems():
    #    setattr(gdat, strg, valu)
    
    gdat.datatype = datatype

    # string for date and time
    gdat.strgtimestmp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    print('Assos initialized at %s...' % gdat.strgtimestmp)
    
    #if ((listicam is not None or listiccd is not None) and listtici is not None):
    #    raise Exception('')

    gdat.strgcntp = gdat.datatype

    # paths
    ## read PCAT path environment variable
    gdat.pathbase = os.environ['ASSOS_DATA_PATH'] + '/'
    gdat.pathdata = gdat.pathbase + 'data/'
    gdat.pathimag = gdat.pathbase + 'imag/'
    ## define paths
    #gdat.pathdataorig = '/pdo/qlp-data/orbit-%d/ffi/cam%d/ccd%d/FITS/' % (isec, icam, iccd)
    gdat.pathdataorig = gdat.pathdata + 'ffis/'
    gdat.pathdatafilt = gdat.pathdata + 'filt/'
    gdat.pathdatainit = gdat.pathdata + 'init/'
    gdat.pathdatainitimag = gdat.pathdatainit + 'imag/'
    gdat.pathdatainitanim = gdat.pathdatainit + 'anim/'
    gdat.pathdatacomm = gdat.pathdata + 'comm/'
    ## make folders 
    os.system('mkdir -p %s' % gdat.pathdatafilt)
    
    gdat.numbsidefilt = 21

    gdat.numbside = 2048
    gdat.numbpixl = gdat.numbside**2
    gdat.numbtime = 100
    gdat.indxtime = np.arange(gdat.numbtime)
    gdat.numbdata = gdat.numbtime * gdat.numbpixl
    
    gdat.factdown = 8
    gdat.numbsidedown = gdat.numbside / gdat.factdown
    if pathfile is None:
        if numbside is None:
            strgmode = 'full'
        else:
            strgmode = 'targ'
    else:
        strgmode = 'file'
        
    random_state = 42

    timeexpo = 1426.
    
    if strgmode == 'targ':
        from astroquery.mast import Tesscut
        from astropy.coordinates import SkyCoord
        cutout_coord = SkyCoord(rasctarg, decltarg, unit="deg")
        listhdundata = Tesscut.get_cutouts(cutout_coord, gdat.numbside)
        sector_table = Tesscut.get_sectors(SkyCoord(gdat.rasctarg, gdat.decltarg, unit="deg"))
        listisec = sector_table['sector'].data
        listicam = sector_table['camera'].data
        listiccd = sector_table['ccd'].data
    
        if len(listhdundata) == 0:
            raise Exception('TESSCut could not find any data.')
    
    arryseco = np.zeros((gdat.numbsidedown, gdat.numbsidedown, gdat.numbtime))
    for t in gdat.indxtime:
        
        # get FFI iimage
        #cntpimag = 
        cntpimag = np.random.randn(gdat.numbpixl).reshape((gdat.numbside, gdat.numbside))
      
        pathsave = ''
        if not os.path.exists(pathsave):
            # filter
            arrysecotemp = scipy.signal.medfilt(cntpimag, (gdat.numbsidefilt, gdat.numbsidefilt))
            
            # plot
            
            # down-sample
            arryseco[:, :, t] = np.mean(arrysecotemp.reshape((gdat.numbsidedown, gdat.factdown, gdat.numbsidedown, gdat.factdown)), (1, 3))
            # save filtered FFI
        else:
            pass
            # load filtered FFI
        
        raise Exception('')
        # plot



    numbsect = len(listisec)
    indxsect = np.arange(numbsect)
    for o in indxsect:

        # check inputs
        print('Sector: %d' % listisec[o])
        print('Camera: %d' % listicam[o])
        print('CCD: %d' % listiccd[o])
        
        isec = listisec[o]

        verbtype = 1
        
        np.random.seed(45)

        # fix the seed
        if gdat.datatype == 'mock':
            gdat.numbsour = 1000
            numbsupn = 10
        
        gdat.numbtimerebn = None#30
        
        # settings
        ## plotting
        gdat.cntpscaltype = 'asnh'
        if pathfile is not None and gdat.datatype == 'mock':
            raise Exception('')
        
        # grid of flux space
        minmproj = 0.1
        maxmproj = 2
        limtproj = [minmproj, maxmproj]
        arry = np.linspace(minmproj, maxmproj, 100)
        xx, yy = np.meshgrid(arry, arry)
        
        magtminm = 12.
        magtmaxm = 19.
        
        # get data
        if gdat.datatype == 'obsd':
            print('Reading files...')
            path = gdat.pathdata + 'qlop/'
            liststrgfile = fnmatch.filter(os.listdir(path), '*.h5')
            liststrgfile = liststrgfile[:10000]
            
            liststrgtici = []
            for strgfile in liststrgfile:
                liststrgtici.append(strgfile[:-3])

            numbdata = len(liststrgfile)
            fracdatanann = np.empty(numbdata)
            listindxtimebadd = []
            for k, strgfile in enumerate(liststrgfile):
                with h5py.File(path + strgfile, 'r') as objtfile:
                    if k == 0:
                        gdat.time = objtfile['LightCurve/BJD'][()]
                        gdat.numbtime = gdat.time.size
                        lcur = np.empty((numbdata, gdat.numbtime, 2))
                    tmag = objtfile['LightCurve/AperturePhotometry/Aperture_002/RawMagnitude'][()]
                    if k == 0:
                        gdat.indxtime = np.arange(gdat.numbtime)
                    indxtimegood = np.where(np.isfinite(tmag))[0]
                    indxtimenann = np.setdiff1d(gdat.indxtime, indxtimegood)
                    lcur[k, :, 0] = 10**(-(tmag - np.median(tmag[indxtimegood])) / 2.5)
                    listindxtimebadd.append(indxtimenann)
                    fracdatanann[k] = indxtimenann.size / float(gdat.numbtime)

            listindxtimebadd = np.concatenate(listindxtimebadd)
            listindxtimebadd = np.unique(listindxtimebadd)
            listindxtimebadd = np.concatenate((listindxtimebadd, np.arange(100)))
            listindxtimebadd = np.concatenate((listindxtimebadd, gdat.numbtime / 2 + np.arange(100)))
            listindxtimegood = np.setdiff1d(gdat.indxtime, listindxtimebadd)
            #listhdundata = fits.open(pathfile)
            print('Filtering the data...')
            # filter the data
            gdat.time = gdat.time[listindxtimegood]
            lcur = lcur[:, listindxtimegood, :]
            gdat.numbtime = gdat.time.size
        
        if (~np.isfinite(lcur)).any():
            raise Exception('')

        # plot the data
        figr, axis = plt.subplots(figsize=(6, 4))
        axis.hist(fracdatanann)
        axis.set_xlabel('$f_{nan}$')
        axis.set_ylabel('$N(f_{nan})$')
        path = gdat.pathimag + 'histfracdatanann_%s.pdf' % (gdat.strgcntp)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()

        # inject signal
        if gdat.datatype == 'mock':
            gdat.numbtime = 50
            gdat.numbdata = 100
            gdat.time = np.arange(gdat.numbtime)
            lcur = np.random.randn(gdat.numbdata * gdat.numbtime).reshape((gdat.numbdata, gdat.numbtime, 1))
            gdat.numbtime = gdat.time.size
            gdat.indxsupn = np.arange(numbsupn)
            truecntpsour = np.empty((gdat.numbtime, gdat.numbsour))
            truemagt = np.empty((gdat.numbtime, gdat.numbsour))
            gdat.indxsour = np.arange(gdat.numbsour)
            gdat.indxsoursupn = np.random.choice(gdat.indxsour, size=numbsupn, replace=False)
            for n in gdat.indxsour:
                if n in gdat.indxsoursupn:
                    timenorm = -0.5 + (gdat.time / np.amax(gdat.time)) + 2. * (np.random.random(1) - 0.5)
                    objtrand = scipy.stats.skewnorm(10.).pdf(timenorm)
                    objtrand /= np.amax(objtrand)
                    truemagt[:, n] = 8. + 6. * (2. - objtrand)
                else:
                    truemagt[:, n] = np.random.rand() * 5 + 15.
                truecntpsour[:, n] = 10**((20.424 - truemagt[:, n]) / 2.5)
            gdat.truemagtmean = np.mean(truemagt, 0)
            gdat.truemagtstdv = np.std(truemagt, 0)

        numbdata = lcur.shape[0]
        if gdat.datatype == 'mock':
            listlabltrue = np.zeros(numbdata, dtype=int)
            numbinli = numbdata - gdat.numbsour
            numboutl = gdat.numbsour
        
        # plot the data
        figr, axis = plt.subplots(10, 4)
        indxdata = np.arange(numbdata)
        numbdataplot = min(40, numbdata)
        indxdataplot = np.random.choice(indxdata, size=numbdataplot, replace=False)
        for a in range(10):
            for b in range(4):
                p = a * 4 + b
                if p >= numbdata:
                    continue
                axis[a][b].plot(gdat.time, lcur[indxdataplot[p], :, 0], color='black', ls='', marker='o', markersize=1)
                if a != 9:
                    axis[a][b].set_xticks([])
                if b != 0:
                    axis[a][b].set_yticks([])
        plt.subplots_adjust(hspace=0, wspace=0)
        path = gdat.pathimag + 'lcurrand_%s.pdf' % (gdat.strgcntp)
        print('Writing to %s...' % path)
        plt.savefig(path)
        plt.close()

        # temporal median filter
        numbtimefilt = min(9, gdat.numbtime)
        if numbtimefilt % 2 == 0:
            numbtimefilt -= 1
        print('Performing the temporal median filter...')
        
        # rebin in time
        if gdat.numbtimerebn is not None and gdat.numbtime > gdat.numbtimerebn:
            print('Rebinning in time...')
            numbtimeoldd = gdat.numbtime
            gdat.numbtime = gdat.numbtimerebn
            numbtimebins = numbtimeoldd / gdat.numbtime
            cntpmemoneww = np.zeros((numbsidememo, numbsidememo, gdat.numbtime)) - 1.
            timeneww = np.zeros(gdat.numbtime)
            for t in range(gdat.numbtime):
                if t == gdat.numbtime - 1:
                    cntpmemoneww[:, :, t] = np.mean(cntpmemo[:, :, (gdat.numbtime-1)*numbtimebins:], axis=2)
                    timeneww[t] = np.mean(gdat.time[(gdat.numbtime-1)*numbtimebins:])
                else:
                    cntpmemoneww[:, :, t] = np.mean(cntpmemo[:, :, t*numbtimebins:(t+1)*numbtimebins], axis=2)
                    timeneww[t] = np.mean(gdat.time[t*numbtimebins:(t+1)*numbtimebins])
            gdat.indxtimegood = np.isfinite(timeneww)
            gdat.time = timeneww[gdat.indxtimegood]
            gdat.numbtime = gdat.indxtimegood.size
            gdat.indxtime = np.arange(gdat.numbtime)
        
        # calculate derived maps
        ## RMS image

        strgtype = 'tsne'

        lcuravgd = np.empty(gdat.numbtime)
        cntr = 0
        prevfrac = -1
        k = 0
        
        scorthrs = -2.

        # machine learning

        n_neighbors = 30
        
        X = lcur[:, :, 0]

        indxdata = np.arange(numbdata)
        
        outliers_fraction = 0.15
        
        # define outlier/anomaly detection methods to be compared
        listobjtalgoanom = [
                            #("One-Class SVM", svm.OneClassSVM(nu=outliers_fraction, kernel="rbf", gamma=0.1)), \
                            #("Isolation Forest", IsolationForest(contamination=outliers_fraction)), \
                            ("Local Outlier Factor", LocalOutlierFactor(n_neighbors=35, contamination=outliers_fraction))
                           ]
        
        numbmeth = len(listobjtalgoanom)
        indxmeth = np.arange(numbmeth)
        
        listindxsideyposaccp = []
        listindxsidexposaccp = []
        listscor = []
        listlablmodl = []
        
        numbtimeplotscat = min(6, gdat.numbtime)
        limt = [np.amin(X), np.amax(X)]
        
        c = 0
        print('Running anomaly-detection algorithms...')
        for name, objtalgoanom in listobjtalgoanom:
            t0 = timemodu.time()
            
            print('X')
            summgene(X)
            objtalgoanom.fit(X)
            t1 = timemodu.time()
        
            # fit the data and tag outliers
            if name == 'Local Outlier Factor':
                scor = objtalgoanom.negative_outlier_factor_
            else:
                scor = objtalgoanom.decision_function(X)
            if name == "Local Outlier Factor":
                lablmodl = np.zeros(numbdata)
                lablmodl[np.where(scor < scorthrs)[0]] = 1.
            else:
                lablmodl = objtalgoanom.fit(X).predict(X)
            
            indxdataposi = np.where(lablmodl == 1)[0]
            indxdatanega = np.setdiff1d(indxdata, indxdataposi)
            numbposi = indxdataposi.size
            gdat.numbpositext = min(200, numbposi)

            listscor.append(scor)
            listlablmodl.append(lablmodl)
            
            gdat.indxdatascorsort = np.argsort(listscor[c])

            # make plots
            ## labeled marginal distributions
            figr, axis = plt.subplots(numbtimeplotscat - 1, numbtimeplotscat - 1, figsize=(10, 10))
            for t in gdat.indxtime[:numbtimeplotscat-1]:
                for tt in gdat.indxtime[:numbtimeplotscat-1]:
                    if t < tt:
                        axis[t][tt].axis('off')
                        continue
                    axis[t][tt].scatter(X[indxdatanega, t+1], X[indxdatanega, tt], s=20, color='r', alpha=0.3)#*listscor[c])
                    axis[t][tt].scatter(X[indxdataposi, t+1], X[indxdataposi, tt], s=20, color='b', alpha=0.3)#*listscor[c])
                    axis[t][tt].set_ylim(limt)
                    axis[t][tt].set_xlim(limt)
                    #axis[t][tt].set_xticks(())
                    #axis[t][tt].set_yticks(())
            path = gdat.pathimag + 'pmar_%s_%04d.pdf'% (gdat.strgcntp, c)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
            
            # plot data with colors based on predicted class
            figr, axis = plt.subplots(10, 4)
            for a in range(10):
                for b in range(4):
                    p = a * 4 + b
                    if p >= numbdata:
                        continue
                    if False and gdat.datatype == 'mock':
                        if listlablmodl[c][p] == 1 and listlabltrue[p] == 1:
                            colr = 'g'
                        elif listlablmodl[c][p] == 0 and listlabltrue[p] == 0:
                            colr = 'r'
                        elif listlablmodl[c][p] == 0 and listlabltrue[p] == 1:
                            colr = 'b'
                        elif listlablmodl[c][p] == 1 and listlabltrue[p] == 0:
                            colr = 'orange'
                    else:
                        if listlablmodl[c][p] == 1:
                            colr = 'b'
                        else:
                            colr = 'r'
                    axis[a][b].plot(gdat.time, X[p, :], color=colr, alpha=0.1, ls='', marker='o', markersize=3)
                    if a != 9:
                        axis[a][b].set_xticks([])
                    if b != 0:
                        axis[a][b].set_yticks([])
            plt.subplots_adjust(hspace=0, wspace=0)
            path = gdat.pathimag + 'lcurpred_%s_%04d.pdf' % (gdat.strgcntp, c)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()

            # plot a histogram of decision functions evaluated at the samples
            figr, axis = plt.subplots()
            axis.hist(listscor[c], color='k')
            if gdat.datatype == 'mock':
                axis.hist(listscor[c][indxdatasupn], color='g', label='True')
            axis.axvline(scorthrs)
            axis.set_xlabel('$S$')
            axis.set_ylabel('$N(S)$')
            axis.set_yscale('log')
            path = gdat.pathimag + 'histscor_%s_%04d.pdf' % (gdat.strgcntp, c)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
            
            # plot data with the least and highest scores
            numbdataplotsort = min(numbdata, 40)
            figr, axis = plt.subplots(numbdataplotsort, 2, figsize=(12, 2 * numbdataplotsort))
            for l in range(2):
                for k in range(numbdataplotsort):
                    if l == 0:
                        indx = gdat.indxdatascorsort[k]
                    else:
                        indx = gdat.indxdatascorsort[numbdata-k-1]
                    
                    if not isinstance(indx, int):
                        indx = indx[0]
                    axis[k][l].plot(gdat.time, X[indx, :], color='black', ls='', marker='o', markersize=1)
                    axis[k][l].text(.9, .9, 'TIC: %s' % liststrgtici[indx], transform=axis[k][l].transAxes, size=15, ha='right', va='center')
                    if l == 1:
                        axis[k][l].yaxis.set_label_position('right')
                        axis[k][l].yaxis.tick_right()
            plt.subplots_adjust(hspace=0, wspace=0)
            path = gdat.pathimag + 'lcursort_%s_%04d.pdf' % (gdat.strgcntp, c)
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()
        
            numbbins = 10
            numbpositrue = np.zeros(numbbins)
            binsmagt = np.linspace(magtminm, magtmaxm, numbbins + 1)
            meanmagt = (binsmagt[1:] + binsmagt[:-1]) / 2.
            reca = np.empty(numbbins)
            numbsupnmagt = np.zeros(numbbins)
            #if gdat.datatype == 'mock':
            #    for n in indxsupn:
            #        indxmagt = np.digitize(np.amax(truemagt[:, n]), binsmagt) - 1
            #        numbsupnmagt[indxmagt] += 1
            #        if indxpixlposi.size > 0:
            #            numbpositrue[indxmagt] += 1
            #    recamagt = numbpositrue.astype(float) / numbsupnmagt
            #    prec = sum(numbpositrue).astype(float) / numbposi
            #    figr, axis = plt.subplots(figsize=(12, 6))
            #    axis.plot(meanmagt, recamagt, ls='', marker='o')
            #    axis.set_ylabel('Recall')
            #    axis.set_xlabel('Tmag')
            #    plt.tight_layout()
            #    path = gdat.pathimag + 'reca_%s_%04d.pdf' % (gdat.strgcntp, c)
            #    print('Writing to %s...' % path)
            #    plt.savefig(path)
            #    plt.close()
        
            c += 1
                
                
            # clustering with pyod
            # fraction of outliers
            fracoutl = 0.25
            
            # initialize a set of detectors for LSCP
            detector_list = [LOF(n_neighbors=5), LOF(n_neighbors=10), LOF(n_neighbors=15),
                             LOF(n_neighbors=20), LOF(n_neighbors=25), LOF(n_neighbors=30),
                             LOF(n_neighbors=35), LOF(n_neighbors=40), LOF(n_neighbors=45),
                             LOF(n_neighbors=50)]
            
            # Show the statics of the data
            # Define nine outlier detection tools to be compared
            classifiers = {
                'Angle-based Outlier Detector (ABOD)':
                    ABOD(contamination=fracoutl),
                'Cluster-based Local Outlier Factor (CBLOF)':
                    CBLOF(contamination=fracoutl,
                          check_estimator=False, random_state=random_state),
                'Feature Bagging':
                    FeatureBagging(LOF(n_neighbors=35),
                                   contamination=fracoutl,
                                   random_state=random_state),
                #'Histogram-base Outlier Detection (HBOS)': HBOS(
                #    contamination=fracoutl),
                'Isolation Forest': IForest(contamination=fracoutl,
                                            random_state=random_state),
                'K Nearest Neighbors (KNN)': KNN(
                    contamination=fracoutl),
                'Average KNN': KNN(method='mean',
                                   contamination=fracoutl),
                # 'Median KNN': KNN(method='median',
                #                   contamination=fracoutl),
                'Local Outlier Factor (LOF)':
                    LOF(n_neighbors=35, contamination=fracoutl),
                # 'Local Correlation Integral (LOCI)':
                #     LOCI(contamination=fracoutl),
                
                #'Minimum Covariance Determinant (MCD)': MCD(
                #    contamination=fracoutl, random_state=random_state),
                
                'One-class SVM (OCSVM)': OCSVM(contamination=fracoutl),
                'Principal Component Analysis (PCA)': PCA(
                    contamination=fracoutl, random_state=random_state, standardization=False),
                # 'Stochastic Outlier Selection (SOS)': SOS(
                #     contamination=fracoutl),
                'Locally Selective Combination (LSCP)': LSCP(
                    detector_list, contamination=fracoutl,
                    random_state=random_state)
            }
            
            return
            raise Exception('')

            # Fit the model
            plt.figure(figsize=(15, 12))
            for i, (clf_name, clf) in enumerate(classifiers.items()):

                # fit the data and tag outliers
                clf.fit(X)
                scores_pred = clf.decision_function(X) * -1
                y_pred = clf.predict(X)
                threshold = np.percentile(scores_pred, 100 * fracoutl)
                n_errors = np.where(y_pred != listlabltrue)[0].size
                # plot the levels lines and the points
                #if i == 1:
                #    continue
                #Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()]) * -1
                #Z = Z.reshape(xx.shape)
                Z = np.zeros((100, 100))
                subplot = plt.subplot(3, 4, i + 1)
                subplot.contourf(xx, yy, Z, #levels=np.linspace(Z.min(), threshold, 7),
                                 cmap=plt.cm.Blues_r)
                subplot.contourf(xx, yy, Z, cmap=plt.cm.Blues_r)
                a = subplot.contour(xx, yy, Z, levels=[threshold],
                                    linewidths=2, colors='red')
                subplot.contourf(xx, yy, Z, #levels=[threshold, Z.max()],
                                 colors='orange')
                b = subplot.scatter(X[:-numboutl, 0], X[:-numboutl, 1], c='green', s=20, edgecolor='k')
                c = subplot.scatter(X[-numboutl:, 0], X[-numboutl:, 1], c='purple', s=20, edgecolor='k')
                subplot.axis('tight')
                subplot.legend(
                    [a.collections[0], b, c],
                    ['learned decision function', 'true inliers', 'true outliers'],
                    prop=matplotlib.font_manager.FontProperties(size=10),
                    loc='lower right')
                subplot.set_xlabel("%d. %s (errors: %d)" % (i + 1, clf_name, n_errors))
                subplot.set_xlim(limtproj)
                subplot.set_ylim(limtproj)
            plt.subplots_adjust(0.04, 0.1, 0.96, 0.94, 0.1, 0.26)
            plt.suptitle("Outlier detection")
            path = gdat.pathimag + 'pyod.png'
            print('Writing to %s...' % path)
            plt.savefig(path, dpi=300)
            plt.close()

            
            default_base = {'quantile': .3,
                            'eps': .3,
                            'damping': .9,
                            'preference': -200,
                            'n_neighbors': 10,
                            'n_clusters': 3,
                            'min_samples': 20,
                            'xi': 0.05,
                            'min_cluster_size': 0.1}
            
            # update parameters with dataset-specific values
            
            algo_params = {'damping': .77, 'preference': -240,
                 'quantile': .2, 'n_clusters': 2,
                 'min_samples': 20, 'xi': 0.25}

            params = default_base.copy()
            params.update(algo_params)
            
            # normalize dataset for easier parameter selection
            X = StandardScaler().fit_transform(X)
            
            # estimate bandwidth for mean shift
            bandwidth = cluster.estimate_bandwidth(X, quantile=params['quantile'])
            
            # connectivity matrix for structured Ward
            connectivity = kneighbors_graph(
                X, n_neighbors=params['n_neighbors'], include_self=False)
            # make connectivity symmetric
            connectivity = 0.5 * (connectivity + connectivity.T)
            
            ms = cluster.MeanShift(bandwidth=bandwidth, bin_seeding=True)
            two_means = cluster.MiniBatchKMeans(n_clusters=params['n_clusters'])
            ward = cluster.AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='ward', connectivity=connectivity)
            spectral = cluster.SpectralClustering(n_clusters=params['n_clusters'], eigen_solver='arpack', affinity="nearest_neighbors")
            dbscan = cluster.DBSCAN(eps=params['eps'])
            
            #optics = cluster.OPTICS(min_samples=params['min_samples'],
            #                        xi=params['xi'],
            #                        min_cluster_size=params['min_cluster_size'])
            
            affinity_propagation = cluster.AffinityPropagation(damping=params['damping'], preference=params['preference'])
            average_linkage = cluster.AgglomerativeClustering(linkage="average", affinity="cityblock", \
                                                                                n_clusters=params['n_clusters'], connectivity=connectivity)
            birch = cluster.Birch(n_clusters=params['n_clusters'])
            gmm = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full')
            
            clustering_algorithms = (
                ('MiniBatchKMeans', two_means),
                ('AffinityPropagation', affinity_propagation),
                ('MeanShift', ms),
                ('SpectralClustering', spectral),
                ('Ward', ward),
                ('AgglomerativeClustering', average_linkage),
                ('DBSCAN', dbscan),
                #('OPTICS', optics),
                ('Birch', birch),
                ('GaussianMixture', gmm)
            )
            
            figr, axis = plt.subplots(1, numbmeth)
            k = 0
            for name, algorithm in clustering_algorithms:
                t0 = timemodu.time()
                
                # catch warnings related to kneighbors_graph
                with warnings.catch_warnings():
                    #warnings.filterwarnings(
                    #    "ignore",
                    #    message="the number of connected components of the " +
                    #    "connectivity matrix is [0-9]{1,2}" +
                    #    " > 1. Completing it to avoid stopping the tree early.",
                    #    category=UserWarning)
                    #warnings.filterwarnings(
                    #    "ignore",
                    #    message="Graph is not fully connected, spectral embedding" +
                    #    " may not work as expected.",
                    #    category=UserWarning)
                    algorithm.fit(X)
            
                t1 = timemodu.time()
                if hasattr(algorithm, 'labels_'):
                    lablmodl = algorithm.labels_.astype(np.int)
                else:
                    lablmodl = algorithm.predict(X)
                

                axis[k].set_title(name, size=18)
            
                colors = np.array(list(islice(cycle(['#377eb8', '#ff7f00', '#4daf4a',
                                                     '#f781bf', '#a65628', '#984ea3',
                                                     '#999999', '#e41a1c', '#dede00']),
                                              int(max(lablmodl) + 1))))
                # add black color for outliers (if any)
                colors = np.append(colors, ["#000000"])
                axis[k].scatter(X[:, 0], X[:, 1], s=10, color=colors[lablmodl])
            
                axis[k].set_xlim(-2.5, 2.5)
                axis[k].set_ylim(-2.5, 2.5)
                axis[k].set_xticks(())
                axis[k].set_yticks(())
                axis[k].text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
                         transform=plt.gca().transAxes, size=15,
                         horizontalalignment='right')
                k += 1
                listlablmodl.append(lablmodl)
            path = gdat.pathimag + 'clus.pdf'
            print('Writing to %s...' % path)
            plt.savefig(path)
            plt.close()


            # Random 2D projection using a random unitary matrix
            rp = random_projection.SparseRandomProjection(n_components=2)
            X_projected = rp.fit_transform(lcurflat)
            plot_embe(gdat, lcurflat, X_projected, 'rand', "Random Projection")
            
            # Projection on to the first 2 principal components
            t0 = timemodl.time()
            X_pca = decomposition.TruncatedSVD(n_components=2).fit_transform(lcurflat)
            plot_embe(gdat, lcurflat, X_pca, 'pcaa', "Principal Components projection (time %.2fs)" % (timemodl.time() - t0))
            
            # Projection on to the first 2 linear discriminant components
            #X2 = lcurflat.copy()
            #X2.flat[::lcurflat.shape[1] + 1] += 0.01  # Make X invertible
            #t0 = timemodl.time()
            #X_lda = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X2, y)
            #plot_embe(gdat, lcurflat, X_lda, 'ldap', "Linear Discriminant projection (time %.2fs)" % (timemodl.time() - t0))
            
            # t-SNE embedding dataset
            tsne = manifold.TSNE(n_components=2, random_state=0, perplexity=30)
            t0 = timemodl.time()
            X_tsne = tsne.fit_transform(lcurflat)
            plot_embe(gdat, lcurflat, X_tsne, 'tsne0030', "t-SNE embedding with perplexity 30")
            
            # t-SNE embedding dataset
            tsne = manifold.TSNE(n_components=2, random_state=0, perplexity=5)
            t0 = timemodl.time()
            X_tsne = tsne.fit_transform(lcurflat)
            plot_embe(gdat, lcurflat, X_tsne, 'tsne0005', "t-SNE embedding with perplexity 5")
            
            # Isomap projection dataset
            t0 = timemodl.time()
            X_iso = manifold.Isomap(n_neighbors, n_components=2).fit_transform(lcurflat)
            plot_embe(gdat, lcurflat, X_iso, 'isop', "Isomap projection (time %.2fs)" % (timemodl.time() - t0))
            
            # Locally linear embedding dataset
            clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='standard')
            t0 = timemodl.time()
            X_lle = clf.fit_transform(lcurflat)
            plot_embe(gdat, lcurflat, X_lle, 'llep', "Locally Linear Embedding (time %.2fs)" % (timemodl.time() - t0))
            
            # Modified Locally linear embedding dataset
            clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='modified')
            t0 = timemodl.time()
            X_mlle = clf.fit_transform(lcurflat)
            plot_embe(gdat, lcurflat, X_mlle, 'mlle', "Modified Locally Linear Embedding (time %.2fs)" % (timemodl.time() - t0))
            
            # HLLE embedding dataset
            clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='hessian')
            t0 = timemodl.time()
            X_hlle = clf.fit_transform(lcurflat)
            plot_embe(gdat, lcurflat, X_hlle, 'hlle', "Hessian Locally Linear Embedding (time %.2fs)" % (timemodl.time() - t0))
            
            # LTSA embedding dataset
            clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='ltsa')
            t0 = timemodl.time()
            X_ltsa = clf.fit_transform(lcurflat)
            plot_embe(gdat, lcurflat, X_ltsa, 'ltsa', "Local Tangent Space Alignment (time %.2fs)" % (timemodl.time() - t0))
            
            # MDS  embedding dataset
            clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
            t0 = timemodl.time()
            X_mds = clf.fit_transform(lcurflat)
            plot_embe(gdat, lcurflat, X_mds, 'mdse', "MDS embedding (time %.2fs)" % (timemodl.time() - t0))
            
            # Random Trees embedding dataset
            hasher = ensemble.RandomTreesEmbedding(n_estimators=200, random_state=0, max_depth=5)
            t0 = timemodl.time()
            X_transformed = hasher.fit_transform(lcurflat)
            pca = decomposition.TruncatedSVD(n_components=2)
            X_reduced = pca.fit_transform(X_transformed)
            plot_embe(gdat, lcurflat, X_reduced, 'rfep', "Random forest embedding (time %.2fs)" % (timemodl.time() - t0))
            
            # Spectral embedding dataset
            embedder = manifold.SpectralEmbedding(n_components=2, random_state=0, eigen_solver="arpack")
            t0 = timemodl.time()
            X_se = embedder.fit_transform(lcurflat)
            plot_embe(gdat, lcurflat, X_se, 'csep', "Spectral embedding (time %.2fs)" % (timemodl.time() - t0))
            
            # NCA projection dataset
            #nca = neighbors.NeighborhoodComponentsAnalysis(n_components=2, random_state=0)
            #t0 = timemodl.time()
            #X_nca = nca.fit_transform(lcurflat, y)
            #plot_embe(gdat, lcurflat, X_nca, 'ncap', "NCA embedding (time %.2fs)" % (timemodl.time() - t0))

            figr, axis = plt.subplots(figsize=(12, 6))
            
            for strgvarb in ['diff']:
                figr, axis = plt.subplots(figsize=(12, 6))
                #if strgvarb == 'diff':
                #    varbtemp = np.arcsinh(dictpara[strgvarb])
                #else:
                #    varbtemp = dictpara[strgvarb]
                varbtemp = dictpara[strgvarb]
                vmin = -1
                vmax = 1
                objtimag = axis.imshow(varbtemp, interpolation='nearest', cmap='Greens', vmin=vmin, vmax=vmax)
                plt.colorbar(objtimag)
                plt.tight_layout()
                path = gdat.pathimag + '%s_%s.pdf' % (strgvarb, gdat.strgcntp)
                print('Writing to %s...' % path)
                plt.savefig(path)
                plt.close()
    

def retr_timeexec():
    # input PCAT speed per 100x100 pixel region
    timeregi = 30. # [min]
    
    # number of time frames in each region
    numbtser = 13.7 * 4 * 24 * 60. / 30.
    
    timeregitser = numbtser * timeregi / 60. / 24 # [day]
    timeffim = 16.8e6 / 1e4 * timeregi # [day]
    timesegm = 4. * timeffim / 7. # [week]
    timefsky = 26 * timesegm / 7. # [week]
    
    print('Full frame, full sky: %d weeks per 1000 cores' % (timefsky / 1000.))


def plot_peri(): 
    ## plot Lomb Scargle periodogram
    figr, axis = plt.subplots(figsize=(12, 4))
    axis.set_ylabel('Power')
    axis.set_xlabel('Frequency [1/day]')
    arryfreq = np.linspace(0.1, 10., 2000)
    for a in range(2):
        indxtemp = np.arange(arryseco.shape[0])
        if a == 0:
            colr = 'g'
        if a == 1:
            colr = 'r'
            for k in range(1400, 1500):
                indxtemp = np.setdiff1d(indxtemp, np.where(abs(arryseco[:, 0] - k * peri - epoc) < dura * 2)[0])
        ydat = scipy.signal.lombscargle(arryseco[indxtemp, 0], arryseco[indxtemp, 1], arryfreq)
        axis.plot(arryfreq * 2. * np.pi, ydat, ls='', marker='o', markersize=5, alpha=0.3, color=colr)
    for a in range(4):
        axis.axvline(a / peri, ls='--', color='black')
    plt.tight_layout()
    path = gdat.pathimag + 'lspd_%s.pdf' % (strgmask)
    print('Writing to %s...' % path)
    plt.savefig(path)
    plt.close()

def plot_catl(gdat, axis, indxsideyposoffs=0, indxsidexposoffs=0):

    try:
        for k in range(gdat.numbpositext):
            axis.text(gdat.indxsideyposdataflat[gdat.indxdatascorsort[k]] - indxsideyposoffs + gdat.numbsideedge, \
                      gdat.indxsidexposdataflat[gdat.indxdatascorsort[k]] - indxsidexposoffs + gdat.numbsideedge, '%d' % k, size=7, color='b', alpha=0.3)
    except:
        pass

    if gdat.datatype == 'mock':

        for k in gdat.indxsour:
            axis.text(np.mean(gdat.trueypos[:, k]) - indxsideyposoffs, \
                      np.mean(gdat.truexpos[:, k]) - indxsidexposoffs, '*', alpha=0.1, size=15, color='y', ha='center', va='center')
            #axis.text(np.mean(gdat.trueypos[:, k]) - indxsideyposoffs + 0.5, \
            #          np.mean(gdat.truexpos[:, k]) - indxsidexposoffs + 0.5, '%.3g, %.3g' % (gdat.truemagtmean[k], gdat.truemagtstdv[k]), \
            #                                                        alpha=0.3, size=5, color='y', ha='center', va='center')

        for k in gdat.indxsoursupn:
            axis.text(np.mean(gdat.trueypos[:, k]) - indxsideyposoffs, \
                      np.mean(gdat.truexpos[:, k]) - indxsidexposoffs, '*', alpha=0.1, size=15, color='g', ha='center', va='center')
            axis.text(np.mean(gdat.trueypos[:, k]) - indxsideyposoffs + 0.5, \
                      np.mean(gdat.truexpos[:, k]) - indxsidexposoffs + 0.5, '%.3g, %.3g' % (gdat.truemagtmean[k], gdat.truemagtstdv[k]), \
                                                                                                alpha=0.1, size=5, color='g', ha='center', va='center')



def setp_modlemis_init(gdat, strgmodl):
    
    print('setp_modlemis_init(): Initial setup for model %s...' % (strgmodl))

    gmod = getattr(gdat, strgmodl)
    
    if strgmodl == 'fitt':
        gmod.lablmodl = 'Model'
    if strgmodl == 'true':
        gmod.lablmodl = 'True'
    
    # transdimensional element populations
    gmod.numbpopl = len(gmod.typeelem)
    gmod.indxpopl = np.arange(gmod.numbpopl)
    
    gmod.namepara.genrelem = [[] for l in gmod.indxpopl]
    gmod.scalpara.genrelem = [[] for l in gmod.indxpopl]
    gmod.namepara.derielemodim = [[] for l in gmod.indxpopl]
    
    # background component
    gmod.numbback = 0
    gmod.indxback = []
    for c in range(len(gmod.typeback)):
        if isinstance(gmod.typeback[c], str):
            if gmod.typeback[c].startswith('bfunfour') or gmod.typeback[c].startswith('bfunwfou'):
                namebfun = gmod.typeback[c][:8]
                ordrexpa = int(gmod.typeback[c][8:])
                numbexpa = 4 * ordrexpa**2
                indxexpa = np.arange(numbexpa)
                del gmod.typeback[c]
                for k in indxexpa:
                    gmod.typeback.insert(c+k, namebfun + '%04d' % k)
    gmod.numbback = len(gmod.typeback)
    gmod.indxback = np.arange(gmod.numbback)
    gmod.numbbacktotl = np.sum(gmod.numbback)
    gmod.indxbacktotl = np.arange(gmod.numbbacktotl)
    
    # name of the generative element parameter used for the amplitude
    gmod.nameparagenrelemampl = [[] for l in gmod.indxpopl]
    gmod.indxpara.genrelemampl = [[] for l in gmod.indxpopl]
    for l in gmod.indxpopl:
        if gmod.typeelem[l] == 'lghtpntspuls':
            gmod.nameparagenrelemampl[l] = 'per0'
            gmod.indxpara.genrelemampl[l] = 2
        elif gmod.typeelem[l] == 'lghtpntsagnntrue':
            gmod.nameparagenrelemampl[l] = 'lum0'
            gmod.indxpara.genrelemampl[l] = 2
        elif gmod.typeelem[l].startswith('lghtline'):
            gmod.nameparagenrelemampl[l] = 'flux'
            gmod.indxpara.genrelemampl[l] = 1
        elif gmod.typeelem[l].startswith('lghtpnts'):
            gmod.nameparagenrelemampl[l] = 'flux'
            gmod.indxpara.genrelemampl[l] = 2
        elif gmod.typeelem[l].startswith('lghtgausbgrd'):
            gmod.nameparagenrelemampl[l] = 'flux'
            gmod.indxpara.genrelemampl[l] = 2
        if gmod.typeelem[l] == 'lens':
            gmod.nameparagenrelemampl[l] = 'defs'
            gmod.indxpara.genrelemampl[l] = 2
        if gmod.typeelem[l].startswith('clus'):
            gmod.nameparagenrelemampl[l] = 'nobj'
            gmod.indxpara.genrelemampl[l] = 2
        if gmod.typeelem[l] == 'lens':
            gmod.nameparagenrelemampl[l] = 'defs'
        if gmod.typeelem[l] == 'clus':
            gmod.nameparagenrelemampl[l] = 'nobj'
        if len(gmod.nameparagenrelemampl[l]) == 0:
            raise Exception('Amplitude feature undefined.')
    
    # galaxy components
    gmod.indxsersfgrd = np.arange(gmod.numbsersfgrd)


def setp_modlemis_finl(gdat, strgmodl):
    
    print('setp_modlemis_finl(): Building parameter indices for model %s...' % (strgmodl))
        
    gmod = getattr(gdat, strgmodl)
    
    # element setup
    ## Boolean flag to calculate the approximation error due to the finite PSF kernel
    boolcalcerrr = [[] for l in gmod.indxpopl]
    for l in gmod.indxpopl:
        # if the PSF evaluation is local and the number of pixels is small, do it
        if gmod.typeelemspateval[l] == 'locl' and gdat.numbpixlfull < 1e5:
            boolcalcerrr[l] = True
        else:
            boolcalcerrr[l] = False
    setp_varb(gdat, 'boolcalcerrr', valu=boolcalcerrr, strgmodl=strgmodl)
    
    ## minimum number of elements summed over all populations
    gmod.minmpara.numbelemtotl = np.amin(gmod.minmpara.numbelem)
            
    ## maximum number of elements summed over all populations
    gmod.maxmpara.numbelemtotl = np.sum(gmod.maxmpara.numbelem) 

    ## name of the element feature used to sort the elements, for each population
    gmod.nameparaelemsort = [[] for l in gmod.indxpopl]
    for l in gmod.indxpopl:
        if gmod.typeelem[l].startswith('lght'):
            gmod.nameparaelemsort[l] = 'flux'
        if gmod.typeelem[l] == 'lens':
            gmod.nameparaelemsort[l] = 'defs'
        if gmod.typeelem[l].startswith('clus'):
            gmod.nameparaelemsort[l] = 'nobj'
    
    ## subscript used to denote the elements of each population
    gmod.lablelemsubs = [[] for l in gmod.indxpopl]
    for l in gmod.indxpopl:
        if gdat.numbgrid > 1:
            if gmod.typeelem[l] == 'lghtpnts':
                gmod.lablelemsubs[l] = r'\rm{fps}'
            if gmod.typeelem[l] == 'lghtgausbgrd':
                gmod.lablelemsubs[l] = r'\rm{bgs}'
        else:
            if gmod.typeelem[l].startswith('lghtpntspuls'):
                gmod.lablelemsubs[l] = r'\rm{pul}'
            if gmod.typeelem[l].startswith('lghtpntsagnn'):
                gmod.lablelemsubs[l] = r'\rm{agn}'
            elif gmod.typeelem[l] == 'lghtpnts':
                gmod.lablelemsubs[l] = r'\rm{pts}'
        if gmod.typeelem[l] == 'lens':
            gmod.lablelemsubs[l] = r'\rm{sub}'
        if gmod.typeelem[l].startswith('clus'):
            gmod.lablelemsubs[l] = r'\rm{cls}'
        if gmod.typeelem[l].startswith('lghtline'):
            gmod.lablelemsubs[l] = r'\rm{lin}'
    
    ## indices of element populations for each coordinate grid
    gmod.indxpoplgrid = [[] for y in gdat.indxgrid]
    for y in gdat.indxgrid: 
        for indx, typeelemtemp in enumerate(gmod.typeelem):
            # foreground grid (image plane) -- the one np.where the data is measured
            if y == 0:
                if typeelemtemp.startswith('lght') and not typeelemtemp.endswith('bgrd') or typeelemtemp.startswith('clus'):
                    gmod.indxpoplgrid[y].append(indx)
            # foreground mass grid
            if y == 1:
                if typeelemtemp.startswith('lens'):
                    gmod.indxpoplgrid[y].append(indx)
            # background grid (source plane)
            if y == 2:
                if typeelemtemp.endswith('bgrd'):
                    gmod.indxpoplgrid[y].append(indx)
    
    ## indices of the coordinate grids for each population
    indxgridpopl = [[] for l in gmod.indxpopl]
    for l in gmod.indxpopl:
        for y in gdat.indxgrid:
            if l in gmod.indxpoplgrid[y]:
                indxgridpopl[l] = y
    
    ## Boolean flag to calculate the surface brightness due to elements
    gmod.boolcalcelemsbrt = False
    for l in gmod.indxpopl:
        if gmod.typeelem[l].startswith('lghtpnts'):
            gmod.boolcalcelemsbrt = True
    
    ## Boolean flag to calculate the surface brightness due to elements in the background coordinate grid
    if 'lghtgausbgrd' in gmod.typeelem:
        gmod.boolcalcelemsbrtbgrd = True
    else:
        gmod.boolcalcelemsbrtbgrd = False
    
    ## Boolean flag indicating, for each element population, whether elements are light sources
    gmod.boolelemlght = [[] for l in gmod.indxpopl]
    for l in gmod.indxpopl:
        if gmod.typeelem[l].startswith('lght'):
            gmod.boolelemlght[l] = True
        else:
            gmod.boolelemlght[l] = False
    
    ## Boolean flag indicating whether any population of elements is a light source
    gmod.boolelemlghtanyy = True in gmod.boolelemlght
    
    ## Boolean flag indicating whether the surface brightness on the background grid is lensed to due elements
    gmod.boolelemlens = False
    for l in gmod.indxpopl:
        if gmod.typeelem[l].startswith('lens'):
            gmod.boolelemlens = True
    
    ## Boolean flag indicating, for each population of elements, whether the elements are delta-function light sources
    gmod.boolelemsbrtdfnc = [[] for l in gmod.indxpopl]
    for l in gmod.indxpopl:
        if gmod.maxmpara.numbelem[l] > 0 and (gmod.typeelem[l].startswith('lght') and not gmod.typeelem[l].endswith('bgrd') or gmod.typeelem[l].startswith('clus')):
            gmod.boolelemsbrtdfnc[l] = True
        else:
            gmod.boolelemsbrtdfnc[l] = False
    ## Boolean flag indicating whether any population of elements has delta-function light sources
    gmod.boolelemsbrtdfncanyy = True in gmod.boolelemsbrtdfnc

    ## Boolean flag indicating, for each population of elements, whether the elements are subhalos deflecting the light from the background grid
    gmod.boolelemdeflsubh = [[] for l in gmod.indxpopl]
    for l in gmod.indxpopl:
        if gmod.typeelem[l] == 'lens':
            gmod.boolelemdeflsubh[l] = True
        else:
            gmod.boolelemdeflsubh[l] = False
    ## Boolean flag indicating whether any population of elements has elements that are subhalos deflecting the light from the background grid
    gmod.boolelemdeflsubhanyy = True in gmod.boolelemdeflsubh

    gmod.boolelemsbrtextsbgrd = [[] for l in gmod.indxpopl]
    for l in gmod.indxpopl:
        if gmod.typeelem[l].startswith('lght') and gmod.typeelem[l].endswith('bgrd'):
            gmod.boolelemsbrtextsbgrd[l] = True
        else:
            gmod.boolelemsbrtextsbgrd[l] = False
    gmod.boolelemsbrtextsbgrdanyy = True in gmod.boolelemsbrtextsbgrd
    
    if gmod.boolelemsbrtextsbgrdanyy:
        gmod.indxpopllens = 1
    else:
        gmod.indxpopllens = 0

    gmod.boolelemsbrtpnts = [[] for l in gmod.indxpopl]
    for l in gmod.indxpopl:
        if gmod.typeelem[l].startswith('lght') and gmod.typeelem[l] != 'lghtline' or gmod.typeelem[l] == 'clus':
            gmod.boolelemsbrtpnts[l] = True
        else:
            gmod.boolelemsbrtpnts[l] = False
    gmod.boolelemsbrtpntsanyy = True in gmod.boolelemsbrtpnts

    # temp -- because there is currently no extended source
    gmod.boolelemsbrt = gmod.boolelemsbrtdfnc
    
    gmod.boolelempsfn = [[] for l in gmod.indxpopl]
    for l in gmod.indxpopl:
        if gmod.typeelem[l].startswith('lghtpnts') or gmod.typeelem[l] == 'clus':
            gmod.boolelempsfn[l] = True
        else:
            gmod.boolelempsfn[l] = False
    gmod.boolelempsfnanyy = True in gmod.boolelempsfn
    
    spectype = [[] for l in gmod.indxpopl]
    for l in gmod.indxpopl:
        if gmod.boolelemlght[l]:
            spectype[l] = 'powr'
        else:
            spectype[l] = 'none'
    setp_varb(gdat, 'spectype', valu=spectype, strgmodl=strgmodl)
    
    if gdat.boolbindspat:
        minmgwdt = 2. * gdat.sizepixl
        maxmgwdt = gdat.maxmgangdata / 4.
        setp_varb(gdat, 'gwdt', minm=minmgwdt, maxm=maxmgwdt, strgmodl=strgmodl)
    
    setp_varb(gdat, 'aerr', minm=-100, maxm=100, strgmodl=strgmodl, popl='full')
    
    if gmod.boolelemlghtanyy:
        # flux
        if gdat.typeexpr == 'ferm':
            minmflux = 1e-9
            maxmflux = 1e-6
        if gdat.typeexpr == 'tess':
            minmflux = 1.
            maxmflux = 1e3
        if gdat.typeexpr == 'chan':
            if gdat.anlytype == 'spec':
                minmflux = 1e4
                maxmflux = 1e7
            else:
                minmflux = 3e-9
                maxmflux = 1e-6
        if gdat.typeexpr == 'gmix':
            minmflux = 0.1
            maxmflux = 100.
        if gdat.typeexpr.startswith('HST_WFC3'):
            minmflux = 1e-20
            maxmflux = 1e-17
        if gdat.typeexpr == 'fire':
            minmflux = 1e-20
            maxmflux = 1e-17
        setp_varb(gdat, 'flux', limt=[minmflux, maxmflux], strgmodl=strgmodl)
        
        if gdat.typeexpr == 'ferm':
            setp_varb(gdat, 'brekprioflux', limt=[3e-9, 1e-6], popl=l, strgmodl=strgmodl)
            setp_varb(gdat, 'sloplowrprioflux', limt=[0.5, 3.], popl=l, strgmodl=strgmodl)
            setp_varb(gdat, 'slopupprprioflux', limt=[0.5, 3.], popl=l, strgmodl=strgmodl)
        
        if gdat.boolbinsener:
            ### spectral parameters
            if gdat.typeexpr == 'ferm':
                sind = [1., 3.]
                minmsind = 1.
                maxmsind = 3.
            if gdat.typeexpr == 'chan':
                minmsind = 0.4
                maxmsind = 2.4
                sind = [0.4, 2.4]
            if gdat.typeexpr.startswith('HST_WFC3'):
                minmsind = 0.5
                maxmsind = 2.5
                sind = [0.4, 2.4]
            if gdat.typeexpr != 'fire':
                setp_varb(gdat, 'sind', limt=[minmsind, maxmsind], strgmodl=strgmodl)
                setp_varb(gdat, 'curv', limt=[-1., 1.], strgmodl=strgmodl)
                setp_varb(gdat, 'expc', limt=[0.1, 10.], strgmodl=strgmodl)
                setp_varb(gdat, 'sinddistmean', limt=sind, popl='full', strgmodl=strgmodl)
                #### standard deviations should not be too small
                setp_varb(gdat, 'sinddiststdv', limt=[0.3, 2.], popl='full', strgmodl=strgmodl)
                setp_varb(gdat, 'curvdistmean', limt=[-1., 1.], popl='full', strgmodl=strgmodl)
                setp_varb(gdat, 'curvdiststdv', limt=[0.1, 1.], popl='full', strgmodl=strgmodl)
                setp_varb(gdat, 'expcdistmean', limt=[1., 8.], popl='full', strgmodl=strgmodl)
                setp_varb(gdat, 'expcdiststdv', limt=[0.01 * gdat.maxmener, gdat.maxmener], popl='full', strgmodl=strgmodl)
                for i in gdat.indxenerinde:
                    setp_varb(gdat, 'sindcolr0001', limt=[-2., 6.], strgmodl=strgmodl)
                    setp_varb(gdat, 'sindcolr0002', limt=[0., 8.], strgmodl=strgmodl)
                    setp_varb(gdat, 'sindcolr%04d' % i, limt=[-5., 10.], strgmodl=strgmodl)
    
    for l in gmod.indxpopl:
        if gmod.typeelem[l] == 'lghtpntspuls':
            setp_varb(gdat, 'gang', limt=[1e-1 * gdat.sizepixl, gdat.maxmgangdata], strgmodl=strgmodl)
            setp_varb(gdat, 'geff', limt=[0., 0.4], strgmodl=strgmodl)
            setp_varb(gdat, 'dglc', limt=[10., 3e3], strgmodl=strgmodl)
            setp_varb(gdat, 'phii', limt=[0., 2. * np.pi], strgmodl=strgmodl)
            setp_varb(gdat, 'thet', limt=[0., np.pi], strgmodl=strgmodl)
            setp_varb(gdat, 'per0distmean', limt=[5e-4, 1e1], popl=l, strgmodl=strgmodl)
            setp_varb(gdat, 'magfdistmean', limt=[1e7, 1e16], popl=l, strgmodl=strgmodl)
            setp_varb(gdat, 'per0diststdv', limt=[1e-2, 1.], popl=l, strgmodl=strgmodl)
            setp_varb(gdat, 'magfdiststdv', limt=[1e-2, 1.], popl=l, strgmodl=strgmodl)
            setp_varb(gdat, 'gangslop', limt=[0.5, 4.], popl=l, strgmodl=strgmodl)
            setp_varb(gdat, 'dglcslop', limt=[0.5, 2.], popl=l, strgmodl=strgmodl)
            setp_varb(gdat, 'spatdistcons', limt=[1e-4, 1e-2], popl='full')
            setp_varb(gdat, 'yposdistscal', limt=[0.5 / gdat.anglfact, 5. / gdat.anglfact], popl='full', strgmodl=strgmodl)
        if gmod.typeelem[l] == 'lghtpntsagnntrue':
            setp_varb(gdat, 'dlos', limt=[1e7, 1e9], strgmodl=strgmodl)
            setp_varb(gdat, 'dlosslop', limt=[-0.5, -3.], popl=l, strgmodl=strgmodl)
            setp_varb(gdat, 'lum0', limt=[1e43, 1e46], strgmodl=strgmodl)
            setp_varb(gdat, 'lum0distbrek', limt=[1e42, 1e46], popl=l, strgmodl=strgmodl)
            setp_varb(gdat, 'lum0sloplowr', limt=[0.5, 3.], popl=l, strgmodl=strgmodl)
            setp_varb(gdat, 'lum0slopuppr', limt=[0.5, 3.], popl=l, strgmodl=strgmodl)
    
    # construct background surface brightness templates from the user input
    gmod.sbrtbacknorm = [[] for c in gmod.indxback]
    gmod.boolunifback = np.ones(gmod.numbback, dtype=bool)
    for c in gmod.indxback:
        gmod.sbrtbacknorm[c] = np.empty((gdat.numbenerfull, gdat.numbpixlfull, gdat.numbdqltfull))
        if gmod.typeback[c] == 'data':
            gmod.sbrtbacknorm[c] = np.copy(gdat.sbrtdata)
            gmod.sbrtbacknorm[c][np.where(gmod.sbrtbacknorm[c] == 0.)] = 1e-100
        elif isinstance(gmod.typeback[c], float):
            gmod.sbrtbacknorm[c] = np.zeros((gdat.numbenerfull, gdat.numbpixlfull, gdat.numbdqltfull)) + gmod.typeback[c]
        elif isinstance(gmod.typeback[c], list) and isinstance(gmod.typeback[c], float):
            gmod.sbrtbacknorm[c] = retr_spec(gdat, np.array([gmod.typeback[c]]), sind=np.array([gmod.typeback[c]]))[:, 0, None, None]
        elif isinstance(gmod.typeback[c], np.ndarray) and gmod.typeback[c].ndim == 1:
            gmod.sbrtbacknorm[c] = np.zeros((gdat.numbenerfull, gdat.numbpixlfull, gdat.numbdqltfull)) + gmod.typeback[c][:, None, None]
        elif gmod.typeback[c].startswith('bfunfour') or gmod.typeback[c].startswith('bfunwfou'):
            indxexpatemp = int(gmod.typeback[c][8:]) 
            indxterm = indxexpatemp // ordrexpa**2
            indxexpaxdat = (indxexpatemp % ordrexpa**2) // ordrexpa + 1
            indxexpaydat = (indxexpatemp % ordrexpa**2) % ordrexpa + 1
            if namebfun == 'bfunfour':
                ampl = 1.
                func = gdat.bctrpara.yposcart 
            if namebfun == 'bfunwfou':
                functemp = np.exp(-0.5 * (gdat.bctrpara.yposcart / (1. / gdat.anglfact))**2)
                ampl = np.sqrt(functemp)
                func = functemp
            argsxpos = 2. * np.pi * indxexpaxdat * gdat.bctrpara.xposcart / gdat.maxmgangdata
            argsypos = 2. * np.pi * indxexpaydat * func / gdat.maxmgangdata
            if indxterm == 0:
                termfrst = np.sin(argsxpos)
                termseco = ampl * np.sin(argsypos)
            if indxterm == 1:
                termfrst = np.sin(argsxpos)
                termseco = ampl * np.cos(argsypos)
            if indxterm == 2:
                termfrst = np.cos(argsxpos)
                termseco = ampl * np.sin(argsypos)
            if indxterm == 3:
                termfrst = np.cos(argsxpos)
                termseco = ampl * np.cos(argsypos)
            gmod.sbrtbacknorm[c] = (termfrst[None, :] * termseco[:, None]).flatten()[None, :, None] * \
                                                        np.ones((gdat.numbenerfull, gdat.numbpixlfull, gdat.numbdqltfull))
            
        else:
            path = gdat.pathinpt + gmod.typeback[c]
            gmod.sbrtbacknorm[c] = astropy.io.fits.getdata(path)
            
            if gdat.typepixl == 'cart':
                if not gdat.boolforccart:
                    if gmod.sbrtbacknorm[c].shape[2] != gdat.numbsidecart:
                        raise Exception('Provided background template must have the chosen image dimensions.')
                
                gmod.sbrtbacknorm[c] = gmod.sbrtbacknorm[c].reshape((gmod.sbrtbacknorm[c].shape[0], -1, gmod.sbrtbacknorm[c].shape[-1]))
    
            if gdat.typepixl == 'cart' and gdat.boolforccart:
                sbrtbacknormtemp = np.empty((gdat.numbenerfull, gdat.numbpixlfull, gdat.numbdqltfull))
                for i in gdat.indxenerfull:
                    for m in gdat.indxdqltfull:
                        sbrtbacknormtemp[i, :, m] = tdpy.retr_cart(gmod.sbrtbacknorm[c][i, :, m], \
                                                numbsidexpos=gdat.numbsidecart, numbsideypos=gdat.numbsidecart, \
                                                minmxpos=gdat.anglfact*gdat.minmxposdata, maxmxpos=gdat.anglfact*gdat.maxmxposdata, \
                                                minmypos=gdat.anglfact*gdat.minmyposdata, maxmypos=gdat.anglfact*gdat.maxmyposdata).flatten()
                gmod.sbrtbacknorm[c] = sbrtbacknormtemp

        # determine spatially uniform background templates
        for i in gdat.indxenerfull:
            for m in gdat.indxdqltfull:
                if np.std(gmod.sbrtbacknorm[c][i, :, m]) > 1e-6:
                    gmod.boolunifback[c] = False

    boolzero = True
    gmod.boolbfun = False
    for c in gmod.indxback:
        if np.amin(gmod.sbrtbacknorm[c]) < 0. and isinstance(gmod.typeback[c], str) and not gmod.typeback[c].startswith('bfun'):
            booltemp = False
            raise Exception('Background templates must be positive-definite every where.')
    
        if not np.isfinite(gmod.sbrtbacknorm[c]).all():
            raise Exception('Background template is not finite.')

        if np.amin(gmod.sbrtbacknorm[c]) > 0. or gmod.typeback[c] == 'data':
            boolzero = False
        
        if isinstance(gmod.typeback[c], str) and gmod.typeback[c].startswith('bfun'):
            gmod.boolbfun = True
    
    if boolzero and not gmod.boolbfun:
        raise Exception('At least one background template must be positive everywhere.')
    
    # temp -- does not take into account dark hosts
    gmod.boolhost = gmod.typeemishost != 'none'
    
    # type of PSF evaluation
    if gmod.maxmpara.numbelemtotl > 0 and gmod.boolelempsfnanyy:
        if gmod.typeemishost != 'none' or not gmod.boolunifback.all():
            # the background is not convolved by a kernel and point sources exist
            typeevalpsfn = 'full'
        else:
            # the background is not convolved by a kernel and point sources exist
            typeevalpsfn = 'kern'
    else:
        if gmod.typeemishost != 'none' or not gmod.boolunifback.all():
            # the background is convolved by a kernel, no point source exists
            typeevalpsfn = 'conv'
        else:
            # the background is not convolved by a kernel, no point source exists
            typeevalpsfn = 'none'
    setp_varb(gdat, 'typeevalpsfn', valu=typeevalpsfn, strgmodl=strgmodl)
    
    if gdat.typeverb > 1:
        print('gmod.typeevalpsfn')
        print(gmod.typeevalpsfn)
    
    gmod.boolapplpsfn = gmod.typeevalpsfn != 'none'
    
    ### PSF model
    if gmod.typeevalpsfn != 'none':
        
        if gmod.typemodlpsfn == 'singgaus':
            numbpsfpform = 1
        elif gmod.typemodlpsfn == 'singking':
            numbpsfpform = 2
        elif gmod.typemodlpsfn == 'doubgaus':
            numbpsfpform = 3
        elif gmod.typemodlpsfn == 'gausking':
            numbpsfpform = 4
        elif gmod.typemodlpsfn == 'doubking':
            numbpsfpform = 5
        
        gmod.numbpsfptotl = numbpsfpform
        
        if gdat.boolpriopsfninfo:
            for i in gdat.indxener:
                for m in gdat.indxdqlt:
                    meansigc = gmod.psfpexpr[i * gmod.numbpsfptotl + m * gmod.numbpsfptotl * gdat.numbener]
                    stdvsigc = meansigc * 0.1
                    setp_varb(gdat, 'sigcen%02devt%d' % (i, m), mean=meansigc, stdv=stdvsigc, labl=['$\sigma$', ''], scal='gaus', strgmodl=strgmodl)
                    
                    if gmod.typemodlpsfn == 'doubking' or gmod.typemodlpsfn == 'singking':
                        meangamc = gmod.psfpexpr[i * numbpsfpform + m * numbpsfpform * gdat.numbener + 1]
                        stdvgamc = meangamc * 0.1
                        setp_varb(gdat, 'gamcen%02devt%d' % (i, m), mean=meangamc, stdv=stdvgamc, strgmodl=strgmodl)
                        if gmod.typemodlpsfn == 'doubking':
                            meansigt = gmod.psfpexpr[i * numbpsfpform + m * numbpsfpform * gdat.numbener + 2]
                            stdvsigt = meansigt * 0.1
                            setp_varb(gdat, 'sigten%02devt%d' % (i, m), mean=meansigt, stdv=stdvsigt, strgmodl=strgmodl)
                            meangamt = gmod.psfpexpr[i * numbpsfpform + m * numbpsfpform * gdat.numbener + 3]
                            stdvgamt = meangamt * 0.1
                            setp_varb(gdat, 'gamten%02devt%d' % (i, m), mean=meangamt, stdv=stdvgamt, strgmodl=strgmodl)
                            meanpsff = gmod.psfpexpr[i * numbpsfpform + m * numbpsfpform * gdat.numbener + 4]
                            stdvpsff = meanpsff * 0.1
                            setp_varb(gdat, 'psffen%02devt%d' % (i, m), mean=meanpsff, stdv=stdvpsff, strgmodl=strgmodl)
        else:
            if gdat.typeexpr == 'gmix':
                minmsigm = 0.01 / gdat.anglfact
                maxmsigm = 0.1 / gdat.anglfact
            if gdat.typeexpr == 'ferm':
                minmsigm = 0.1
                maxmsigm = 10.
            if gdat.typeexpr.startswith('HST_WFC3'):
                minmsigm = 0.01 / gdat.anglfact
                maxmsigm = 0.1 / gdat.anglfact
            if gdat.typeexpr == 'chan':
                minmsigm = 0.1 / gdat.anglfact
                maxmsigm = 2. / gdat.anglfact
            minmgamm = 1.5
            maxmgamm = 20.
            setp_varb(gdat, 'sigc', valu= 0.05/gdat.anglfact, minm=minmsigm, maxm=maxmsigm, labl=['$\sigma_c$', ''], ener='full', dqlt='full', strgmodl=strgmodl, strgstat='this')
            setp_varb(gdat, 'sigt', minm=minmsigm, maxm=maxmsigm, ener='full', dqlt='full', strgmodl=strgmodl)
            setp_varb(gdat, 'gamc', minm=minmgamm, maxm=maxmgamm, ener='full', dqlt='full', strgmodl=strgmodl)
            setp_varb(gdat, 'gamt', minm=minmgamm, maxm=maxmgamm, ener='full', dqlt='full', strgmodl=strgmodl)
            
        setp_varb(gdat, 'psff', minm=0., maxm=1., ener='full', dqlt='full', strgmodl=strgmodl)
 
    # background
    ## number of background parameters
    numbbacp = 0
    for c in gmod.indxback:
        if gmod.boolspecback[c]:
            numbbacp += 1
        else:
            numbbacp += gdat.numbener
   
    ## background parameter indices
    gmod.indxbackbacp = np.zeros(numbbacp, dtype=int)
    indxenerbacp = np.zeros(numbbacp, dtype=int)
    cntr = 0
    for c in gmod.indxback:
        if gmod.boolspecback[c]:
            gmod.indxbackbacp[cntr] = c
            cntr += 1
        else:
            for i in gdat.indxener:
                indxenerbacp[cntr] = i
                gmod.indxbackbacp[cntr] = c
                cntr += 1
    
    # indices of background parameters for each background component
    gmod.indxbacpback = [[] for c in gmod.indxback]
    for c in gmod.indxback:
        gmod.indxbacpback[c] = np.where((gmod.indxbackbacp == c))[0]
            
    # list of names of diffuse components
    gmod.listnamediff = []
    for c in gmod.indxback:
        gmod.listnamediff += ['back%04d' % c]
    if gmod.typeemishost != 'none':
        for e in gmod.indxsersfgrd:
            gmod.listnamediff += ['hostisf%d' % e]
    if gmod.boollens:
        gmod.listnamediff += ['lens']
    
    # list of names of emission components
    listnameecom = deepcopy(gmod.listnamediff)
    for l in gmod.indxpopl:
        if gmod.boolelemsbrt[l] and gmod.maxmpara.numbelem[l] > 0:
            if not 'dfnc' in listnameecom:
                listnameecom += ['dfnc']
            if not 'dfncsubt' in listnameecom:
                listnameecom += ['dfncsubt']
    gmod.listnameecomtotl = listnameecom + ['modl']
    
    for c in gmod.indxback:
        setp_varb(gdat, 'cntpback%04d' % c, labl=['$C_{%d}$' % c, ''], minm=1., maxm=100., scal='logt', strgmodl=strgmodl)
    
    gmod.listnamegcom = deepcopy(gmod.listnameecomtotl)
    if gmod.boollens:
        gmod.listnamegcom += ['bgrd']
        if gmod.numbpopl > 0 and gmod.boolelemsbrtextsbgrdanyy:
            gmod.listnamegcom += ['bgrdgalx', 'bgrdexts']
    
    numbdiff = len(gmod.listnamediff)
    convdiff = np.zeros(numbdiff, dtype=bool)
    for k, namediff in enumerate(gmod.listnamediff):
        if not (gdat.boolthindata or gmod.typeevalpsfn == 'none' or gmod.typeevalpsfn == 'kern'):
            if namediff.startswith('back'):
                indx = int(namediff[-4:])
                convdiff[k] = not gmod.boolunifback[indx] 
            else:
                convdiff[k] = True
    
    if gdat.typeverb > 0:
        if strgmodl == 'true':
            strgtemp = 'true'
        if strgmodl == 'fitt':
            strgtemp = 'fitting'
        print('Building elements for the %s model...' % strgtemp)
    
    # element parameters that correlate with the statistical significance of the element
    gmod.namepara.elemsign = [[] for l in gmod.indxpopl]
    for l in gmod.indxpopl:
        if gmod.typeelem[l].startswith('lght'):
            gmod.namepara.elemsign[l] = 'flux'
        if gmod.typeelem[l] == 'lens':
            gmod.namepara.elemsign[l] = 'defs'
        if gmod.typeelem[l].startswith('clus'):
            gmod.namepara.elemsign[l] = 'nobj'
    
    # define the names and scalings of element parameters
    for l in gmod.indxpopl:
        
        # 'temp' these scale definitions are already done in setp_varb (which allows user-defined values) and redundant here. Should be removed.
        if gmod.typeelem[l].startswith('lghtline'):
            gmod.namepara.genrelem[l] = ['elin']
            gmod.scalpara.genrelem[l] = ['logt']
        elif gmod.typespatdist[l] == 'diskscal':
            gmod.namepara.genrelem[l] = ['xpos', 'ypos']
            gmod.scalpara.genrelem[l] = ['self', 'dexp']
        elif gmod.typespatdist[l] == 'gangexpo':
            gmod.namepara.genrelem[l] = ['gang', 'aang']
            gmod.scalpara.genrelem[l] = ['expo', 'self']
        elif gmod.typespatdist[l] == 'glc3':
            gmod.namepara.genrelem[l] = ['dglc', 'thet', 'phii']
            gmod.scalpara.genrelem[l] = ['powr', 'self', 'self']
        else:
            gmod.namepara.genrelem[l] = ['xpos', 'ypos']
            gmod.scalpara.genrelem[l] = ['self', 'self']
        
        # amplitude
        if gmod.typeelem[l] == 'lghtpntsagnntrue':
            gmod.namepara.genrelem[l] += ['lum0']
            gmod.scalpara.genrelem[l] += ['dpowslopbrek']
        elif gmod.typeelem[l] == 'lghtpntspuls':
            gmod.namepara.genrelem[l] += ['per0']
            gmod.scalpara.genrelem[l] += ['lnormeanstdv']
        elif gmod.typeelem[l].startswith('lght'):
            gmod.namepara.genrelem[l] += ['flux']
            gmod.scalpara.genrelem[l] += [gmod.typeprioflux[l]]
        elif gmod.typeelem[l] == 'lens':
            gmod.namepara.genrelem[l] += ['defs']
            gmod.scalpara.genrelem[l] += ['powr']
        elif gmod.typeelem[l].startswith('clus'):
            gmod.namepara.genrelem[l] += ['nobj']
            gmod.scalpara.genrelem[l] += ['powr']
       
        # shape
        if gmod.typeelem[l] == 'lghtgausbgrd' or gmod.typeelem[l] == 'clusvari':
            gmod.namepara.genrelem[l] += ['gwdt']
            gmod.scalpara.genrelem[l] += ['powr']
        if gmod.typeelem[l] == 'lghtlinevoig':
            gmod.namepara.genrelem[l] += ['sigm']
            gmod.scalpara.genrelem[l] += ['logt']
            gmod.namepara.genrelem[l] += ['gamm']
            gmod.scalpara.genrelem[l] += ['logt']
        
        # others
        if gmod.typeelem[l] == 'lghtpntspuls':
            gmod.namepara.genrelem[l] += ['magf']
            gmod.scalpara.genrelem[l] += ['lnormeanstdv']
            gmod.namepara.genrelem[l] += ['geff']
            gmod.scalpara.genrelem[l] += ['self']
        elif gmod.typeelem[l] == 'lghtpntsagnntrue':
            gmod.namepara.genrelem[l] += ['dlos']
            gmod.scalpara.genrelem[l] += ['powr']

        if gdat.numbener > 1 and gmod.typeelem[l].startswith('lghtpnts'):
            if gmod.spectype[l] == 'colr':
                for i in gdat.indxener:
                    if i == 0:
                        continue
                    gmod.namepara.genrelem[l] += ['sindcolr%04d' % i]
                    gmod.scalpara.genrelem[l] += ['self']
            else:
                gmod.namepara.genrelem[l] += ['sind']
                gmod.scalpara.genrelem[l] += ['self']
                if gmod.spectype[l] == 'curv':
                    gmod.namepara.genrelem[l] += ['curv']
                    gmod.scalpara.genrelem[l] += ['self']
                if gmod.spectype[l] == 'expc':
                    gmod.namepara.genrelem[l] += ['expc']
                    gmod.scalpara.genrelem[l] += ['self']
        if gmod.typeelem[l] == 'lens':
            if gdat.variasca:
                gmod.namepara.genrelem[l] += ['asca']
                gmod.scalpara.genrelem[l] += ['self']
            if gdat.variacut:
                gmod.namepara.genrelem[l] += ['acut']
                gmod.scalpara.genrelem[l] += ['self']
    
    # names of element parameters for each scaling
    gmod.namepara.genrelemscal = [{} for l in gmod.indxpopl]
    for l in gmod.indxpopl:
        for scaltype in gdat.listscaltype:
            gmod.namepara.genrelemscal[l][scaltype] = []
            for k, nameparagenrelem in enumerate(gmod.namepara.genrelem[l]):
                if scaltype == gmod.scalpara.genrelem[l][k]:
                    gmod.namepara.genrelemscal[l][scaltype].append(nameparagenrelem)

    # variables for which whose marginal distribution and pair-correlations will be plotted
    for l in gmod.indxpopl:
        gmod.namepara.derielemodim[l] = ['deltllik']
        
        if gmod.typeelem[l] == 'lens':
            gmod.namepara.derielemodim[l] += ['deflprof']
        
        if gdat.boolbindspat:
            # to be deleted
            #if not 'xpos' in gmod.namepara.derielemodim[l]:
            #    gmod.namepara.derielemodim[l] += ['xpos']
            #if not 'ypos' in gmod.namepara.derielemodim[l]:
            #    gmod.namepara.derielemodim[l] += ['ypos']
            
            if not 'gang' in gmod.namepara.derielemodim[l]:
                gmod.namepara.derielemodim[l] += ['gang']
            if not 'aang' in gmod.namepara.derielemodim[l]:
                gmod.namepara.derielemodim[l] += ['aang']
        
        if gmod.typeelem[l].startswith('lght'):
            gmod.namepara.derielemodim[l] += ['cnts']
            if gdat.typeexpr == 'ferm':
                gmod.namepara.derielemodim[l] + ['sbrt0018']
            
        if gmod.typeelem[l] == 'lghtpntsagnntrue':
            gmod.namepara.derielemodim[l] += ['reds']
            gmod.namepara.derielemodim[l] += ['lumi']
            gmod.namepara.derielemodim[l] += ['flux']
        if gmod.typeelem[l] == 'lghtpntspuls':
            gmod.namepara.derielemodim[l] += ['lumi']
            gmod.namepara.derielemodim[l] += ['flux']
            gmod.namepara.derielemodim[l] += ['mass']
            gmod.namepara.derielemodim[l] += ['dlos']
        if gmod.typeelem[l] == 'lens':
            gmod.namepara.derielemodim[l] += ['mcut', 'distsour', 'rele']#, 'reln', 'relk', 'relf', 'relm', 'reld', 'relc']
    
        #for k in range(len(gmod.namepara.derielemodim[l])):
        #    gmod.namepara.derielemodim[l][k] += 'pop%d' % l
        
        # check later
        # temp
        #if strgmodl == 'fitt':
        #    for q in gdat.indxrefr: 
        #        if gmod.nameparagenrelemampl[l] in gdat.refr.namepara.elem[q]:
        #            gmod.namepara.derielemodim[l].append('aerr' + gdat.listnamerefr[q])
    
    # derived parameters
    #gmod.listnameparaderitotl = [temptemp for temp in gmod.namepara.deri.elem for temptemp in temp]
    ##gmod.listnameparaderitotl += gmod.namepara.scal
    #
    #for namediff in gmod.listnamediff:
    #    gmod.listnameparaderitotl += ['cntp' + namediff]
    #
    #if gdat.typeverb > 1:
    #    print('gmod.listnameparaderitotl')
    #    print(gmod.listnameparaderitotl)

    if strgmodl == 'fitt':
        # add reference element parameters that are not available in the fitting model
        gdat.refr.namepara.elemonly = [[[] for l in gmod.indxpopl] for q in gdat.indxrefr]
        gmod.namepara.extrelem = [[] for l in gmod.indxpopl]
        for q in gdat.indxrefr: 
            if gdat.refr.numbelem[q] == 0:
                continue
            for name in gdat.refr.namepara.elem[q]:
                for l in gmod.indxpopl:
                    if gmod.typeelem[l].startswith('lght') and (name == 'defs' or name == 'acut' or name == 'asca' or name == 'mass'):
                        continue
                    if gmod.typeelem[l] == ('lens') and (name == 'cnts' or name == 'flux' or name == 'spec' or name == 'sind'):
                        continue
                    if not name in gmod.namepara.derielemodim[l]:
                        nametotl = name + gdat.listnamerefr[q]
                        if name == 'etag':
                            continue
                        gmod.namepara.derielemodim[l].append(nametotl)
                        
                        if gdat.refr.numbelem[q] == 0:
                            continue

                        gdat.refr.namepara.elemonly[q][l].append(name)
                        if not nametotl in gmod.namepara.extrelem[l]:
                            gmod.namepara.extrelem[l].append(nametotl) 
                        #if name == 'reds':
                        #    for nametemp in ['lumi', 'dlos']:
                        #        nametemptemp = nametemp + gdat.listnamerefr[q]
                        #        if not nametemptemp in gmod.namepara.extrelem[l]:
                        #            gmod.namepara.derielemodim[l].append(nametemp + gdat.listnamerefr[q])
                        #            gmod.namepara.extrelem[l].append(nametemptemp)
        
        if gdat.typeverb > 1:
            print('gdat.refr.namepara.elemonly')
            print(gdat.refr.namepara.elemonly)
    
        if gdat.typeexpr == 'chan' and gdat.typedata == 'inpt':
            for l in gmod.indxpopl:
                if gmod.typeelem[l] == 'lghtpnts':
                    gmod.namepara.extrelem[l].append('lumiwo08')
                    gmod.namepara.derielemodim[l].append('lumiwo08')
        
        if gdat.typeverb > 1:
            print('gmod.namepara.extrelem')
            print(gmod.namepara.extrelem)

    gmod.namepara.elemodim = [[] for l in gmod.indxpopl]
    for l in gmod.indxpopl:
        gmod.namepara.elemodim[l] += gmod.namepara.genrelem[l] + gmod.namepara.derielemodim[l]
    
    if gdat.typeverb > 1:
        print('gmod.namepara.derielemodim')
        print(gmod.namepara.derielemodim)
        print('gmod.namepara.elemodim')
        print(gmod.namepara.elemodim)
        
    if gdat.booldiag:
        if np.unique(np.array(gmod.namepara.derielemodim[l])).size != len(gmod.namepara.derielemodim[l]):
            print('gmod.namepara.derielemodim')
            print(gmod.namepara.derielemodim)
            raise Exception('')
        
        for name in gmod.namepara.derielemodim[l]:
            if name in gmod.namepara.genrelem[l]:
                print('gmod.namepara.derielemodim')
                print(gmod.namepara.derielemodim)
                print('gmod.namepara.genrelem')
                print(gmod.namepara.genrelem)
                raise Exception('')
    
    # defaults
    gmod.liststrgpdfnmodu = [[] for l in gmod.indxpopl]
    gmod.namepara.genrelemmodu = [[] for l in gmod.indxpopl]
    for l in gmod.indxpopl:
        if gmod.typeelem[l].startswith('lght'): 
            if gdat.typeexpr == 'ferm' and gdat.xposcntr == 0.:
                if l == 1:
                    gmod.liststrgpdfnmodu[l] += ['tmplnfwp']
                    gmod.namepara.genrelemmodu[l] += ['xposypos']
                if l == 2:
                    gmod.liststrgpdfnmodu[l] += ['tmplnfwp']
                    gmod.namepara.genrelemmodu[l] += ['xposypos']
    
    gmod.namepara.elem = [[] for l in gmod.indxpopl]
    for l in gmod.indxpopl:
        for liststrg in [gmod.namepara.genrelem[l], gmod.namepara.derielemodim[l]]:
            for strgthis in liststrg:
                if not strgthis in gmod.namepara.elem[l]:
                    gmod.namepara.elem[l].append(strgthis)
    
    # temp
    for l in gmod.indxpopl:
        if gmod.typeelem[l].startswith('lghtline'):
            gmod.namepara.genrelem[l] += ['spec']
        if gmod.typeelem[l].startswith('lght'):
            gmod.namepara.genrelem[l] += ['spec', 'specplot']
    
    #gmod.namepara.genr.elemeval = [[] for l in gmod.indxpopl]
    #for l in gmod.indxpopl:
    #    if gmod.typeelem[l].startswith('clus'):
    #        gmod.namepara.genr.elemeval[l] = ['xpos', 'ypos', 'nobj']
    #    if gmod.typeelem[l] == 'clusvari':
    #        gmod.namepara.genr.elemeval[l] += ['gwdt']
    #    if gmod.typeelem[l] == 'lens':
    #        gmod.namepara.genr.elemeval[l] = ['xpos', 'ypos', 'defs', 'asca', 'acut']
    #    if gmod.typeelem[l].startswith('lghtline'):
    #        gmod.namepara.genr.elemeval[l] = ['elin', 'spec']
    #    elif gmod.typeelem[l] == 'lghtgausbgrd':
    #        gmod.namepara.genr.elemeval[l] = ['xpos', 'ypos', 'gwdt', 'spec']
    #    elif gmod.typeelem[l].startswith('lght'):
    #        gmod.namepara.genr.elemeval[l] = ['xpos', 'ypos', 'spec']
    
    ## element legends
    lablpopl = [[] for l in gmod.indxpopl]
    for l in gmod.indxpopl:
        if gdat.numbgrid > 1:
            if gmod.typeelem[l] == 'lghtpnts':
                lablpopl[l] = 'FPS'
            if gmod.typeelem[l] == 'lghtgausbgrd':
                lablpopl[l] = 'BGS'
        else:
            if gmod.typeelem[l] == 'lghtpntspuls':
                lablpopl[l] = 'Pulsar'
            elif gmod.typeelem[l].startswith('lghtpntsagnn'):
                lablpopl[l] = 'AGN'
            elif gmod.typeelem[l].startswith('lghtpnts'):
                lablpopl[l] = 'PS'
        if gmod.typeelem[l] == 'lens':
            lablpopl[l] = 'Subhalo'
        if gmod.typeelem[l].startswith('clus'):
            lablpopl[l] = 'Cluster'
        if gmod.typeelem[l].startswith('lghtline'):
            lablpopl[l]= 'Line'
    setp_varb(gdat, 'lablpopl', valu=lablpopl, strgmodl=strgmodl)
    
    for l in gmod.indxpopl:
        setp_varb(gdat, 'slopprionobjpop%d' % l, labl=['$\alpha_{%d}$' % l, ''], strgmodl=strgmodl, popl=l)

    if strgmodl == 'true':
        gmod.indxpoplassc = [[] for l in gmod.indxpopl]
        for l in gmod.indxpopl:
            if gmod.numbpopl == 3 and gmod.typeelem[1] == 'lens':
                gmod.indxpoplassc[l] = [l]
            else:
                gmod.indxpoplassc[l] = gmod.indxpopl

    # number of element parameters
    if gmod.numbpopl > 0:
        gmod.numbparagenrelemsing = np.zeros(gmod.numbpopl, dtype=int)
        gmod.numbparagenrelempopl = np.zeros(gmod.numbpopl, dtype=int)
        gmod.numbparagenrelemcuml = np.zeros(gmod.numbpopl, dtype=int)
        gmod.numbparagenrelemcumr = np.zeros(gmod.numbpopl, dtype=int)
        gmod.numbparaderielempopl = np.zeros(gmod.numbpopl, dtype=int)
        gmod.numbparaderielemsing = np.zeros(gmod.numbpopl, dtype=int)
        gmod.numbparaelempopl = np.zeros(gmod.numbpopl, dtype=int)
        gmod.numbparaelemsing = np.zeros(gmod.numbpopl, dtype=int)
        for l in gmod.indxpopl:
            # number of generative element parameters for a single element of a specific population
            gmod.numbparagenrelemsing[l] = len(gmod.namepara.genrelem[l])
            # number of derived element parameters for a single element of a specific population
            gmod.numbparaderielemsing[l] = len(gmod.namepara.derielemodim[l])
            # number of element parameters for a single element of a specific population
            gmod.numbparaelemsing[l] = len(gmod.namepara.elem[l])
            # number of generative element parameters for all elements of a specific population
            gmod.numbparagenrelempopl[l] = gmod.numbparagenrelemsing[l] * gmod.maxmpara.numbelem[l]
            # number of generative element parameters up to the beginning of a population
            gmod.numbparagenrelemcuml[l] = np.sum(gmod.numbparagenrelempopl[:l])
            # number of generative element parameters up to the end of a population
            gmod.numbparagenrelemcumr[l] = np.sum(gmod.numbparagenrelempopl[:l+1])
            # number of derived element parameters for all elements of a specific population
            gmod.numbparaderielempopl[l] = gmod.numbparaderielemsing[l] * gmod.maxmpara.numbelem[l]
            # number of element parameters for all elements of a specific population
            gmod.numbparaelempopl[l] = gmod.numbparaelemsing[l] * gmod.maxmpara.numbelem[l]
        # number of generative element parameters summed over all populations
        gmod.numbparagenrelem = np.sum(gmod.numbparagenrelempopl)
        # number of derived element parameters summed over all populations
        gmod.numbparaderielem = np.sum(gmod.numbparaderielempopl)
        # number of element parameters summed over all populations
        gmod.numbparaelemtotl = np.sum(gmod.numbparaelempopl)
    
        gmod.indxparagenrelemsing = []
        for l in gmod.indxpopl:
            gmod.indxparagenrelemsing.append(np.arange(gmod.numbparagenrelemsing[l]))
        
        gmod.indxparaderielemsing = []
        for l in gmod.indxpopl:
            gmod.indxparaderielemsing.append(np.arange(gmod.numbparaderielemsing[l]))
        
        gmod.indxparaelemsing = []
        for l in gmod.indxpopl:
            gmod.indxparaelemsing.append(np.arange(gmod.numbparaelemsing[l]))
        
    # size of the auxiliary variable propobability density vector
    if gmod.maxmpara.numbelemtotl > 0:
        gmod.numblpri = 1
        #+ gmod.numbparagenrelempopl * gmod.numbpopl
    else:
        gmod.numblpri = 0
    if gdat.boolpenalpridiff:
        gmod.numblpri += 1
    indxlpri = np.arange(gmod.numblpri)

    # append the population tags to element parameter names
    #for l in gmod.indxpopl:
    #    gmod.namepara.genrelem[l] = [gmod.namepara.genrelem[l][g] + 'pop%d' % l for g in gmod.indxparagenrelemsing[l]]
    
    # Boolean flag indicating if a generative element parameter is "positional"
    gmod.boolcompposi = [[] for l in gmod.indxpopl]
    for l in gmod.indxpopl:
        gmod.boolcompposi[l] = np.zeros(gmod.numbparagenrelemsing[l], dtype=bool)
        if gmod.typeelem[l].startswith('lghtline'):
            gmod.boolcompposi[l][0] = True
        else:
            gmod.boolcompposi[l][0] = True
            gmod.boolcompposi[l][1] = True
    
    # flattened list of element parameters
    for strgtypepara in ['genrelem', 'derielemodim', 'elemodim']:
        nameparaelem = getattr(gmod.namepara, strgtypepara)
        setattr(gmod.namepara, strgtypepara + 'flat', [])
        nameparaelemflat = getattr(gmod.namepara, strgtypepara + 'flat')
        for l in gmod.indxpopl:
            for nameparaelemodim in nameparaelem[l]:
                nameparaelemflat.append(nameparaelemodim + 'pop%d' % l)
    
    #gmod.numbparaelem = np.empty(gmod.numbpopl, dtype=int)
    #for l in gmod.indxpopl:
    #    gmod.numbparaelem[l] = len(gmod.namepara.elem[l])
    
    gmod.numbdeflsingplot = gdat.numbdeflsubhplot
    if gmod.numbpopl > 0:
        gmod.numbdeflsingplot += 3

    gmod.convdiffanyy = True in convdiff

    cntr = tdpy.cntr()
    
    # collect lensing here
    if gmod.boollens:
        if gmod.boollenshost:
            setp_varb(gdat, 'redshost', valu=0.2, minm=0., maxm=0.4, strgmodl=strgmodl)
            setp_varb(gdat, 'redssour', valu=1., minm=0.5, maxm=1.5, strgmodl=strgmodl)

        gmod.adislens = gdat.adisobjt(gmod.redshost)
        gmod.adissour = gdat.adisobjt(gmod.redssour)
        gmod.adislenssour = gmod.adissour - (1. + gmod.redshost) / (1. + gmod.redssour) * gmod.adislens
        gmod.ratimassbeinsqrd = chalcedon.retr_ratimassbeinsqrd(gmod.adissour, gmod.adislens, gmod.adislenssour)
        gmod.mdencrit = chalcedon.retr_mdencrit(gmod.adissour, gmod.adislens, gmod.adislenssour)
    
        gmod.bctrpara.adislens = gdat.adisobjt(gmod.bctrpara.redshost)
        
    
    # base parameter indices (indxpara) are being defined here
    # define parameter indices
    print('Defining the indices of individual parameters...')
    if gmod.numbpopl > 0:

        # number of elements
        for l in gmod.indxpopl:
            indx = cntr.incr()
            setattr(gmod.indxpara, 'numbelempop%d' % l, indx)
        
        # hyperparameters
        ## mean number of elements
        if gmod.typemodltran == 'pois':
            #gmod.indxpara.meanelem = np.empty(gmod.numbpopl, dtype=int)
            for l in gmod.indxpopl:
                if gmod.maxmpara.numbelem[l] > 0:
                    indx = cntr.incr()
                    print('meanelempop%d % l')
                    print('meanelempop%d' % l)
                    print('indx')
                    print(indx)
                    setattr(gmod.indxpara, 'meanelempop%d' % l, indx)
                    #gmod.indxpara.meanelem[l] = indx

        ## parameters parametrizing priors on element parameters
        liststrgvarb = []
        for l in gmod.indxpopl:
            if gmod.maxmpara.numbelem[l] > 0:
                for strgpdfnelemgenr, strgfeat in zip(gmod.scalpara.genrelem[l], gmod.namepara.genrelem[l]):
                    liststrgvarb = []
                    if strgpdfnelemgenr == 'expo' or strgpdfnelemgenr == 'dexp':
                        liststrgvarb += [strgfeat + 'distscal']
                    if strgpdfnelemgenr == 'powr':
                        liststrgvarb += ['slopprio' + strgfeat]
                    if strgpdfnelemgenr == 'dpow':
                        liststrgvarb += [strgfeat + 'distbrek', strgfeat + 'sloplowr', strgfeat + 'slopuppr']
                    if strgpdfnelemgenr == 'gausmean' or strgpdfnelemgenr == 'lnormean':
                        liststrgvarb += [strgfeat + 'distmean']
                    if strgpdfnelemgenr == 'gausstdv' or strgpdfnelemgenr == 'lnorstdv':
                        liststrgvarb += [strgfeat + 'diststdv']
                    if strgpdfnelemgenr == 'gausmeanstdv' or strgpdfnelemgenr == 'lnormeanstdv':
                        liststrgvarb += [strgfeat + 'distmean', strgfeat + 'diststdv']
                    for strgvarb in liststrgvarb:
                        indx = cntr.incr()
                        setattr(gmod.indxpara, strgvarb + 'pop%d' % l, indx)
            #setattr(gmod.indxpara, strgvarb, np.zeros(gmod.numbpopl, dtype=int) - 1)

        # can potentially be used to define arrays of parameter indices for a given hyperparameter across all populations
        ## this is where 'slopprionobj'-like definitions happen
        ## turned off
        #for l in gmod.indxpopl:
        #    if gmod.maxmpara.numbelem[l] > 0:
        #        for k, nameparagenrelem in enumerate(gmod.namepara.genrelem[l]):
        #            
        #            if gmod.scalpara.genrelem[l][k] == 'self':
        #                continue
        #            indx = cntr.incr()

        #            if gmod.scalpara.genrelem[l][k] == 'dpow':
        #                for nametemp in ['brek', 'sloplowr', 'slopuppr']:
        #                    strg = '%s' % nametemp + nameparagenrelem
        #                    setattr(gmod.indxpara, strg, indx)
        #                    setattr(gmod.indxpara, strg, indx)
        #            else:
        #                if gmod.scalpara.genrelem[l][k] == 'expo' or gmod.scalpara.genrelem[l][k] == 'dexp':
        #                    strghypr = 'scal'
        #                if gmod.scalpara.genrelem[l][k] == 'powr':
        #                    strghypr = 'slop'
        #                if gmod.scalpara.genrelem[l][k] == 'gausmean' or gmod.scalpara.genrelem[l][k] == 'gausmeanstdv' or \
        #                                gmod.scalpara.genrelem[l][k] == 'lnormean' or gmod.scalpara.genrelem[l][k] == 'lnormeanstdv':
        #                    strghypr = 'mean'
        #                if gmod.scalpara.genrelem[l][k] == 'gausstdv' or gmod.scalpara.genrelem[l][k] == 'gausmeanstdv' or \
        #                                gmod.scalpara.genrelem[l][k] == 'lnorstdv' or gmod.scalpara.genrelem[l][k] == 'lnormeanstdv':
        #                    strghypr = 'stdv'
        #                strg = strghypr + 'prio' + nameparagenrelem
        #                print('strg')
        #                print(strg)
        #                setattr(gmod.indxpara, strg, indx)
        
        #raise Exception('')
    
    # parameter groups 
    ## number element elements for all populations
    if gmod.numbpopl > 0:
        gmod.indxpara.numbelem = np.empty(gmod.numbpopl, dtype=int)
        for l in gmod.indxpopl:
            gmod.indxpara.numbelem[l] = indx
    
    ## PSF parameters
    if gmod.typeevalpsfn != 'none':
        for m in gdat.indxdqlt:
            for i in gdat.indxener:
                setattr(gmod.indxpara, 'sigcen%02devt%d' % (i, m), cntr.incr())
                if gmod.typemodlpsfn == 'doubking' or gmod.typemodlpsfn == 'singking':
                    setattr(gmod.indxpara, 'gamcen%02devt%d' % (i, m), cntr.incr())
                    if gmod.typemodlpsfn == 'doubking':
                        setattr(gmod.indxpara, 'sigten%02devt%d' % (i, m), cntr.incr())
                        setattr(gmod.indxpara, 'gamten%02devt%d' % (i, m), cntr.incr())
                        setattr(gmod.indxpara, 'ffenen%02devt%d' % (i, m), cntr.incr())
        
        gmod.indxpara.psfp = []
        for strg, valu in gmod.indxpara.__dict__.items():
            if strg.startswith('sigce') or strg.startswith('sigte') or strg.startswith('gamce') or strg.startswith('gamte') or strg.startswith('psffe'):
                gmod.indxpara.psfp.append(valu)
        gmod.indxpara.psfp = np.array(gmod.indxpara.psfp) 

        gmod.numbpsfptotldqlt = gdat.numbdqlt * gmod.numbpsfptotl
        gmod.numbpsfptotlener = gdat.numbener * gmod.numbpsfptotl
        numbpsfp = gmod.numbpsfptotl * gdat.numbener * gdat.numbdqlt
        indxpsfpform = np.arange(numbpsfpform)
        indxpsfptotl = np.arange(gmod.numbpsfptotl)
   
        gmod.indxpara.psfp = np.sort(gmod.indxpara.psfp)

    ## background parameters
    gmod.indxpara.bacp = []
    for c in gmod.indxback:
        if gmod.boolspecback[c]:
            indx = cntr.incr()
            setattr(gmod.indxpara, 'bacpback%04d' % c, indx)
            gmod.indxpara.bacp.append(indx)
        else:
            for i in gdat.indxener:
                indx = cntr.incr()
                setattr(gmod.indxpara, 'bacpback%04den%02d' % (c, i), indx)
                gmod.indxpara.bacp.append(indx)
    gmod.indxpara.bacp = np.array(gmod.indxpara.bacp)

    # temp
    #gmod.indxpara.anglsour = []
    #gmod.indxpara.anglhost = []
    #gmod.indxpara.angllens = []
    
    if gmod.typeemishost != 'none':
        gmod.indxpara.specsour = []
        gmod.indxpara.spechost = []

    if gmod.boollens:
        gmod.indxpara.xpossour = cntr.incr()
        gmod.indxpara.ypossour = cntr.incr()
        gmod.indxpara.fluxsour = cntr.incr()
        if gdat.numbener > 1:
            gmod.indxpara.sindsour = cntr.incr()
        gmod.indxpara.sizesour = cntr.incr()
        gmod.indxpara.ellpsour = cntr.incr()
        gmod.indxpara.anglsour = cntr.incr()
    if gmod.typeemishost != 'none' or gmod.boollens:
        for e in gmod.indxsersfgrd: 
            if gmod.typeemishost != 'none':
                setattr(gmod.indxpara, 'xposhostisf%d' % e, cntr.incr())
                setattr(gmod.indxpara, 'yposhostisf%d' % e, cntr.incr())
                setattr(gmod.indxpara, 'fluxhostisf%d' % e, cntr.incr())
                if gdat.numbener > 1:
                    setattr(gmod.indxpara, 'sindhostisf%d' % e, cntr.incr())
                setattr(gmod.indxpara, 'sizehostisf%d' % e, cntr.incr())
            if gmod.boollens:
                setattr(gmod.indxpara, 'beinhostisf%d' % e, cntr.incr())
            if gmod.typeemishost != 'none':
                setattr(gmod.indxpara, 'ellphostisf%d' % e, cntr.incr())
                setattr(gmod.indxpara, 'anglhostisf%d' % e, cntr.incr())
                setattr(gmod.indxpara, 'serihostisf%d' % e, cntr.incr())
    if gmod.boollens:
        gmod.indxpara.sherextr = cntr.incr()
        gmod.indxpara.sangextr = cntr.incr()
        gmod.indxpara.sour = []
    
    if gmod.boollens and gmod.typeemishost == 'none':
        raise Exception('Lensing cannot be modeled without host galaxy emission.')
    
    # collect groups of parameters
    if gdat.typeexpr.startswith('HST_WFC3'):
        gmod.listnamecomplens = ['hostlght', 'hostlens', 'sour', 'extr']
        for namecomplens in gmod.listnamecomplens:
            setattr(gmod, 'liststrg' + namecomplens, [])
            setattr(gmod.indxpara, namecomplens, [])
    if gmod.boollens or gmod.typeemishost != 'none':
        gmod.liststrghostlght += ['xposhost', 'yposhost', 'ellphost', 'anglhost']
        gmod.liststrghostlens += ['xposhost', 'yposhost', 'ellphost', 'anglhost']
    if gmod.typeemishost != 'none':
        gmod.liststrghostlght += ['fluxhost', 'sizehost', 'serihost']
        if gdat.numbener > 1:
            gmod.liststrghostlght += ['sindhost']
    if gmod.boollens:
        gmod.liststrghostlens += ['beinhost']
        gmod.liststrgextr += ['sherextr', 'sangextr']
        gmod.liststrgsour += ['xpossour', 'ypossour', 'fluxsour', 'sizesour', 'ellpsour', 'anglsour']
        if gdat.numbener > 1:
            gmod.liststrgsour += ['sindsour']
    
    for strg, valu in gmod.__dict__.items():
        
        if isinstance(valu, list) or isinstance(valu, np.ndarray):
            continue
        
        if gdat.typeexpr.startswith('HST_WFC3'):
            for namecomplens in gmod.listnamecomplens:
                for strgtemp in getattr(gmod, 'liststrg' + namecomplens):
                    if strg[12:].startswith(strgtemp):
                        
                        if isinstance(valu, list):
                            for valutemp in valu:
                                gmod['indxparagenr' + namecomplens].append(valutemp)
                        else:
                            gmod['indxparagenr' + namecomplens].append(valu)
        
        # remove indxpara. from strg
        strg = strg[12:]
        
        if strg.startswith('fluxsour') or strg.startswith('sindsour'):
            gmod.indxpara.specsour.append(valu)

        if strg.startswith('fluxhost') or strg.startswith('sindhost'):
            gmod.indxpara.spechost.append(valu)
    
    if gmod.boollens or gmod.boolhost:
        gmod.indxpara.host = gmod.indxpara.hostlght + gmod.indxpara.hostlens
        gmod.indxpara.lens = gmod.indxpara.host + gmod.indxpara.sour + gmod.indxpara.extr

    ## number of model spectral parameters for each population
    #numbspep = np.empty(gmod.numbpopl, dtype=int)
    #liststrgspep = [[] for l in range(gmod.numbpopl)]
    #for l in gmod.indxpopl:
    #    if gdat.numbener > 1:
    #        liststrgspep[l] += ['sind']
    #        if gmod.spectype[l] == 'expc':
    #            liststrgspep[l] += ['expc']
    #        if gmod.spectype[l] == 'curv':
    #            liststrgspep[l] = ['curv']
    #    numbspep[l] = len(liststrgspep[l]) 
        

def eval_emislens( \
                  # grid
                  xposgrid=None, \
                  yposgrid=None, \

                  # input dictionary
                  dictchalinpt=None, \
                  
                  gmod=None, \

                  # number of pixels on a side
                  numbsidecart=None, \

                  # type of verbosity
                  ## -1: absolutely no text
                  ##  0: no text output except critical warnings
                  ##  1: minimal description of the execution
                  ##  2: detailed description of the execution
                  typeverb=1, \
                  
                  # Boolean flag to turn on diagnostic mode
                  booldiag=True, \
                 ):
    
    '''
    Calculate the emission due to graviationally lensed-sources background sources
    '''
    timeinit = modutime.time()
    
    if gmod is not None:
        gmod = setp_modlemis_init(gdat, strgmodl)
        gmod = setp_modlemis_finl(gdat, strgmodl)

    # construct global object
    gdat = tdpy.gdatstrt()
    
    # copy locals (inputs) to the global object
    dictinpt = dict(locals())
    for attr, valu in dictinpt.items():
        if '__' not in attr and attr != 'gdat':
            setattr(gdat, attr, valu)

    if dictchalinpt is not None:
        pass
    
    # output dictionary
    dictchaloutp = dict()
    
    if dictchalinpt is None:
        dictchalinpt = dict()
    
    #objttimeprof = tdpt.retr_objttimeprof('emislens')

    #objttimeprof.initchro(gdat, gdatmodi, 'elem')
    # grab the sample vector
    #indxpara = np.arange(paragenr.size) 
    
    boollenssubh = True

    if not 'listnamesersfgrd' in dictchalinpt:
        dictchalinpt['listnamesersfgrd'] = []

    if not 'typeemishost' in dictchalinpt:
        dictchalinpt['typeemishost'] = 'Sersic'

    if not 'numbener' in dictchalinpt:
        dictchalinpt['numbener'] = 1
    
    boolneedpsfnintp = True
    
    boolcalcpsfnintp = True

    # process a sample vector and the occupancy list to calculate secondary variables

    numbsersfgrd = len(dictchalinpt['listnamesersfgrd'])
    indxsersfgrd = np.arange(numbsersfgrd)
    beinhost = [[] for e in indxsersfgrd]
    for e in indxsersfgrd:
        beinhost[e] = paragenr[getattr(indxpara, 'beinhostisf%d' % e)]
    
    # maybe to be deleted
    #if dictchalinpt['typeemishost'] != 'none':
    #    xposhost = [[] for e in indxsersfgrd]
    #    yposhost = [[] for e in indxsersfgrd]
    #    fluxhost = [[] for e in indxsersfgrd]
    #    if dictchalinpt['numbener'] > 1:
    #        sindhost = [[] for e in indxsersfgrd]
    #    sizehost = [[] for e in indxsersfgrd]
    #    for e in indxsersfgrd:
    #        xposhost[e] = paragenr[getattr(indxpara, 'xposhostisf%d' % e)]
    #        yposhost[e] = paragenr[getattr(indxpara, 'yposhostisf%d' % e)]
    #        fluxhost[e] = paragenr[getattr(indxpara, 'fluxhostisf%d' % e)]
    #        if dictchalinpt['numbener'] > 1:
    #            sindhost[e] = paragenr[getattr(indxpara, 'sindhostisf%d' % e)]
    #        sizehost[e] = paragenr[getattr(indxpara, 'sizehostisf%d' % e)]
    #    ellphost = [[] for e in indxsersfgrd]
    #    anglhost = [[] for e in indxsersfgrd]
    #    serihost = [[] for e in indxsersfgrd]
    #    for e in indxsersfgrd:
    #        ellphost[e] = paragenr[getattr(indxpara, 'ellphostisf%d' % e)]
    #        anglhost[e] = paragenr[getattr(indxpara, 'anglhostisf%d' % e)]
    #        serihost[e] = paragenr[getattr(indxpara, 'serihostisf%d' % e)]
    
    ## host halo deflection
    #objttimeprof.initchro(gdat, gdatmodi, 'deflhost')
    deflhost = [[] for e in indxsersfgrd]
    
    if numbsidecart is None:
        numbsidecart = 100
        maxmfovw = 2.
        xposside = np.linspace(-maxmfovw, maxmfovw, 100)
        yposside = xposside
        xposgrid, yposgrid = np.meshgrid(xposside, yposside, indexing='ij')
        xposgridflat = xposgrid.flatten()
        yposgridflat = yposgrid.flatten()
        
    numbpixl = numbsidecart**2
    indxpixl = np.arange(numbpixl)
    
    numbsubh = dictchalinpt['xpossubh'].size
    indxsubh = np.arange(numbsubh)

    defl = np.zeros((numbpixl, 2))
        
    for e in indxsersfgrd:
        
        dictchalinpt['xposhost'] = xposhost[e]
        dictchalinpt['yposhost'] = yposhost[e]
        dictchalinpt['beinhost'] = beinhost[e]
        dictchalinpt['ellphost'] = ellphost[e]
        dictchalinpt['anglhost'] = anglhost[e]
        deflhost[e] = chalcedon.retr_defl(gdat.xposgrid, gdat.yposgrid, indxpixl, dictchalinpt)
         
        if gdat.booldiag:
            if not np.isfinite(deflhost[e]).all():
                print('')
                print('')
                print('')
                raise Exception('not np.isfinite(deflhost[e]).all()')
    
        if gdat.booldiag:
            indxpixltemp = slice(None)
        
        if typeverb > 1:
            print('deflhost[e]')
            summgene(deflhost[e])
            
        defl += deflhost[e]
        if typeverb > 1:
            print('After adding the host deflection...')
            print('defl')
            summgene(defl)
    
    #objttimeprof.stopchro(gdat, gdatmodi, 'deflhost')

    ## external shear
    #objttimeprof.initchro(gdat, gdatmodi, 'deflextr')
    deflextr = chalcedon.retr_deflextr(xposgridflat, yposgridflat, dictchalinpt['sherextr'], dictchalinpt['sangextr'])
    defl += deflextr
    
    if typeverb > 1:
        print('After adding the external deflection...')
        print('defl')
        summgene(defl)
    
    #objttimeprof.stopchro(gdat, gdatmodi, 'deflextr')
    
    typeevalpsfn = 'full'

    boolneedpsfnconv = typeevalpsfn == 'conv' or typeevalpsfn == 'full'
    boolcalcpsfnconv = boolneedpsfnconv
    
    typepixl = 'cart'
    
    kernevaltype = 'ulip'

    sizepixl = 0.11 # [arcsec]

    numbdqlt = 1
    indxdqlt = np.arange(numbdqlt)
    numbener = 1
    indxener = np.arange(numbener)
    
    arryangl = np.linspace(0.1, 2., 100)
    
    typemodlpsfn = 'singgaus'
    
    dictpara = dict()

    dictpara['sigc'] = np.array([[1.]])

    if boolneedpsfnconv:
        
        #objttimeprof.initchro(gdat, gdatmodi, 'psfnconv')
        
        # compute the PSF convolution object
        if boolcalcpsfnconv:
            objtpsfnconv = [[[] for i in indxener] for m in indxdqlt]
            psfn = nicomedia.retr_psfn(dictpara, indxener, arryangl, typemodlpsfn)
            fwhm = 2. * nicomedia.retr_psfnwdth(psfn, arryangl, 0.5)
            for mm, m in enumerate(indxdqlt):
                for ii, i in enumerate(indxener):
                    if typemodlpsfn == 'singgaus':
                        sigm = dictpara['sigc'][i, m]
                    else:
                        sigm = fwhm[i, m] / 2.355
                    objtpsfnconv[mm][ii] =  astropy.convolution.AiryDisk2DKernel(sigm / sizepixl)
            
        #objttimeprof.stopchro(gdat, gdatmodi, 'psfnconv')
    
    
    if boolneedpsfnintp:
        
        # compute the PSF interpolation object
        if boolcalcpsfnintp:
            if typepixl == 'heal':
                psfn = nicomedia.retr_psfn(dictpara, indxener, arryangl, typemodlpsfn)
                psfnintp = scipy.interpolate.interp1d(arryangl, psfn, axis=1, fill_value='extrapolate')
                fwhm = 2. * nicomedia.retr_psfnwdth(gdat, arryangl, psfn, 0.5)
            
            elif typepixl == 'cart':
                if kernevaltype == 'ulip':
                    print('arryangl')
                    summgene(arryangl)
                    psfn = nicomedia.retr_psfn(dictpara, indxener, arryangl, typemodlpsfn)
                    psfnintp = scipy.interpolate.interp1d(arryangl, psfn, axis=1, fill_value='extrapolate')

                if kernevaltype == 'bspx':
                    
                    psfn = nicomedia.retr_psfn(dictpara, indxener, arryanglcart.flatten(), typemodlpsfn)
                    
                    # side length of the upsampled kernel
                    numbsidekernusam = 100
                    # side length of the original kernel
                    numbsidekern = numbsidekernusam / factkernusam 
                    indxsidekern = np.arange(numbsidekern)

    	        	# pad by one row and one column
    	        	#psf = np.zeros((numbsidekernusam+1, numbsidekernusam+1))
    	        	#psf[0:numbsidekernusam, 0:numbsidekernusam] = psf0
		        	
    	        	# make design matrix for each factkernusam x factkernusam region
                    nx = factkernusam + 1
                    y, x = mgrid[0:nx, 0:nx] / float(factkernusam)
                    x = x.flatten()
                    y = y.flatten()
                    kernmatrdesi = np.array([full(nx*nx, 1), x, y, x*x, x*y, y*y, x*x*x, x*x*y, x*y*y, y*y*y]).T
    	        	
                    # output np.array of coefficients
                    psfnintp = np.empty((numbsidekern, numbsidekern, kernmatrdesi.shape[1]))

    	        	# solve p = kernmatrdesi psfnintp for psfnintp
                    for iy in indxsidekern:
                        for ix in indxsidekern:
                            p = psf[iy*factkernusam:(iy+1)*factkernusam+1, ix*factkernusam:(ix+1)*factkernusam+1].flatten()
                            psfnintp[iy, ix, :] = dot(linalg.inv(dot(kernmatrdesi.T, kernmatrdesi)), dot(kernmatrdesi.T, p))
        else:
            psfnintp = gdat.fitt.this.psfnintp
    
        if gdat.booldiag:
            if not np.isfinite(psfnintp(0.05)).all():
                print('')
                print('')
                print('')
                raise Exception('')
    
    if boollenssubh:
        deflsubh = np.zeros((numbpixl, 2))
    
        #objttimeprof.initchro(gdat, gdatmodi, 'elemdeflsubh')
        
        if typeverb > 1:
            print('Perturbing subhalo deflection field')
        if boollenssubh:
            for k in indxsubh:
                asca = dictchalinpt['ascasubh'][k]
                acut = dictchalinpt['acutsubh'][k]
                if False and typeelemspateval[l] == 'locl':
                    indxpixltemp = listindxpixlelem[l][k]
                else:
                    indxpixltemp = indxpixl
                deflsubh[indxpixl, :] += chalcedon.retr_defl(xposgridflat, yposgridflat, indxpixltemp, dictchalinpt)
        
            # temp -- find out what is causing the features in the element convergence maps
            #for k, k in enumerate(indxelem[l]):
            #    indxpixlpnts = retr_indxpixl(gdat, dictchalinpt['ypos'][k], dictchalinpt['xpos'][k])
            #    if deflsubh[listindxpixlelem[l][k], :]
        
        if gdat.booldiag:
            if not np.isfinite(deflsubh).all():
                raise Exception('Element deflection is not finite.')

        defl += deflsubh
        if typeverb > 1:
            print('After adding subhalo deflection to the total deflection')
            print('defl')
            summgene(defl)

        
        #objttimeprof.stopchro(gdat, gdatmodi, 'elemdeflsubh')

    # evaluate surface brightnesses
    sbrt = dict()
    for name in listnamediff:
        sbrt[name] = []
    
    ## due to elements
    if boolelemsbrtdfncanyy:
        sbrtdfnc = np.zeros_like(gdat.expo)
    if boolelemsbrtextsbgrdanyy: 
        sbrtextsbgrd = np.zeros_like(gdat.expo)
    
    # element kernel evaluation
    if boolelemsbrtdfncanyy:
        #objttimeprof.initchro(gdat, gdatmodi, 'elemsbrtdfnc')
        sbrt['dfnc'] = []
        if boolelemsbrtdfnc[l]:
            for k in range(numbelem[l]):
                if boolelemlght[l]:
                    varbamplextd = dictchalinpt['spec'][:, k]
                if typeelem[l].startswith('clus'):
                    varbamplextd = dictchalinpt['nobj'][None, k]
                if typeelem[l] == 'clusvari':
                    sbrtdfnc[0, listindxpixlelem[l][k], 0] += dictchalinpt['nobj'][k] / 2. / np.pi / dictchalinpt['gwdt'][k]**2 * \
                        np.exp(-0.5 * ((dictchalinpt['xpos'][k] - gdat.xposgrid[listindxpixlelem[l][k]])**2 + \
                            (dictchalinpt['ypos'][k] - gdat.yposgrid[listindxpixlelem[l][k]])**2) / dictchalinpt['gwdt'][k]**2)
                    
                if boolelempsfn[l]:
                    sbrtdfnc[:, listindxpixlelem[l][k], :] += retr_sbrtpnts(gdat, dictchalinpt['xpos'][k], \
                                                     dictchalinpt['ypos'][k], varbamplextd, psfnintp, listindxpixlelem[l][k])
                
                if typeelem[l].startswith('lghtline'):
                    sbrtdfnc[:, 0, 0] += dictchalinpt['spec'][:, k]
                    
        sbrt['dfnc'] = sbrtdfnc
        
        #setattr(gmodstat, 'sbrtmodlconv', sbrt['modlconv'])
        #    setattr(gmodstat, 'sbrthostisf%d' % e, sbrt['hostisf%d' % e])
        #setattr(gmodstat, 'sbrtlens', sbrt['lens'])
        #setattr(gmodstat, 'deflhostisf%d' % e, deflhost[e])
        #setattr(gmodstat, 'deflsubh', deflsubh)
        #setattr(gmodstat, 'sbrtdfnc', sbrt['dfnc'])
        #    setattr(gmodstat, 'sbrtextsbgrd', sbrtextsbgrd)
        #objttimeprof.stopchro(gdat, gdatmodi, 'elemsbrtdfnc')
        
        if gdat.booldiag:
            if not np.isfinite(sbrtdfnc).all():
                raise Exception('Element delta function brightness not finite.')

            cntppntschec = retr_cntp(gdat, sbrt['dfnc'])
            numbelemtemp = 0
            if boolelemsbrtdfnc[l]:
                numbelemtemp += np.sum(numbelem[l])
            if np.amin(cntppntschec) < -0.1:
                raise Exception('Point source spectral surface brightness is not positive-definite.')
        
    
    if boolelemsbrtextsbgrdanyy:
        #objttimeprof.initchro(gdat, gdatmodi, 'elemsbrtextsbgrd')
        if strgstat == 'this':
            if typeelem[l] == 'lghtgausbgrd':
                for k in range(numbelem[l]):
                    sbrtextsbgrd[:, listindxpixlelem[l][k], :] += dictchalinpt['spec'][:, k, None, None] / \
                            2. / np.pi / dictchalinpt['gwdt'][k]**2 * \
                            np.exp(-0.5 * ((dictchalinpt['xpos'][k] - gdat.xposgrid[None, listindxpixlelem[l][k], None])**2 + \
                            (dictchalinpt['ypos'][k] - gdat.yposgrid[None, listindxpixlelem[l][k], None])**2) / dictchalinpt['gwdt'][k]**2)
            
        sbrt['extsbgrd'] = []
        sbrt['extsbgrd'] = sbrtextsbgrd
        #objttimeprof.stopchro(gdat, gdatmodi, 'elemsbrtextsbgrd')
        
        if gdat.booldiag:
            cntppntschec = retr_cntp(gdat, sbrt['extsbgrd'])
            if np.amin(cntppntschec) < -0.1:
                raise Exception('Point source spectral surface brightness is not positive-definite.')
    
    
    ## lensed surface brightness
    if boollens:
        
        #objttimeprof.initchro(gdat, gdatmodi, 'sbrtlens')
        
        if typeverb > 1:
            print('Evaluating lensed surface brightness...')
        
        if strgstat == 'this' or boolelemsbrtextsbgrdanyy:
            sbrt['bgrd'] = []
        if boolelemsbrtextsbgrdanyy:
            sbrt['bgrdgalx'] = []
        
        if dictchalinpt['numbener'] > 1:
            specsour = retr_spec(gdat, np.array([fluxsour]), sind=np.array([sindsour]))
        else:
            specsour = np.array([fluxsour])
        
        if boolelemsbrtextsbgrdanyy:
        
            if typeverb > 1:
                print('Interpolating the background emission...')

            sbrt['bgrdgalx'] = retr_sbrtsers(gdat, gdat.xposgrid[indxpixlelem[0]], gdat.yposgrid[indxpixlelem[0]], \
                                                                            xpossour, ypossour, specsour, sizesour, ellpsour, anglsour)
            
            sbrt['bgrd'] = sbrt['bgrdgalx'] + sbrtextsbgrd
        
            sbrt['lens'] = np.empty_like(gdat.cntpdata)
            for ii, i in enumerate(indxener):
                for mm, m in enumerate(indxdqlt):
                    sbrtbgrdobjt = scipy.interpolate.RectBivariateSpline(gdat.bctrpara.yposcart, gdat.bctrpara.xposcart, \
                                                            sbrt['bgrd'][ii, :, mm].reshape((gdat.numbsidecart, gdat.numbsidecart)).T)
                    
                    yposprim = gdat.yposgrid[indxpixlelem[0]] - defl[indxpixlelem[0], 1]
                    xposprim = gdat.xposgrid[indxpixlelem[0]] - defl[indxpixlelem[0], 0]
                    # temp -- T?
                    sbrt['lens'][ii, :, m] = sbrtbgrdobjt(yposprim, xposprim, grid=False).flatten()
        else:
            if typeverb > 1:
                print('Not interpolating the background emission...')
            
            sbrt['lens'] = retr_sbrtsers(gdat, gdat.xposgrid - defl[indxpixl, 0], \
                                                   gdat.yposgrid - defl[indxpixl, 1], \
                                                   xpossour, ypossour, specsour, sizesour, ellpsour, anglsour)
            
            sbrt['bgrd'] = retr_sbrtsers(gdat, gdat.xposgrid, \
                                                   gdat.yposgrid, \
                                                   xpossour, ypossour, specsour, sizesour, ellpsour, anglsour)
            
        if gdat.booldiag:
            if not np.isfinite(sbrt['lens']).all():
                raise Exception('Lensed emission is not finite.')
            if (sbrt['lens'] == 0).all():
                raise Exception('Lensed emission is zero everywhere.')

        #objttimeprof.stopchro(gdat, gdatmodi, 'sbrtlens')
        
    ## host galaxy
    if typeemishost != 'none':
        #objttimeprof.initchro(gdat, gdatmodi, 'sbrthost')

        for e in indxsersfgrd:
            if typeverb > 1:
                print('Evaluating the host galaxy surface brightness...')
            
            if dictchalinpt['numbener'] > 1:
                spechost = retr_spec(gdat, np.array([fluxhost[e]]), sind=np.array([sindhost[e]]))
            else:
                spechost = np.array([fluxhost[e]])
            
            sbrt['hostisf%d' % e] = retr_sbrtsers(gdat, gdat.xposgrid, gdat.yposgrid, xposhost[e], \
                                                         yposhost[e], spechost, sizehost[e], ellphost[e], anglhost[e], serihost[e])
            
        #objttimeprof.stopchro(gdat, gdatmodi, 'sbrthost')
    
    ## total model
    #objttimeprof.initchro(gdat, gdatmodi, 'sbrtmodl')
    if typeverb > 1:
        print('Summing up the model emission...')
    
    sbrt['modlraww'] = np.zeros((dictchalinpt['numbener'], numbpixl, gdat.numbdqlt))
    for name in listnamediff:
        if name.startswith('back'):
            indxbacktemp = int(name[4:8])
            
            if typepixl == 'heal' and (typeevalpsfn == 'full' or typeevalpsfn == 'conv') and not boolunifback[indxbacktemp]:
                sbrttemp = getattr(gmod, 'sbrtbackhealfull')[indxbacktemp]
            else:
                sbrttemp = sbrtbacknorm[indxbacktemp]
           
            if boolspecback[indxbacktemp]:
                sbrt[name] = sbrttemp * bacp[indxbacpback[indxbacktemp]]
            else:
                sbrt[name] = sbrttemp * bacp[indxbacpback[indxbacktemp][indxener]][:, None, None]
        
        sbrt['modlraww'] += sbrt[name]
        
        if gdat.booldiag:
            if np.amax(sbrttemp) == 0.:
                raise Exception('')

    # convolve the model with the PSF
    if convdiffanyy and (typeevalpsfn == 'full' or typeevalpsfn == 'conv'):
        sbrt['modlconv'] = []
        # temp -- isotropic background proposals are unnecessarily entering this clause
        if typeverb > 1:
            print('Convolving the model image with the PSF...') 
        sbrt['modlconv'] = np.zeros((dictchalinpt['numbener'], numbpixl, gdat.numbdqlt))
        for ii, i in enumerate(indxener):
            for mm, m in enumerate(indxdqlt):
                if gdat.strgcnfg == 'pcat_ferm_igal_simu_test':
                    print('Convolving ii, i, mm, m')
                    print(ii, i, mm, m)
                if typepixl == 'cart':
                    if numbpixl == numbpixl:
                        sbrt['modlconv'][ii, :, mm] = convolve_fft(sbrt['modlraww'][ii, :, mm].reshape((gdat.numbsidecart, gdat.numbsidecart)), \
                                                                                                                             objtpsfnconv[mm][ii]).flatten()
                    else:
                        sbrtfull = np.zeros(numbpixl)
                        sbrtfull[indxpixlrofi] = sbrt['modlraww'][ii, :, mm]
                        sbrtfull = sbrtfull.reshape((gdat.numbsidecart, gdat.numbsidecart))
                        sbrt['modlconv'][ii, :, mm] = convolve_fft(sbrtfull, objtpsfnconv[mm][ii]).flatten()[indxpixlrofi]
                    indx = np.where(sbrt['modlconv'][ii, :, mm] < 1e-50)
                    sbrt['modlconv'][ii, indx, mm] = 1e-50
                if typepixl == 'heal':
                    sbrt['modlconv'][ii, :, mm] = hp.smoothing(sbrt['modlraww'][ii, :, mm], fwhm=fwhm[i, m])[indxpixlrofi]
                    sbrt['modlconv'][ii, :, mm][np.where(sbrt['modlraww'][ii, :, mm] <= 1e-50)] = 1e-50
        
        # temp -- this could be made faster -- need the copy() statement because sbrtdfnc gets added to sbrtmodl afterwards
        sbrt['modl'] = np.copy(sbrt['modlconv'])
    else:
        if typeverb > 1:
            print('Skipping PSF convolution of the model...')
        sbrt['modl'] = np.copy(sbrt['modlraww'])
    
    if typeverb > 1:
        print('sbrt[modl]')
        summgene(sbrt['modl'])

    ## add PSF-convolved delta functions to the model
    if boolelemsbrtdfncanyy:
        sbrt['modl'] += sbrt['dfnc']
    #bjttimeprof.stopchro(gdat, gdatmodi, 'sbrtmodl')
    
    if typeverb > 1:
        print('sbrt[modl]')
        summgene(sbrt['modl'])
    
    #dictchaloutp['objttimeprof'] = objttimeprof

    return dictchaloutp



