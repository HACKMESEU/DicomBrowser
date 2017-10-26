# DicomBrowser
# Copyright (C) 2016-7 Eric Kerfoot, King's College London, all rights reserved
# 
# This file is part of DicomBrowser.
#
# DicomBrowser is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# DicomBrowser is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License along
# with this program (LICENSE.txt).  If not, see <http://www.gnu.org/licenses/>
'''
DicomBrowser - simple lightweight Dicom browsing application. 
'''
import dwilib.dwi.models

import sys, os, threading, math, re
from operator import itemgetter
from multiprocessing import Pool, Manager, cpu_count, freeze_support
from contextlib import closing
from collections import OrderedDict
import time
try: # Python 2 and 3 support
    from Queue import Queue, Empty
    from StringIO import StringIO
except ImportError:
    from queue import Queue, Empty
    from io import StringIO

from PyQt5 import QtGui, QtCore, uic
import PyQt5
dirname = os.path.dirname(PyQt5.__file__)
plugin_path = os.path.join(dirname,'Qt', 'plugins')

os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = plugin_path

from PyQt5.QtCore import Qt, QStringListModel
import Resources_rc5 # import resources manually since we have to do this to get the ui file
    
# add test
scriptdir=os.path.dirname(os.path.abspath(__file__)) # path of the current file

 #this allows the script to be run directly from the repository without having to install pydicom or pyqtgraph
#if os.path.isdir(scriptdir+'/../pydicom'):
#    sys.path.append(scriptdir+'/../pydicom')
#    sys.path.append(scriptdir+'/../pyqtgraph')

import numpy as np
import pyqtgraph as pg
#from pyqtgraph.widgets import PlotWidget
from pydicom import dicomio, datadict, errors
#import pydicom
#from __init__ import __version__
__version_info__=(1,0,0) # global application version, major/minor/patch
__version__='%i.%i.%i'%__version_info__

# load the ui file from the resource, removing the "resources" tag so that uic doesn't try (and fail) to load the resources
# this allows a loading the UI at runtime rather than generating a .py file with pyuic which isn't cross-compatible with PyQt4/5
with closing(QtCore.QFile(':/layout/DicomBrowserWin.ui')) as layout:
    if layout.open(QtCore.QFile.ReadOnly):
        s=bytes(layout.readAll()).decode('utf-8')
        s=re.sub('<resources>.*</resources>','',s,flags=re.DOTALL) # get rid of the resources section in the XML
        Ui_DicomBrowserWin,_=uic.loadUiType(StringIO(s)) # create a local type definition


# tag names of default columns in the series list, this can be changed to pull out different tag names for columns
seriesListColumns=('PatientName','SeriesDescription','SeriesNumber','NumImages')
# names of columns in tag tree, this shouldn't ever change
tagTreeColumns=('Name','Tag','Value')
# list of tags to initially load when a directory is scanned, loading only these speeds up scanning immensely
loadTags=('SeriesInstanceUID','TriggerTime','PatientName','SeriesDescription','SeriesNumber')

# keyword/full name pairs for extra properties not represented as Dicom tags
extraKeywords={
    'NumImages':'# Images',
    'TimestepSpec':'Timestep Info',
    'StartTime':'Start Time',
    'NumTimesteps':'# Timesteps',
    'TimeInterval':'Time Interval'
}

# maps keywords to their full names
keywordNameMap={v[4]:v[2] for v in datadict.DicomDictionary.values()}
keywordNameMap.update(extraKeywords)

fullNameMap={v:k for k,v in keywordNameMap.items()} # maps full names to keywords

        
def fillTagModel(model,dcm,regex=None):
    '''Fill a QStandardItemModel object `model' with a tree derived from tags in `dcm', filtering by pattern `regex'.'''
    try:
        regex=re.compile(str(regex),re.DOTALL)
    except:
        regex='' # no regex or bad pattern
            
    def _datasetToItem(parent,d):
        '''Add every element in `d' to the QStandardItem object `parent', this will be recursive for list elements.'''
        for elem in d:
            value=_elemToValue(elem)
            tag='(%04x, %04x)'%(elem.tag.group,elem.tag.elem)
            parent1 = QtGui.QStandardItem(str(elem.name))
            tagitem = QtGui.QStandardItem(tag)
            
            if isinstance(value,str):
                try:
                    value=value.decode('ascii')
                    if '\n' in value or '\r' in value: # multiline text data should be shown as repr
                        value=repr(value)
                except:
                    value=repr(value)
                    
                if not regex or re.search(regex,str(elem.name)+tag+value) is not None:
                    parent.appendRow([parent1,tagitem,QtGui.QStandardItem(value)])
                    
            elif value is not None and len(value)>0:
                parent.appendRow([parent1,tagitem])
                for v in value:
                    parent1.appendRow(v)
        
    def _elemToValue(elem):
        '''Return the value in `elem', which will be a string or a list of QStandardItem objects if elem.VR=='SQ'.'''
        value=None
        if elem.VR=='SQ':
            value=[]
            for i,item in enumerate(elem):
                parent1 = QtGui.QStandardItem('%s %i'%(elem.name,i))
                _datasetToItem(parent1,item)
                if not regex or parent1.hasChildren(): # discard sequences whose children have been filtered out
                    value.append(parent1)
        elif elem.name!='Pixel Data':
            value=str(elem.value)

        return value        
                
    _datasetToItem(model,dcm)
    

def loadDicomFiles(filenames,queue):
    '''Load the Dicom files `filenames' and put an abbreviated tag->value map for each onto `queue'.'''
    for filename in filenames:
        try:
            dcm=dicomio.read_file(filename,stop_before_pixels=True)
            tags={t:dcm.get(t) for t in loadTags if t in dcm}
            queue.put((filename,tags))
        except errors.InvalidDicomError:
            pass


def loadDicomDir(rootdir,statusfunc=lambda s,c,n:None,numprocs=None):
    '''
    Load all the Dicom files from `rootdir' using `numprocs' number of processes. This will attempt to load each file
    found in `rootdir' and store from each file the tags defined in loadTags. The filenames and the loaded tags for
    Dicom files are stored in a DicomSeries object representing the acquisition series each file belongs to. The 
    `statusfunc' callback is used to indicate loading status, taking as arguments a status string, count of loaded 
    objects, and the total number to load. A status string of '' indicates loading is done. The default value causes 
    no status indication to be made. Return value is a sequence of DicomSeries objects in no particular order.
    '''
    allfiles=[]
    for root,_,files in os.walk(rootdir):
        allfiles+=[os.path.join(root,f) for f in files]
        
    numprocs=numprocs or cpu_count()
    m = Manager()
    queue=m.Queue()
    numfiles=len(allfiles)
    res=[]
    series={}
    count=0
    
    if not numfiles:
        return []

    with closing(Pool(processes=numprocs)) as pool:
        for i in range(numprocs):
            # partition the list of files amongst each processor
            partsize=numfiles/float(numprocs)
            start=int(math.floor(i*partsize))
            end=int(math.floor((i+1)*partsize))
            if (numfiles-end)<partsize:
                end=numfiles
                
            r=pool.apply_async(loadDicomFiles,(allfiles[start:end],queue))
            res.append(r)
    
        # loop so long as any process is busy or there are files on the queue to process
        while any(not r.ready() for r in res) or not queue.empty():
            try:
                filename,dcm=queue.get(False)
                seriesid=dcm.get('SeriesInstanceUID','???')
                if seriesid not in series:
                    series[seriesid]=DicomSeries(seriesid,rootdir)
    
                series[seriesid].addFile(filename,dcm)
                count+=1
                
                # update status only 100 times, doing it too frequently really slows things down
                if numfiles<100 or count%(numfiles//100)==0: 
                    statusfunc('Loading DICOM files',count,numfiles)
            except Empty: # from queue.get(), keep trying so long as the loop condition is true
                pass


    statusfunc('',0,0)



    return list(series.values())
def  GFRmodelCalSeries  (series,phasetimegap,AifData):
    TotalNum = len(series.dcms)
    # timelist = series.acquiretimelist
    sliceNum = series.sliceNum
    phaseNum = int(TotalNum / sliceNum)

    rows = series.getTagObject(0).Rows
    columns = series.getTagObject(0).Columns

    # data = np.zeros((rows,columns,sliceNum,phaseNum))

    # for i in range(TotalNum):data[:,:,int(i%sliceNum),int(i/sliceNum)]=series.getPixelData(i)

    # phasetimegap = self.PhaseTimeGap.value()

    #m_GRF = np.zeros((rows, columns, sliceNum))
    deltaT = phasetimegap / 60.

    m_GRF = GFRmodelCalSlices(series,phaseNum,sliceNum,rows,columns,AifData,deltaT,range(sliceNum))
    return m_GRF


def GFRmodelCalSeriesMP(series,phasetimegap,AifData,statusfunc=lambda s,c,n:None):
    TotalNum = len(series.dcms)
    # timelist = series.acquiretimelist
    sliceNum = series.sliceNum
    phaseNum = int(TotalNum / sliceNum)

    rows = series.getTagObject(0).Rows
    columns = series.getTagObject(0).Columns

    # data = np.zeros((rows,columns,sliceNum,phaseNum))

    # for i in range(TotalNum):data[:,:,int(i%sliceNum),int(i/sliceNum)]=series.getPixelData(i)

    #phasetimegap = self.PhaseTimeGap.value()

    m_GRF = np.zeros((rows, columns,sliceNum))
    deltaT = phasetimegap / 60.

    #numprocs = cpu_count()
    numprocs = 1
    m = Manager()
    queue = m.Queue()
    numfiles = sliceNum
    res = []
    #series = {}
    count = 0

    if not numfiles:
        return []

    with closing(Pool(processes=numprocs)) as pool:
        for i in range(numprocs):
            # partition the list of files amongst each processor
            partsize = numfiles / float(numprocs)
            start = int(math.floor(i * partsize))
            end = int(math.floor((i + 1) * partsize))
            if (numfiles - end) < partsize:
                end = numfiles

            r = pool.apply_async(GFRmodelCalSlicesMP, (series,phaseNum,sliceNum,rows,columns,AifData,deltaT,range(sliceNum), queue))
            res.append(r)

        # loop so long as any process is busy or there are files on the queue to process
        while any(not r.ready() for r in res) or not queue.empty():
            try:
                GRF, s = queue.get(True,0.5)
                m_GRF[:,:,s] = GRF
                count += 1
                # update status only 100 times, doing it too frequently really slows things down
                if numfiles < 100 or count % (numfiles // 100) == 0:
                    statusfunc('Calculate slices', count, numfiles)
            except Empty:  # from queue.get(), keep trying so long as the loop condition is true
                pass

    statusfunc('', 0, 0)

    #for s in range(sliceNum):
        #m_GRF[:,:,s] = GFRmodelCalSlice(series,phaseNum,sliceNum,rows,columns,AifData,deltaT,s)
    return  m_GRF




def GFRmodelCalSlicesMP(series,phaseNum,sliceNum,rows,columns,AifData,deltaT,slices,queue):
    for s in slices:
        GFRmodelCalMP(series, phaseNum, sliceNum, rows, columns, AifData, deltaT, s, queue)



def GFRmodelCalSlices(series,phaseNum,sliceNum,rows,columns,AifData,deltaT,slices):
    dcms = np.zeros((rows, columns, phaseNum))
    Croi = np.zeros((phaseNum, 1))
    A = np.zeros((phaseNum, 2))
    C = np.zeros((phaseNum, 1))
    m_GRF = np.zeros((rows, columns,len(slices)))

    for s in slices:
        for p in range(phaseNum): dcms[:, :, p] = series.getPixelData(p * sliceNum + s)
        for i in range(rows):
            for j in range(columns):
                for p in range(phaseNum):
                    if (p == 0):
                        Croi[p] = dcms[i, j, p]
                    else:
                        Croi[p] = dcms[i, j, p] - Croi[0]
                cumAif = 0
                for p in range(phaseNum):
                    cumAif += AifData[p] * deltaT
                    A[p, 0] = cumAif
                    A[p, 1] = AifData[p]
                    C[p] = Croi[p]
                B = np.dot(np.linalg.pinv(A), C)
                m_GRF[i, j, s] = B[0]
    return m_GRF

def GFRmodelCalMP(series,phaseNum,sliceNum,rows,columns,AifData,deltaT,s,queue):
    dcms = np.zeros((rows, columns, phaseNum))
    Croi = np.zeros((phaseNum, 1))
    A = np.zeros((phaseNum, 2))
    C = np.zeros((phaseNum, 1))
    m_GRF = np.zeros((rows, columns))
    for p in range(phaseNum): dcms[:, :, p] = series.getPixelData(p * sliceNum + s)
    for i in range(rows):
        for j in range(columns):
            for p in range(phaseNum):
                if (p == 0):
                    Croi[p] = dcms[i, j, p]
                else:
                    Croi[p] = dcms[i, j, p] - Croi[0]
            cumAif = 0
            for p in range(phaseNum):
                cumAif += AifData[p] * deltaT
                A[p, 0] = cumAif
                A[p, 1] = AifData[p]
                C[p] = Croi[p]
            B = np.dot(np.linalg.pinv(A), C)
            m_GRF[i, j] = abs(B[0])
    queue.put((m_GRF,s))


import numpy as np

import dipy.reconst.dki as dki
import dipy.reconst.dti as dti
import dipy.reconst.dki_micro as dki_micro

from dipy.segment.mask import median_otsu
from scipy.ndimage.filters import gaussian_filter
from dipy.core.gradients import gradient_table

def DKImodelCal(bvals,bvecs,data,mask1):

    gtab = gradient_table(np.array(bvals),np.array(bvecs))
    #maskdata, mask = median_otsu(data, 4, 2, False, vol_idx=[0, 1], dilate=1)
    mask = np.ones((data.shape[0],data.shape[1],1))
    mask[:, :, 0] = mask1


    fwhm = 1.25
    gauss_std = fwhm / np.sqrt(8 * np.log(2))  # converting fwhm to Gaussian std
    data_smooth = np.zeros(data.shape)
    for v in range(data.shape[-1]):
        data_smooth[..., v] = gaussian_filter(data[..., v], sigma=gauss_std)
    dkimodel = dki.DiffusionKurtosisModel(gtab)
    dkifit = dkimodel.fit(data_smooth,mask=mask)
    FA = dkifit.fa
    MD = dkifit.md
    AD = dkifit.ad
    RD = dkifit.rd
    MK = dkifit.mk(0, 3)
    AK = dkifit.ak(0, 3)
    RK = dkifit.rk(0, 3)
    #AKC,ADC = dkifit.akc(0,3)
    return FA,MD,AD,RD,MK,AK,RK


def DTImodelCal(bvals,bvecs,data,mask1):

    gtab = gradient_table(np.array(bvals),np.array(bvecs))
    #maskdata, mask = median_otsu(data, 4, 2, False, vol_idx=[0, 1], dilate=1)
    mask = np.ones((data.shape[0],data.shape[1],1))
    mask[:,:,0] = mask1
    #mask[50:60,50:60,0] = 1


    fwhm = 1.25
    gauss_std = fwhm / np.sqrt(8 * np.log(2))  # converting fwhm to Gaussian std
    data_smooth = np.zeros(data.shape)
    for v in range(data.shape[-1]):
        data_smooth[..., v] = gaussian_filter(data[..., v], sigma=gauss_std)

    tenmodel = dti.TensorModel(gtab)
    tenfit = tenmodel.fit(data_smooth, mask=mask)
    FA = tenfit.fa
    MD = tenfit.md
    AD = tenfit.ad
    RD = tenfit.rd

    return FA,MD,AD,RD


def ADCMonomodelCal(bvalslist,data,mask):

    ADCm = np.zeros((data.shape[0], data.shape[1],data.shape[2]))
    for p in range(data.shape[2]):
        datap = data[:,:,p,:]
        for i in range(datap.shape[0]):
            for j in range(datap.shape[1]):
                if not mask[i,j]:
                    datap[i,j,0] = np.nan


        model = [x for x in dwilib.dwi.models.Models if x.name == 'MonoN'][0]
        bvalslist = np.asanyarray(bvalslist)
        images = datap.reshape(-1, len(bvalslist))
        pmap = model.fit(bvalslist, images)
        pmap = pmap.reshape((datap.shape[0], datap.shape[1], pmap.shape[1]))
        ADCm[:,:,p] = pmap[:,:,0]
        for i in range(ADCm.shape[0]):
            for j in range(ADCm.shape[1]):
                if math.isnan(ADCm[i,j,p]):
                    ADCm[i,j,p] = 0

    return ADCm

def SEmodelCal(bvalslist,data,mask):
    ADCs = np.zeros((data.shape[0], data.shape[1],data.shape[2]))
    Alpha = np.zeros((data.shape[0], data.shape[1],data.shape[2]))
    for p in range(data.shape[2]):
        datap = data[:,:,p,:]
        for i in range(datap.shape[0]):
            for j in range(datap.shape[1]):
                if not mask[i,j]:
                    datap[i,j,0] = np.nan

        model = [x for x in dwilib.dwi.models.Models if x.name == 'StretchedN'][0]
        bvalslist = np.asanyarray(bvalslist)
        images = datap.reshape(-1, len(bvalslist))
        pmap = model.fit(bvalslist, images)
        pmap = pmap.reshape((datap.shape[0], datap.shape[1], pmap.shape[1]))
        ADCs[:,:,p] = pmap[:,:,0]
        Alpha[:,:,p] = pmap[:,:,1]
        ADCbound = 0.01
        for i in range(ADCs.shape[0]):
            for j in range(Alpha.shape[1]):
                if ADCs[i,j,p]>ADCbound or math.isnan(ADCs[i,j,p]):
                    ADCs[i, j, p] = 0
                if math.isnan(Alpha[i,j,p]):
                    Alpha[i,j,p] = 0


    return ADCs,Alpha


from dipy.reconst.ivim import IvimModel
def IVIMmodelCal(bvals,bvecs,data,mask1):
    gtab = gradient_table(np.array(bvals), np.array(bvecs))
    mask = np.ones((data.shape[0],data.shape[1],1))
    mask[:, :, 0] = mask1
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if data[i,j,0,0] == 0:
                mask[i,j,0] = 0


    ivimmodel = IvimModel(gtab)
    ivimfit = ivimmodel.fit(data,mask=mask)
    return ivimfit.S0_predicted,ivimfit.perfusion_fraction,ivimfit.D_star,ivimfit.D


def GFRmodelCal(series,phaseNum,sliceNum,rows,columns,AifData,deltaT,s,statusfunc=lambda s,c,n:None):
    dcms = np.zeros((rows, columns, phaseNum))
    Croi = np.zeros((phaseNum, 1))
    A = np.zeros((phaseNum, 2))
    C = np.zeros((phaseNum, 1))
    m_GRF = np.zeros((rows, columns,1))



    for p in range(phaseNum): dcms[:, :, p] = series.getPixelData(p * sliceNum + s)
    for i in range(rows):
        for j in range(columns):
            for p in range(phaseNum):
                if (p == 0):
                    Croi[p] = dcms[i, j, p]
                else:
                    Croi[p] = dcms[i, j, p] - Croi[0]
            cumAif = 0
            for p in range(phaseNum):
                cumAif += AifData[p] * deltaT
                A[p, 0] = cumAif
                A[p, 1] = AifData[p]
                C[p] = Croi[p]
            B = np.dot(np.linalg.pinv(A), C)
            m_GRF[i, j,0] = abs(B[0])
        if(i%10==0):
            statusfunc('Calculate DICOM rows', i, rows)

    statusfunc('', 0, 0)
    return m_GRF


class DicomSeries(object):
    '''
    This type represents a Dicom series as a list of Dicom files sharing a series UID. The assumption is that the images
    of a series were captured together and so will always have a number of fields in common, such as patient name, so
    Dicoms should be organized by series. This type will also cache loaded Dicom tags and images
    '''
    def __init__(self,seriesID,rootdir):
        self.seriesID=seriesID # ID of the series or ???
        self.rootdir=rootdir # the directory where Dicoms were loaded from, files for this series may be in subdirectories
        self.filenames=[] # list of filenames for the Dicom associated with this series
        self.dcms=[] # loaded abbreviated tag->(name,value) maps, 1 for each of self.filenames
        self.imgcache={} # cache of loaded image data, mapping index in self.filenames to ndarray objects or None for non-images files
        self.tagcache={} # cache of all loaded tag values, mapping index in self.filenames to OrderedDict of tag->(name,value) maps
        self.sliceNum = 1
        self.acquiretimelist = []
        self.bvalslist = []
        self.bvecslist = []
        self.poslist = []
        self.pos = []

    def addFile(self,filename,dcm):
        '''Add a filename and abbreviated tag map to the series.'''
        self.filenames.append(filename)
        self.dcms.append(dcm)
        
    def getTagObject(self,index):
        '''Get the object storing tag information from Dicom file at the given index.'''
        if index not in self.tagcache:
            dcm=dicomio.read_file(self.filenames[index],stop_before_pixels=True)
            self.tagcache[index]=dcm
            
        return self.tagcache[index]

    def getExtraTagValues(self):
        '''Return the extra tag values calculated from the series tag info stored in self.dcms.'''
        start,interval,numtimes=self.getTimestepSpec()
        extravals={
            'NumImages':len(self.dcms),
            'TimestepSpec':'start: %i, interval: %i, # Steps: %i'%(start,interval,numtimes),
            'StartTime':start,
            'NumTimesteps':numtimes,
            'TimeInterval':interval
        }

        return extravals
        
    def getTagValues(self,names,index=0):
        '''Get the tag values for tag names listed in `names' for image at the given index.'''
        if not self.filenames:
            return ()

        dcm=self.getTagObject(index)
        extravals=self.getExtraTagValues()
        
        return tuple(str(dcm.get(n,extravals.get(n,''))) for n in names)

    def getPixelData(self,index):
        '''Get the pixel data Numpy array for file at position `index` in self.filenames, or None if there is no pixel data.'''
        if index not in self.imgcache:
            img=None
            try:
                dcm=dicomio.read_file(self.filenames[index])
                if dcm.pixel_array is not None:
                    rslope=float(dcm.get('RescaleSlope',1))
                    rinter=float(dcm.get('RescaleIntercept',0))
                    img= dcm.pixel_array*rslope+rinter
            except Exception:
                pass
                
            self.imgcache[index]=img
            
        return self.imgcache[index]

    def addSeries(self,series):
        '''Add every loaded dcm file from DicomSeries object `series` into this series.'''
        for f,dcm in zip(series.filenames,series.dcms):
            self.addFile(f,dcm)

    def getTimestepSpec(self,tag='TriggerTime'):
        '''Returns (start time, interval, num timesteps) triple.'''
        times=sorted(set(int(dcm.get(tag,0)) for dcm in self.dcms))

        if not times or times==[0]:
            return 0.0,0.0,0.0
        else:
            if len(times)==1:
                times=times*2
            
            avgspan=np.average([b-a for a,b in zip(times,times[1:])])
            return times[0],avgspan,len(times)

    def getSliceNum(self):
        TotalNum = len(self.dcms)
        pos = []
        for i in range(TotalNum):
            sliceloc = self.getTagObject(i).SliceLocation
            if sliceloc not in pos:
                pos.append(sliceloc)
        self.sliceNum = len(pos)

    def getAcquireTimeList(self):
        acquiretimelist = []
        TotalNum = len(self.dcms)
        for i in range(TotalNum):
            sliceloc = self.getTagObject(i).AcquisitionTime
            acquiretimelist.append(sliceloc)
        self.acquiretimelist = acquiretimelist
    def getDiffusionVals(self):
        bvalslist = []
        bvecslist = []
        TotalNum = len(self.dcms)
        for i in range(TotalNum):
            bval = self.getTagObject(i).DiffusionBValue
            bvec = self.getTagObject(i).DiffusionGradientOrientation
            bvalslist.append(bval)
            bvecslist.append(bvec)
        self.bvalslist = bvalslist
        self.bvecslist = bvecslist

    def getPos(self):
        poslist = []
        postemp = []
        TotalNum = len(self.dcms)
        for i in range(TotalNum):
            pos = self.getTagObject(i).SliceLocation
            poslist.append(pos)
            if pos not in postemp:
                postemp.append(pos)
        self.poslist = poslist
        self.pos = postemp


class SeriesTableModel(QtCore.QAbstractTableModel):
    '''This manages the list of series with a sorting feature.'''
    def __init__(self, seriesColumns,parent=None):
        QtCore.QAbstractTableModel.__init__(self, parent)
        self.seriesTable=[]
        self.seriesColumns=seriesColumns
        self.sortCol=0
        self.sortOrder=Qt.AscendingOrder

    def rowCount(self, parent):
        return len(self.seriesTable)

    def columnCount(self,parent):
        return len(self.seriesTable[0]) if self.seriesTable else 0

    def sort(self,column,order):
        self.layoutAboutToBeChanged.emit()
        self.sortCol=column
        self.sortOrder=order

        self.seriesTable.sort(key=itemgetter(column),reverse=order==Qt.DescendingOrder)
        self.layoutChanged.emit()
        
    def updateSeriesTable(self,seriesTable):
        self.seriesTable=list(seriesTable)
        self.sort(self.sortCol,self.sortOrder) # sort using existing parameters
        
    def getRow(self,i):
        return self.seriesTable[i]

    def headerData(self, section, orientation, role):
        if role == Qt.DisplayRole and orientation==Qt.Horizontal:
            return keywordNameMap[self.seriesColumns[section]]

    def data(self, index, role):
        if index.isValid() and role == Qt.DisplayRole:
            return str(self.seriesTable[index.row()][index.column()])

from threading import Thread
class ThreadWithReturnValue(Thread):
    def __init__(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        Thread.__init__(self, group, target, name, args, kwargs, daemon=daemon)

        self._return = None

    def run(self):
        if self._target is not None:
            self._return = self._target(*self._args, **self._kwargs)

    def join(self):
        Thread.join(self)
        return self._return

from PyQt5.QtCore import *
from PyQt5.QtGui import *
class DicomBrowser(QtGui.QMainWindow,Ui_DicomBrowserWin):
    '''
    This is the window class for the app which implements the UI functionality and the directory loading thread. It 
    inherits from the type loaded from the .ui file in the resources. 
    '''
    statusSignal=QtCore.pyqtSignal(str,int,int) # signal for updating the status bar asynchronously
    updateSignal=QtCore.pyqtSignal() # signal for updating the source list and series table

    statusSignalCal = QtCore.pyqtSignal(str,int,int)
    statusSignalDTICal = QtCore.pyqtSignal(str, int, int)
    statusSignalDTISave = QtCore.pyqtSignal(str, int, int)
    statusSignalDKICal = QtCore.pyqtSignal(str, int, int)
    statusSignalDKISave = QtCore.pyqtSignal(str, int, int)
    statusSignalIVIMCal = QtCore.pyqtSignal(str, int, int)
    statusSignalIVIMSave = QtCore.pyqtSignal(str, int, int)

    statusSignalMonoECal = QtCore.pyqtSignal(str, int, int)
    statusSignalMonoESave = QtCore.pyqtSignal(str, int, int)
    statusSignalSECal = QtCore.pyqtSignal(str, int, int)
    statusSignalSESave = QtCore.pyqtSignal(str, int, int)

    def __init__(self,args,parent=None):
        QtGui.QMainWindow.__init__(self,parent)

        self.srclist=[] # list of source directories
        self.imageIndex=0 # index of selected image
        self.seriesMap=OrderedDict() # maps series table row tuples to DicomSeries object it was generated from
        self.seriesColumns=list(seriesListColumns) # keywords for columns
        self.selectedRow=-1 # selected series row
        self.lastDir='.' # last loaded directory root
        self.filterRegex='' # regular expression to filter tags by

        # create the directory queue and loading thread objects
        self.dirQueue=Queue() # queue of directories to load
        self.loadDirThread=threading.Thread(target=self._loadDirsThread)
        self.loadDirThread.daemon=True # clean shutdown possible with daemon threads
        self.loadDirThread.start() # start the thread now, it will wait until something is put on self.dirQueue
#
        self.GFRQueue = Queue()
        self.GFRmodelCalThread = threading.Thread(target=self._GFRmodelCalThread)
        self.GFRmodelCalThread.daemon = True
        self.GFRmodelCalThread.start()

        self.DKIQueue = Queue()
        self.DKImodelCalThreading = threading.Thread(target=self._DKImodelCalThread)
        self.DKImodelCalThreading.daemon = True
        self.DKImodelCalThreading.start()

        self.DKISave = Queue()
        self.DKImodelSaveThreading = threading.Thread(target=self._DKImodelSaveThread)
        self.DKImodelSaveThreading.daemon = True
        self.DKImodelSaveThreading.start()

        self.DTIQueue = Queue()
        self.DTImodelCalThreading = threading.Thread(target=self._DTImodelCalThread)
        self.DTImodelCalThreading.daemon = True
        self.DTImodelCalThreading.start()

        self.DTISave = Queue()
        self.DTImodelSaveThreading = threading.Thread(target=self._DTImodelSaveThread)
        self.DTImodelSaveThreading.daemon = True
        self.DTImodelSaveThreading.start()

        self.IVIMQueue = Queue()
        self.IVIMmodelCalThreading = threading.Thread(target=self._IVIMmodelCalThread)
        self.IVIMmodelCalThreading.daemon = True
        self.IVIMmodelCalThreading.start()

        self.IVIMSave = Queue()
        self.IVIMmodelSaveThreading = threading.Thread(target=self._IVIMmodelSaveThread)
        self.IVIMmodelSaveThreading.daemon = True
        self.IVIMmodelSaveThreading.start()

        self.ADCMonoQueue = Queue()
        self.ADCMonomodelCalThreading = threading.Thread(target=self._ADCMonomodelCalThread)
        self.ADCMonomodelCalThreading.daemon = True
        self.ADCMonomodelCalThreading.start()

        self.ADCMonoSave = Queue()
        self.ADCMonomodelSaveThreading = threading.Thread(target=self._ADCMonomodelSaveThread)
        self.ADCMonomodelSaveThreading.daemon = True
        self.ADCMonomodelSaveThreading.start()

        self.SEQueue = Queue()
        self.SEmodelCalThreading = threading.Thread(target=self._SEmodelCalThread)
        self.SEmodelCalThreading.daemon = True
        self.SEmodelCalThreading.start()

        self.SESave = Queue()
        self.SEmodelSaveThreading = threading.Thread(target=self._SEmodelSaveThread)
        self.SEmodelSaveThreading.daemon = True
        self.SEmodelSaveThreading.start()

        # setup ui
        #self.roiPlot = pg.widgets.PlotWidget(self.verticalLayout_3)
        self.setupUi(self) # create UI elements based on the loaded .ui file
        self.setWindowTitle('DICOM Toolkit v%s '%(__version__))
        self.setStatus('')
        self.setStatusCal('')
        self.statusProgressBarCal.setRange(0,100)
        self.statusProgressBarCal.setValue(0)

        self.statusProgressBarDTICal.setRange(0, 100)
        self.statusProgressBarDTICal.setValue(0)
        self.statusProgressBarDTICal.setVisible(0)

        self.statusProgressBarDTISave.setRange(0, 100)
        self.statusProgressBarDTISave.setValue(0)
        self.statusProgressBarDTISave.setVisible(0)

        self.statusProgressBarDKICal.setRange(0, 100)
        self.statusProgressBarDKICal.setValue(0)
        self.statusProgressBarDKICal.setVisible(0)

        self.statusProgressBarDKISave.setRange(0, 100)
        self.statusProgressBarDKISave.setValue(0)
        self.statusProgressBarDKISave.setVisible(0)

        self.statusProgressBarIVIMCal.setRange(0, 100)
        self.statusProgressBarIVIMCal.setValue(0)
        self.statusProgressBarIVIMCal.setVisible(0)

        self.statusProgressBarIVIMSave.setRange(0, 100)
        self.statusProgressBarIVIMSave.setValue(0)
        self.statusProgressBarIVIMSave.setVisible(0)

        self.statusProgressBarMonoECal.setRange(0, 100)
        self.statusProgressBarMonoECal.setValue(0)
        self.statusProgressBarMonoECal.setVisible(0)

        self.statusProgressBarMonoESave.setRange(0, 100)
        self.statusProgressBarMonoESave.setValue(0)
        self.statusProgressBarMonoESave.setVisible(0)

        self.statusProgressBarSECal.setRange(0, 100)
        self.statusProgressBarSECal.setValue(0)
        self.statusProgressBarSECal.setVisible(0)

        self.statusProgressBarSESave.setRange(0, 100)
        self.statusProgressBarSESave.setValue(0)
        self.statusProgressBarSESave.setVisible(0)
        
        # connect signals
        self.importButton.clicked.connect(self._openDirDialog)
        self.statusSignal.connect(self.setStatus)
        self.updateSignal.connect(self._updateSeriesTable)
        self.filterLine.textChanged.connect(self._setFilterString)
        self.imageSlider.valueChanged.connect(self.setSeriesImage)
        self.seriesView.clicked.connect(self._seriesTableClicked)
        self.SaveROIButton.clicked.connect(self.saveAifROI)
        self.GRF_Cal.clicked.connect(self.GRFmodelCalThread)

        self.statusSignalCal.connect(self.setStatusCal)
        self.statusSignalDTICal.connect(self.setStatusDTICal)
        self.statusSignalDTISave.connect(self.setStatusDTISave)
        self.statusSignalDKICal.connect(self.setStatusDKICal)
        self.statusSignalDKISave.connect(self.setStatusDKISave)
        self.statusSignalIVIMCal.connect(self.setStatusIVIMCal)
        self.statusSignalIVIMSave.connect(self.setStatusIVIMSave)

        self.statusSignalMonoECal.connect(self.setStatusMonoECal)
        self.statusSignalMonoESave.connect(self.setStatusMonoESave)
        self.statusSignalSECal.connect(self.setStatusSECal)
        self.statusSignalSESave.connect(self.setStatusSESave)

        self.GFRmodelSaveResult.clicked.connect(self.saveGRFmodelResult)
        self.pushButton_DKI_Save.clicked.connect(self.DKImodelSaveThread)
        #self.pushButton_DTI_Save.clicked.connect(self.saveDTIResult)
        self.pushButton_DTI_Save.clicked.connect(self.DTImodelSaveThread)
        self.pushButton_IVIM_Save.clicked.connect(self.IVIMmodelSaveThread)
        self.pushButton_MonoESave.clicked.connect(self.ADCMonoEmodelSaveThread)
        self.pushButton_SE_Save.clicked.connect(self.SEmodelSaveThread)

        self.CurrentSlice_checkBox.setCheckState(Qt.Checked)

        self.checkBox_CurrentSlice_Diffusion.setCheckState(Qt.Checked)


        self.CurrentSlice_checkBox.stateChanged.connect(self.checkboxstateCS)
        self.AllSlice_checkBox.stateChanged.connect(self.checkboxstateAS)

        self.checkBox_CurrentSlice_Diffusion.stateChanged.connect(self.checkboxstateCSDiff)
        self.checkBox_AllSlices_Diffusion.stateChanged.connect(self.checkboxstateASDiff)

        self.pushButton_ADC_Mono.clicked.connect(self.ADCMonomodelCalThread)
        self.pushButton_SE.clicked.connect(self.SEmodelCalThread)
        self.pushButton_DKI.clicked.connect(self.DKImodelCalThread)
        self.pushButton_DTI.clicked.connect(self.DTImodelCalThread)
        self.pushButton_IVIM.clicked.connect(self.IVIMmodelCalThread)

        # setup the list and table models
        self.srcmodel=QStringListModel()
        self.seriesmodel=SeriesTableModel(self.seriesColumns)
        self.seriesmodel.layoutChanged.connect(self._seriesTableResize)
        self.tagmodel=QtGui.QStandardItemModel()

        # assign models to views
        self.sourceListView.setModel(self.srcmodel)
        self.seriesView.setModel(self.seriesmodel)
        self.tagView.setModel(self.tagmodel)

        # create the pyqtgraph object for viewing images
        self.imageview=pg.ImageView()
        layout=QtGui.QGridLayout(self.view2DGroup)
        layout.addWidget(self.imageview)

        self.pw = pg.PlotWidget(name='T-DCurve')
        layout2 = QtGui.QGridLayout(self.TDCurve)
        layout2.addWidget(self.pw)

        ## Create an empty plot curve to be filled later, set its pen
        self.TDCurvePlt = self.pw.plot()
        self.TDCurvePlt.setPen((200, 200, 100))

        self.PhaseTimeGap.setValue(4.0)

        #
        self.CalCurrentSlice = True
        self.CalAllSlice = False




        
        # load the empty image placeholder into a ndarray
        qimg=QtGui.QImage(':/icons/noimage.png')
        bytedata=qimg.constBits().asstring(qimg.width()*qimg.height())
        self.noimg=np.ndarray((qimg.width(),qimg.height()),dtype=np.ubyte,buffer=bytedata)
        
        # add the directories passed as arguments to the directory queue to start loading
        for i in args:
            if os.path.isdir(i):
                self.addSourceDir(i)

        #self.roi = pg.ImageView().PlotROI(10)

    def checkboxstateCS(self):
        if(self.CurrentSlice_checkBox.checkState() == Qt.Checked):
            self.AllSlice_checkBox.setCheckState(Qt.Unchecked)
        else:
            self.AllSlice_checkBox.setCheckState(Qt.Checked)

    def checkboxstateAS(self):
        if (self.AllSlice_checkBox.checkState() == Qt.Checked):
            self.CurrentSlice_checkBox.setCheckState(Qt.Unchecked)
        else:
            self.CurrentSlice_checkBox.setCheckState(Qt.Checked)


    def checkboxstateCSDiff(self):
        if(self.checkBox_CurrentSlice_Diffusion.checkState() == Qt.Checked):
            self.checkBox_AllSlices_Diffusion.setCheckState(Qt.Unchecked)
        else:
            self.checkBox_AllSlices_Diffusion.setCheckState(Qt.Checked)

    def checkboxstateASDiff(self):
        if (self.checkBox_AllSlices_Diffusion.checkState() == Qt.Checked):
            self.checkBox_CurrentSlice_Diffusion.setCheckState(Qt.Unchecked)
        else:
            self.checkBox_CurrentSlice_Diffusion.setCheckState(Qt.Checked)



    def keyPressEvent(self,e):
        '''Close the window if escape is pressed, otherwise do as inherited.'''
        if e.key() == Qt.Key_Escape:
            self.close()
        else:
            QtGui.QMainWindow.keyPressEvent(self,e)

    def show(self):
        '''Calls the inherited show() method then sets the splitter positions.'''
        QtGui.QMainWindow.show(self)
        self.listSplit.moveSplitter(100,1)
        self.seriesSplit.moveSplitter(80,1)
        self.viewMetaSplitter.moveSplitter(600,1)
        self.tabWidget

    def _loadDirsThread(self):
        '''
        This method is run in a daemon thread and continually checks self.dirQueue for a queued directory to scan for
        Dicom files. It calls loadDicomDir() for a given directory and adds the results the self.srclist member.
        '''
        while True:
            try:
                rootdir=self.dirQueue.get(True,0.5)
                series=loadDicomDir(rootdir,self.statusSignal.emit)
                if series and all(len(s.filenames)>0 for s in series):
                    for s in series:
                        s.filenames,s.dcms=zip(*sorted(zip(s.filenames,s.dcms))) # sort series contents by filename
                        s.getSliceNum()
                        s.getAcquireTimeList()
                        s.getDiffusionVals()
                        s.getPos()
                    self.srclist.append((rootdir,series))

                self.updateSignal.emit()
            except Empty:
                pass

    def _GFRmodelCalThread(self):
        while True:
            try:
                series, phasetimegap, AifData = self.GFRQueue.get(True,0.5)
                if(self.AllSlice_checkBox.checkState() == Qt.Checked):
                    self.m_GRF = GFRmodelCalSeriesMP(series, phasetimegap, AifData,self.statusSignalCal.emit)
                    self.imageview.setImage((self.m_GRF[:, :, 0]).T, autoRange=True, autoLevels=self.autoLevelsCheck.isChecked())
                    self._fillTagView()
                    self.statusProgressBarCal.setRange(0, 100)
                    self.statusProgressBarCal.setValue(0)
                    self.GFRQueue = Queue()
                else:
                    currentIndex = self.imageSlider.value()
                    sliceNum  = series.sliceNum
                    currentSlice = currentIndex%sliceNum
                    TotalNum = len(series.dcms)
                    # timelist = series.acquiretimelist
                    sliceNum = series.sliceNum
                    phaseNum = int(TotalNum / sliceNum)
                    rows = series.getTagObject(0).Rows
                    columns = series.getTagObject(0).Columns
                    deltaT = phasetimegap / 60.
                    self.m_GRF = GFRmodelCal(series,phaseNum,sliceNum,rows,columns,AifData,deltaT,currentSlice,self.statusSignalCal.emit)

                    self.imageview.setImage((self.m_GRF[:, :,0]).T, autoRange=True,
                                            autoLevels=True)
                    self._fillTagView()
                    self.statusProgressBarCal.setRange(0, 100)
                    self.statusProgressBarCal.setValue(0)
                    self.GFRQueue = Queue()

                #series, phasetimegap, AifData =self.GFRQueue.get(False)
                #grf = GFRmodelCalSeries(series, phasetimegap, AifData)
                #self.grf.append((series,grf))
            except Empty:
                pass


    def _DKImodelCalThread(self):
        while True:
            try:
                series = self.DKIQueue.get(True,0.5)
                currentIndex = self.imageSlider.value()
                rows = series.getTagObject(0).Rows
                columns = series.getTagObject(0).Columns

                self.FA = np.zeros((rows,columns,len(series.pos)))
                self.MD= np.zeros((rows,columns,len(series.pos)))
                self.AD= np.zeros((rows,columns,len(series.pos)))
                self.RD= np.zeros((rows,columns,len(series.pos)))
                self.MK= np.zeros((rows,columns,len(series.pos)))
                self.AK= np.zeros((rows,columns,len(series.pos)))
                self.RK = np.zeros((rows,columns,len(series.pos)))
                self.statusProgressBarDKICal.setVisible(1)
                self.statusSignalDKICal.emit("", 0, 1)
                if(self.checkBox_AllSlices_Diffusion.checkState() == Qt.Checked):
                    for p in range(len(series.pos)):
                        pos = series.pos[p]
                        bvalslist = []
                        bvecslist = []
                        totalNum = len(series.dcms)
                        sliceNum = series.sliceNum
                        DuffusionNum = totalNum / sliceNum

                        count = 0
                        dcmlist = []
                        for i in range(totalNum):
                            if series.getTagObject(i).SliceLocation == pos:
                                bval = series.getTagObject(i).DiffusionBValue
                                bvec = series.getTagObject(i).DiffusionGradientOrientation
                                if (not (bval and not (bvec[0] and bvec[1] and bvec[2]))):
                                    bvalslist.append(bval)
                                    bvecslist.append(bvec)
                                    dcmlist.append(i)
                                    count = count + 1
                        data = np.zeros((rows, columns, 1, count))
                        for i in range(count):
                            data[:, :, 0, i] = series.getPixelData(dcmlist[i])

                        FA, MD, AD, RD, MK, AK, RK = DKImodelCal(bvalslist,bvecslist, data,np.ones((rows, columns)))
                        self.FA[:,:,p] = FA[:,:,0]
                        self.MD[:, :, p] = MD[:, :, 0]
                        self.AD[:, :, p] = AD[:, :, 0]
                        self.RD[:, :, p] = RD[:, :, 0]
                        self.MK[:, :, p] = MK[:, :, 0]
                        self.AK[:, :, p] = AK[:, :, 0]
                        self.RK[:, :, p] = RK[:, :, 0]
                    #self.statusSignalDTICal("DTI",p,len(series.pos))
                        self.statusSignalDKICal.emit("DKI", p + 1, len(series.pos))
                    time.sleep(1)
                    self.statusProgressBarDKICal.setVisible(0)
                    self.statusSignalDKICal.emit("", 0, 1)
                    self.imageview.setImage(self.RD[:, :, 0].T, autoRange=True,
                                            autoLevels=True)
                    self._fillTagView()

                else:
                    pos = series.getTagObject(currentIndex).SliceLocation
                    bvalslist = []
                    bvecslist = []
                    totalNum = len(series.dcms)
                    sliceNum = series.sliceNum
                    DuffusionNum = totalNum/sliceNum

                    count = 0
                    dcmlist = []
                    for i in range(totalNum):
                        if series.getTagObject(i).SliceLocation==pos:
                            bval = series.getTagObject(i).DiffusionBValue
                            bvec = series.getTagObject(i).DiffusionGradientOrientation
                            if(not(bval and not(bvec[0] and bvec[1] and bvec[2]))):
                                bvalslist.append(bval)
                                bvecslist.append(bvec)
                                dcmlist.append(i)
                                count = count +1
                    data = np.zeros((rows, columns, 1, count))
                    for i in range(count):
                        data[:, :, 0, i] = series.getPixelData(dcmlist[i])

                    if (self.checkBox.checkState() == Qt.Checked):
                        image = self.imageview.getProcessedImage()
                        if image.ndim == 2:
                            axes = (0, 1)
                        elif image.ndim == 3:
                            axes = (1, 2)
                        else:
                            return

                        data1, coords = self.imageview.roi.getArrayRegion(image.view(np.ndarray),
                                                                          self.imageview.imageItem,
                                                                          axes,
                                                                          returnMappedCoords=True)
                        mask = np.zeros((rows, columns))
                        for i in range(coords.shape[1]):
                            for j in range(coords.shape[2]):
                                mask[int(coords[1, i, j]), int(coords[0, i, j])] = 1
                    else:
                        mask = np.ones((rows, columns))

                    self.FA, self.MD, self.AD, self.RD, self.MK, self.AK, self.RK = DKImodelCal(bvalslist,bvecslist,data,mask)
                    self.statusSignalDKICal.emit("DKI", 1, 1)
                    time.sleep(1)
                    self.statusProgressBarDKICal.setVisible(0)
                    self.statusSignalDKICal.emit("", 0, 1)
                    self.imageview.setImage(self.RD[:,:,0].T, autoRange=True,
                                            autoLevels=True)
                    self._fillTagView()
            except Empty:
                pass

    def _DTImodelCalThread(self):
        while True:
            try:
                series = self.DTIQueue.get(True,0.5)
                currentIndex = self.imageSlider.value()
                rows = series.getTagObject(0).Rows
                columns = series.getTagObject(0).Columns

                self.DTI_FA = np.zeros((rows,columns,len(series.pos)))
                self.DTI_MD= np.zeros((rows,columns,len(series.pos)))
                self.DTI_AD= np.zeros((rows,columns,len(series.pos)))
                self.DTI_RD = np.zeros((rows,columns,len(series.pos)))

                if(self.checkBox_AllSlices_Diffusion.checkState() == Qt.Checked):
                    self.statusProgressBarDTICal.setVisible(1)
                    self.statusSignalDTICal.emit("", 0, 1)
                    for p in range(len(series.pos)):
                        pos = series.pos[p]
                        bvalslist = []
                        bvecslist = []
                        totalNum = len(series.dcms)
                        sliceNum = series.sliceNum
                        DuffusionNum = totalNum / sliceNum

                        count = 0
                        dcmlist = []
                        for i in range(totalNum):
                            if series.getTagObject(i).SliceLocation == pos:
                                bval = series.getTagObject(i).DiffusionBValue
                                bvec = series.getTagObject(i).DiffusionGradientOrientation
                                if (not (bval and not (bvec[0] and bvec[1] and bvec[2]))):
                                    bvalslist.append(bval)
                                    bvecslist.append(bvec)
                                    dcmlist.append(i)
                                    count = count + 1
                        data = np.zeros((rows, columns, 1, count))
                        for i in range(count):
                            data[:, :, 0, i] = series.getPixelData(dcmlist[i])



                        FA, MD, AD, RD = DTImodelCal(bvalslist,bvecslist, data,np.ones((rows, columns)))
                        self.DTI_FA[:,:,p] = FA[:,:,0]
                        self.DTI_MD[:, :, p] = MD[:, :, 0]
                        self.DTI_AD[:, :, p] = AD[:, :, 0]
                        self.DTI_RD[:, :, p] = RD[:, :, 0]
                        self.statusSignalDTICal.emit("DTI", p+1, len(series.pos))
                    time.sleep(1)
                    self.statusProgressBarDTICal.setVisible(0)
                    self.statusSignalDTICal.emit("", 0, 1)
                    self.imageview.setImage(self.DTI_RD[:, :, 0].T, autoRange=True,
                                            autoLevels=True)
                    self._fillTagView()

                else:
                    pos = series.getTagObject(currentIndex).SliceLocation
                    bvalslist = []
                    bvecslist = []
                    totalNum = len(series.dcms)
                    sliceNum = series.sliceNum
                    DuffusionNum = totalNum/sliceNum

                    count = 0
                    dcmlist = []
                    self.statusProgressBarDTICal.setVisible(1)
                    self.statusSignalDTICal.emit("", 0, 0)
                    for i in range(totalNum):
                        if series.getTagObject(i).SliceLocation==pos:
                            bval = series.getTagObject(i).DiffusionBValue
                            bvec = series.getTagObject(i).DiffusionGradientOrientation
                            if(not(bval and not(bvec[0] and bvec[1] and bvec[2]))):
                                bvalslist.append(bval)
                                bvecslist.append(bvec)
                                dcmlist.append(i)
                                count = count +1
                    data = np.zeros((rows, columns, 1, count))
                    for i in range(count):
                        data[:, :, 0, i] = series.getPixelData(dcmlist[i])

                    if (self.checkBox.checkState() == Qt.Checked):
                        image = self.imageview.getProcessedImage()
                        if image.ndim == 2:
                            axes = (0, 1)
                        elif image.ndim == 3:
                            axes = (1, 2)
                        else:
                            return

                        data1, coords = self.imageview.roi.getArrayRegion(image.view(np.ndarray),
                                                                          self.imageview.imageItem,
                                                                          axes,
                                                                          returnMappedCoords=True)
                        mask = np.zeros((rows, columns))
                        for i in range(coords.shape[1]):
                            for j in range(coords.shape[2]):
                                mask[int(coords[1, i, j]), int(coords[0, i, j])] = 1
                    else:
                        mask = np.ones((rows, columns))


                    self.DTI_FA, self.DTI_MD, self.DTI_AD, self.DTI_RD = DTImodelCal(bvalslist,bvecslist,data,mask)

                    self.statusSignalDTICal.emit("DTI", 1, 1)
                    time.sleep(1)
                    self.statusProgressBarDTICal.setVisible(0)
                    self.statusSignalDTICal.emit("", 0, 1)
                    self.imageview.setImage(self.DTI_RD[:,:,0].T, autoRange=True,
                                            autoLevels=True)
                    self._fillTagView()
            except Empty:
                pass

    def _DTImodelSaveThread(self):
        while True:
            try:
                series = self.DTISave.get(True, 0.5)
                self.saveDTIResult()
            except Empty:
                pass

    def _DKImodelSaveThread(self):
        while True:
            try:
                series = self.DKISave.get(True, 0.5)
                self.saveDKIResult()
            except Empty:
                pass

    def _IVIMmodelSaveThread(self):
        while True:
            try:
                series = self.IVIMSave.get(True, 0.5)
                self.saveIVIMResult()
            except Empty:
                pass

    def _ADCMonomodelSaveThread(self):
        while True:
            try:

                series = self.ADCMonoSave.get(True, 0.5)
                self.saveADCMonoResult()
            except Empty:
                pass

    def _SEmodelSaveThread(self):
        while True:
            try:

                series = self.SESave.get(True, 0.5)
                self.saveSEResult()
            except Empty:
                pass

    def _IVIMmodelCalThread(self):
        while True:
            try:
                series = self.IVIMQueue.get(True,0.5)
                currentIndex = self.imageSlider.value()
                rows = series.getTagObject(0).Rows
                columns = series.getTagObject(0).Columns

                self.S0_predicted = np.zeros((rows,columns,len(series.pos)))
                self.perfusion_fraction= np.zeros((rows,columns,len(series.pos)))
                self.D_star= np.zeros((rows,columns,len(series.pos)))
                self.D=np.zeros((rows,columns,len(series.pos)))
                self.statusProgressBarIVIMCal.setVisible(1)
                self.statusSignalIVIMCal.emit("", 0, 1)
                if(self.checkBox_AllSlices_Diffusion.checkState() == Qt.Checked):
                    for p in range(len(series.pos)):
                        pos = series.pos[p]
                        bvalslist = []
                        bvecslist = []
                        totalNum = len(series.dcms)
                        sliceNum = series.sliceNum
                        DuffusionNum = totalNum / sliceNum

                        count = 0
                        dcmlist = []
                        for i in range(totalNum):
                            if series.getTagObject(i).SliceLocation == pos:
                                bval = series.getTagObject(i).DiffusionBValue
                                bvec = series.getTagObject(i).DiffusionGradientOrientation
                                #if (not (bval and not (bvec[0] and bvec[1] and bvec[2]))):
                                bvalslist.append(bval)
                                bvecslist.append(bvec)
                                dcmlist.append(i)
                                count = count + 1
                        data = np.zeros((rows, columns, 1, count))
                        for i in range(count):
                            data[:, :, 0, i] = series.getPixelData(dcmlist[i])



                        S0_predicted, perfusion_fraction, D_star, D = IVIMmodelCal(bvalslist,bvecslist, data,np.ones((rows, columns)))
                        self.S0_predicted[:,:,p] = S0_predicted[:,:,0]
                        self.perfusion_fraction[:, :, p] = perfusion_fraction[:, :, 0]
                        self.D_star[:, :, p] = D_star[:, :, 0]
                        self.D[:, :, p] = D[:, :, 0]
                        self.statusSignalIVIMCal.emit("IVIM", p + 1, len(series.pos))
                    time.sleep(1)
                    self.statusProgressBarIVIMCal.setVisible(0)
                    self.statusSignalIVIMCal.emit("", 0, 1)


                    self.imageview.setImage(self.D_star[:, :, 0].T, autoRange=True,
                                            autoLevels=True)
                    self._fillTagView()

                else:
                    pos = series.getTagObject(currentIndex).SliceLocation
                    bvalslist = []
                    bvecslist = []
                    totalNum = len(series.dcms)
                    sliceNum = series.sliceNum
                    DuffusionNum = totalNum/sliceNum

                    count = 0
                    dcmlist = []
                    for i in range(totalNum):
                        if series.getTagObject(i).SliceLocation==pos:
                            bval = series.getTagObject(i).DiffusionBValue
                            bvec = series.getTagObject(i).DiffusionGradientOrientation
                            #if(not(bval and not(bvec[0] and bvec[1] and bvec[2]))):
                            bvalslist.append(bval)
                            bvecslist.append(bvec)
                            dcmlist.append(i)
                            count = count +1
                    data = np.zeros((rows, columns, 1, count))
                    for i in range(count):
                        data[:, :, 0, i] = series.getPixelData(dcmlist[i])


                    if(self.checkBox.checkState() == Qt.Checked):
                        image = self.imageview.getProcessedImage()
                        if image.ndim == 2:
                            axes = (0, 1)
                        elif image.ndim == 3:
                            axes = (1, 2)
                        else:
                            return

                        data1, coords = self.imageview.roi.getArrayRegion(image.view(np.ndarray), self.imageview.imageItem,
                                                                          axes,
                                                                          returnMappedCoords=True)
                        mask = np.zeros((rows, columns))
                        for i in range(coords.shape[1]):
                            for j in range(coords.shape[2]):
                                mask[int(coords[1, i, j]), int(coords[0, i, j])] = 1
                    else:
                        mask = np.ones((rows, columns))

                    self.S0_predicted, self.perfusion_fraction, self.D_star, self.D = IVIMmodelCal(bvalslist, bvecslist, data,mask)
                    self.statusSignalIVIMCal.emit("IVIM", 1, 1)
                    time.sleep(1)
                    self.statusProgressBarIVIMCal.setVisible(0)
                    self.statusSignalIVIMCal.emit("", 0, 1)
                    self.imageview.setImage(self.S0_predicted[:,:,0].T, autoRange=True,
                                            autoLevels=True)
                    self._fillTagView()
            except Empty:
                pass


    def _ADCMonomodelCalThread(self):
        while True:
            try:
                series = self.ADCMonoQueue.get(True, 0.5)
                currentIndex = self.imageSlider.value()
                rows = series.getTagObject(0).Rows
                columns = series.getTagObject(0).Columns


                self.ADCm = np.zeros((rows, columns, len(series.pos)))
                self.statusProgressBarMonoECal.setVisible(1)
                self.statusSignalMonoECal.emit("", 0, 1)
                if (self.checkBox_AllSlices_Diffusion.checkState() == Qt.Checked):

                    for p in range(len(series.pos)):
                        pos = series.pos[p]
                        bvalslist = []
                        bvecslist = []
                        totalNum = len(series.dcms)
                        sliceNum = series.sliceNum
                        DuffusionNum = totalNum / sliceNum

                        count = 0
                        dcmlist = []
                        bvec0 = series.getTagObject(0).DiffusionGradientOrientation
                        for i in range(totalNum):
                            if series.getTagObject(i).SliceLocation == pos:
                                bval = series.getTagObject(i).DiffusionBValue
                                bvec = series.getTagObject(i).DiffusionGradientOrientation
                                if (bvec == bvec0):
                                    bvalslist.append(bval)
                                    dcmlist.append(i)
                                    count = count + 1
                                    # bvecslist.append(bvec)
                                    # if(not(bval and not(bvec[0] and bvec[1] and bvec[2]))):
                        data = np.zeros((rows, columns, 1, count))
                        for i in range(count):
                            data[:, :, 0, i] = series.getPixelData(dcmlist[i])

                        ADCm = ADCMonomodelCal(bvalslist, data,np.ones((rows, columns)))
                        self.statusSignalMonoECal.emit("MonoE", p + 1, len(series.pos))
                        self.ADCm[:,:,p] = ADCm[:,:,0]




                else:
                    pos = series.getTagObject(currentIndex).SliceLocation
                    bvalslist = []
                    bvecslist = []
                    totalNum = len(series.dcms)
                    sliceNum = series.sliceNum
                    DuffusionNum = totalNum / sliceNum

                    count = 0
                    dcmlist = []
                    bvec0 = series.getTagObject(0).DiffusionGradientOrientation
                    for i in range(totalNum):
                        if series.getTagObject(i).SliceLocation == pos:
                            bval = series.getTagObject(i).DiffusionBValue
                            bvec = series.getTagObject(i).DiffusionGradientOrientation
                            if(bvec==bvec0):
                                bvalslist.append(bval)
                                dcmlist.append(i)
                                count = count + 1
                                #bvecslist.append(bvec)
                            # if(not(bval and not(bvec[0] and bvec[1] and bvec[2]))):
                    data = np.zeros((rows, columns, 1,count))
                    for i in range(count):
                        data[:, :,0, i] = series.getPixelData(dcmlist[i])

                    if (self.checkBox.checkState() == Qt.Checked):
                        image = self.imageview.getProcessedImage()
                        if image.ndim == 2:
                            axes = (0, 1)
                        elif image.ndim == 3:
                            axes = (1, 2)
                        else:
                            return

                        data1, coords = self.imageview.roi.getArrayRegion(image.view(np.ndarray),
                                                                          self.imageview.imageItem,
                                                                          axes,
                                                                          returnMappedCoords=True)
                        mask = np.zeros((rows, columns))
                        for i in range(coords.shape[1]):
                            for j in range(coords.shape[2]):
                                mask[int(coords[1, i, j]), int(coords[0, i, j])] = 1
                    else:
                        mask = np.ones((rows, columns))

                    self.ADCm = ADCMonomodelCal(bvalslist,data,mask)
                    self.statusSignalMonoECal.emit("MonoE", 1, 1)

                time.sleep(1)
                self.statusProgressBarMonoECal.setVisible(0)
                self.statusSignalMonoECal.emit("", 0, 1)

                self.imageview.setImage(self.ADCm[:, :, 0].T, autoRange=True,
                                        autoLevels=True)
                self._fillTagView()





            except Empty:
                pass

    def _SEmodelCalThread(self):
        while True:
            try:
                series = self.SEQueue.get(True, 0.5)
                currentIndex = self.imageSlider.value()
                rows = series.getTagObject(0).Rows
                columns = series.getTagObject(0).Columns


                self.ADCs = np.zeros((rows,columns,len(series.pos)))
                self.Alpha = np.zeros((rows, columns, len(series.pos)))
                self.statusProgressBarSECal.setVisible(1)
                self.statusSignalSECal.emit("", 0, 1)
                if (self.checkBox_AllSlices_Diffusion.checkState() == Qt.Checked):

                    for p in range(len(series.pos)):
                        pos = series.pos[p]
                        bvalslist = []
                        bvecslist = []
                        totalNum = len(series.dcms)
                        sliceNum = series.sliceNum
                        DuffusionNum = totalNum / sliceNum

                        count = 0
                        dcmlist = []
                        bvec0 = series.getTagObject(0).DiffusionGradientOrientation
                        for i in range(totalNum):
                            if series.getTagObject(i).SliceLocation == pos:
                                bval = series.getTagObject(i).DiffusionBValue
                                bvec = series.getTagObject(i).DiffusionGradientOrientation
                                if (bvec == bvec0):
                                    bvalslist.append(bval)
                                    dcmlist.append(i)
                                    count = count + 1
                                    # bvecslist.append(bvec)
                                    # if(not(bval and not(bvec[0] and bvec[1] and bvec[2]))):
                        data = np.zeros((rows, columns, 1, count))
                        for i in range(count):
                            data[:, :, 0, i] = series.getPixelData(dcmlist[i])

                        ADCs,Alpha = SEmodelCal(bvalslist, data,np.ones((rows, columns)))
                        self.statusSignalSECal.emit("SE", p + 1, len(series.pos))
                        self.ADCs[:,:,p] = ADCs[:,:,0]
                        self.Alpha[:,:,p] = Alpha[:,:,0]




                else:
                    pos = series.getTagObject(currentIndex).SliceLocation
                    bvalslist = []
                    bvecslist = []
                    totalNum = len(series.dcms)
                    sliceNum = series.sliceNum
                    DuffusionNum = totalNum / sliceNum

                    count = 0
                    dcmlist = []
                    bvec0 = series.getTagObject(0).DiffusionGradientOrientation
                    for i in range(totalNum):
                        if series.getTagObject(i).SliceLocation == pos:
                            bval = series.getTagObject(i).DiffusionBValue
                            bvec = series.getTagObject(i).DiffusionGradientOrientation
                            if(bvec==bvec0):
                                bvalslist.append(bval)
                                dcmlist.append(i)
                                count = count + 1
                                #bvecslist.append(bvec)
                            # if(not(bval and not(bvec[0] and bvec[1] and bvec[2]))):
                    data = np.zeros((rows, columns, 1,count))
                    for i in range(count):
                        data[:, :,0, i] = series.getPixelData(dcmlist[i])

                    if (self.checkBox.checkState() == Qt.Checked):
                        image = self.imageview.getProcessedImage()
                        if image.ndim == 2:
                            axes = (0, 1)
                        elif image.ndim == 3:
                            axes = (1, 2)
                        else:
                            return

                        data1, coords = self.imageview.roi.getArrayRegion(image.view(np.ndarray),
                                                                          self.imageview.imageItem,
                                                                          axes,
                                                                          returnMappedCoords=True)
                        mask = np.zeros((rows, columns))
                        for i in range(coords.shape[1]):
                            for j in range(coords.shape[2]):
                                mask[int(coords[1, i, j]), int(coords[0, i, j])] = 1
                    else:
                        mask = np.ones((rows, columns))

                    self.ADCs,self.Alpha = SEmodelCal(bvalslist,data,mask)
                    self.statusSignalSECal.emit("SE", 1, 1)

                time.sleep(1)
                self.statusProgressBarSECal.setVisible(0)
                self.statusSignalSECal.emit("", 0, 1)

                self.imageview.setImage(self.ADCs[:, :, 0].T, autoRange=True,
                                        autoLevels=True)
                self._fillTagView()





            except Empty:
                pass












    def _openDirDialog(self):
        '''Opens the open file dialog to choose a directory to scan for Dicoms.'''
        rootdir=str(QtGui.QFileDialog.getExistingDirectory(self,'Choose Source Directory',self.lastDir))
        if rootdir:
            self.addSourceDir(rootdir)

    def _updateSeriesTable(self):
        '''
        Updates the self.seriesMap object from self.srclist, and refills the self.srcmodel object. This will refresh 
        the list of source directories and the table of available series.
        '''
        self.seriesMap.clear()

        for _,series in self.srclist: # add each series in each source into self.seriesMap 
            for s in series:
                entry=s.getTagValues(self.seriesColumns)
                self.seriesMap[entry]=s

        self.srcmodel.setStringList([s[0] for s in self.srclist])
        self.seriesmodel.updateSeriesTable(self.seriesMap.keys())
        self.seriesmodel.layoutChanged.emit()

    def _seriesTableClicked(self,item):
        '''Called when a series is clicked on, set the viewed image to be from the clicked series.'''
        self.selectedRow=item.row()
        self.setSeriesImage(self.imageSlider.value(),True)
        
    def _seriesTableResize(self):
        '''Resizes self.seriesView columns to contents, setting the last section to stretch.'''
        self.seriesView.horizontalHeader().setStretchLastSection(False)
        self.seriesView.resizeColumnsToContents()
        self.seriesView.horizontalHeader().setStretchLastSection(True)
            
    def _setFilterString(self,regex):
        '''Set the filtering regex to be `regex'.'''
        self.filterRegex=regex
        self._fillTagView()
            
    def _fillTagView(self):
        '''Refill the Dicom tag view, this will rejig the columns and (unfortunately) reset column sorting.'''
        series=self.getSelectedSeries()
        vpos=self.tagView.verticalScrollBar().value()
        self.tagmodel.clear()
        self.tagmodel.setHorizontalHeaderLabels(tagTreeColumns)
        fillTagModel(self.tagmodel,series.getTagObject(self.imageIndex),self.filterRegex)
        self.tagView.expandAll()
        self.tagView.resizeColumnToContents(0)
        self.tagView.verticalScrollBar().setValue(vpos)
        
    def getSelectedSeries(self):
        '''Returns the DicomSeries object for the selected series, None if no series is selected.'''
        if 0<=self.selectedRow<len(self.seriesMap):
            return self.seriesMap[self.seriesmodel.getRow(self.selectedRow)]

    def setSeriesImage(self,i,autoRange=False):
        '''
        Set the view image to be that at index `i' of the selected series. The `autoRange' boolean value sets whether
        the data value range is reset or not when this is done. The tag table is also set to that of image `i'.
        '''
        series=self.getSelectedSeries()
        if series:
            maxindex=len(series.filenames)-1
            self.imageIndex=np.clip(i,0,maxindex)
            img=series.getPixelData(self.imageIndex) # image matrix
            interval=1 # tick interval on the slider
            
            # choose a more sensible tick interval if there's a lot of images
            if maxindex>=5000:
                interval=100
            elif maxindex>=500:
                interval=10
            
            if img is None: # if the image is None use the default "no image" object
                img=self.noimg
            #elif len(img.shape)==3: # multi-channel or multi-dimensional image, use average of dimensions
            #    img=np.mean(img,axis=2)

            self.imageview.setImage(img.T,autoRange=autoRange,autoLevels=self.autoLevelsCheck.isChecked())
            self._fillTagView()
            self.imageSlider.setTickInterval(interval)
            self.imageSlider.setMaximum(maxindex)
            self.numLabel.setText(str(self.imageIndex))
            self.view2DGroup.setTitle('2D View - '+series.filenames[self.imageIndex])
            
    def setStatus(self,msg,progress=0,progressmax=0):
        '''
        Set the status bar with message `msg' with progress set to `progress' out of `progressmax', or hide the status 
        elements if `msg' is empty or None.
        '''
        if not msg:
            progress=0
            progressmax=0

        self.statusText.setText(msg)
        self.statusText.setVisible(bool(msg))
        self.importButton.setVisible(not bool(msg))
        self.statusProgressBar.setVisible(progressmax>0)
        self.statusProgressBar.setRange(0,progressmax)
        self.statusProgressBar.setValue(progress)

    def setStatusCal(self,msg,progress=0,progressmax=0):
        if not msg:
            progress=0
            progressmax=0
        #self.statusProgressBarCal.setVisible(progressmax>0)
        self.statusProgressBarCal.setRange(0,progressmax)
        self.statusProgressBarCal.setValue(progress)

    def setStatusDTICal(self,msg,progress=0,progressmax=0):
        if not msg:
            progress=0
            progressmax=0
        #self.statusProgressBarCal.setVisible(progressmax>0)
        self.statusProgressBarDTICal.setRange(0,progressmax)
        self.statusProgressBarDTICal.setValue(progress)

    def setStatusDTISave(self,msg,progress=0,progressmax=0):
        if not msg:
            progress=0
            progressmax=0

        self.statusProgressBarDTISave.setRange(0,progressmax)
        self.statusProgressBarDTISave.setValue(progress)

    def setStatusDKICal(self,msg,progress=0,progressmax=0):
        if not msg:
            progress=0
            progressmax=0
        #self.statusProgressBarCal.setVisible(progressmax>0)
        self.statusProgressBarDKICal.setRange(0,progressmax)
        self.statusProgressBarDKICal.setValue(progress)

    def setStatusDKISave(self,msg,progress=0,progressmax=0):
        if not msg:
            progress=0
            progressmax=0

        self.statusProgressBarDKISave.setRange(0,progressmax)
        self.statusProgressBarDKISave.setValue(progress)

    def setStatusIVIMCal(self,msg,progress=0,progressmax=0):
        if not msg:
            progress=0
            progressmax=0
        #self.statusProgressBarCal.setVisible(progressmax>0)
        self.statusProgressBarIVIMCal.setRange(0,progressmax)
        self.statusProgressBarIVIMCal.setValue(progress)

    def setStatusIVIMSave(self,msg,progress=0,progressmax=0):
        if not msg:
            progress=0
            progressmax=0

        self.statusProgressBarIVIMSave.setRange(0,progressmax)
        self.statusProgressBarIVIMSave.setValue(progress)


    def setStatusMonoECal(self,msg,progress=0,progressmax=0):
        if not msg:
            progress=0
            progressmax=0
        #self.statusProgressBarCal.setVisible(progressmax>0)
        self.statusProgressBarMonoECal.setRange(0,progressmax)
        self.statusProgressBarMonoECal.setValue(progress)


    def setStatusMonoESave(self,msg,progress=0,progressmax=0):
        if not msg:
            progress=0
            progressmax=0

        self.statusProgressBarMonoESave.setRange(0,progressmax)
        self.statusProgressBarMonoESave.setValue(progress)

    def setStatusSECal(self,msg,progress=0,progressmax=0):
        if not msg:
            progress=0
            progressmax=0
        #self.statusProgressBarCal.setVisible(progressmax>0)
        self.statusProgressBarSECal.setRange(0,progressmax)
        self.statusProgressBarSECal.setValue(progress)


    def setStatusSESave(self,msg,progress=0,progressmax=0):
        if not msg:
            progress=0
            progressmax=0

        self.statusProgressBarSESave.setRange(0,progressmax)
        self.statusProgressBarSESave.setValue(progress)

    def removeSourceDir(self,index):
        '''Remove the source directory at the given index.'''
        self.srclist.pop(index)
        self.updateSignal.emit()

    def addSourceDir(self,rootdir):
        '''Add the given directory to the queue of directories to load and set the self.lastDir value to its parent.'''
        self.dirQueue.put(rootdir)
        self.lastDir=os.path.dirname(rootdir)

    def saveAifROI(self):

        self.Aifroi = self.imageview.roi
        try:
            self.setAifCurve()
            self.showCurve()
        except:
            return



    def setAifCurve(self):
        imageview = pg.ImageView()
        series = self.getSelectedSeries()
        TotalNum = len(series.dcms)
        #timelist = series.acquiretimelist
        self.sliceNum = series.sliceNum
        self.phaseNum = int(TotalNum/self.sliceNum)

        aifdata = np.arange(self.phaseNum)
        if series:
            maxindex = len(series.filenames)
            #imageIndex = np.clip(i, 0, maxindex)
            #imageIndex = 1
            for i in range(maxindex):
                if(i%self.sliceNum==self.imageIndex%self.sliceNum):
                    img = series.getPixelData(i)  # image matrix
                    if img is None:  # if the image is None use the default "no image" object
                        img = self.noimg

                    imageview.setImage(img.T, autoRange=True, autoLevels=self.autoLevelsCheck.isChecked())
                    state = self.Aifroi.getState()
                    imageview.roi.setState(state)

                    image = imageview.getProcessedImage()
                    if image.ndim == 2:
                        axes = (0, 1)
                    elif image.ndim == 3:
                        axes = (1, 2)
                    else:
                        return

                    data, coords = imageview.roi.getArrayRegion(image.view(np.ndarray), imageview.imageItem, axes,
                                                           returnMappedCoords=True)
                    aifdata[int(i/self.sliceNum)] = data.mean()
        temp = aifdata[0]
        aifdata = [i-temp for i in aifdata]
        #temp = timelist[0]
        #timelist = [float(i)-float(temp) for i in timelist]

        self.AifData = aifdata
            #x = state['pos'].x()
            #y =  state['pos'].y()
            #state['size'].x()
            #state['size'].y()
            #roidata = img[int(state['pos'].y()):int(state['pos'].y()+state['size'].y()),int(state['pos'].x()):int(state['pos'].x()+state['size'].x())]
            #roidataMean = roidata.mean()
           #size = self.Aifroi.size

            #roidata = img
            # elif len(img.shape)==3: # multi-channel or multi-dimensional image, use average of dimensions
            #    img=np.mean(img,axis=2)
    #pass

    def showCurve(self):

        #from PyQt5 import QtWidgets
#
        ##self.verticalLayout_3 = QtWidgets.QWidget()
        ##self.verticalLayout_3.setGeometry(QtCore.QRect(180, 10, 1100, 500))  # 定义gridLayout控件的大小和位置，4个数字分别为左边坐标，上边坐标，长，宽
        ##self.verticalLayout_3.setObjectName("gridLayoutWidget")
        ##self.gridLayout_3 = QtWidgets.QGridLayout(self.verticalLayout_3)
        ##self.gridLayout_3.setContentsMargins(0, 0, 0, 0)  # 在gridLayoutWidget 上创建一个网格Layout，注意以gridLayoutWidget为参
        ##self.gridLayout_3.setObjectName("gridLayout_3")
        ### ===通过graphicview来显示图形
        ##self.curveplot = QtWidgets.QGraphicsView(
        ##    self.verticalLayout_3)  # 第一步，创建一个QGraphicsView，注意同样以gridLayoutWidget为参
        ##self.curveplot.setObjectName("graphicview")
        ##self.gridLayout_3.addWidget(self.curveplot, 0,  0)
#
#
        #dr = Figure_Canvas()

#
        #dr.line(phase,self.AifData)
        #dr.axes.set_autoscale_on(True)
        #graphicscene = QtWidgets.QGraphicsScene()  # 第三步，创建一个QGraphicsScene，因为加载的图形（FigureCanvas）不能直接放到graphicview控件中，必须先放到graphicScene，然后再把graphicscene放到graphicview中
#
#
        #self.curveplot.setScene(graphicscene)  # 第五步，把QGraphicsScene放入QGraphicsView
#
        #graphicscene.addWidget(dr)  # 第四步，把图形放到QGraphicsScene中，注意：图形是作为一个QWidget放到QGraphicsScene中的
        ##self.curveplot.fitInView(graphicscene.sceneRect(), Qt.IgnoreAspectRatio)#graphicscene.fitInView()
        #self.curveplot.show()  # 最后，调用show方法呈现图形！Voila!!
        if self.Aifroi:
            phase = range(self.phaseNum)
            phase = [i+1 for i in phase]
            self.pw.clear()
            ## Add in some extra graphics
            #rect = QtGui.QGraphicsRectItem(QtCore.QRectF(0, 0, 1, 5e-11))
            #rect.setPen(pg.mkPen(100, 200, 100))
            #self.pw.addItem(rect)
            self.pw.setLabel('left', 'Density')
            self.pw.setLabel('bottom', 'Phases')

            self.pw.plot().setData(y=self.AifData, x=list(phase))


    def DKImodelCalThread(self):
        series = self.getSelectedSeries()
        self.DKIQueue.put((series))
        pass

    def ADCMonomodelCalThread(self):
        series = self.getSelectedSeries()
        self.ADCMonoQueue.put((series))
        pass
    def SEmodelCalThread(self):
        series = self.getSelectedSeries()
        self.SEQueue.put((series))
        pass


    def DTImodelCalThread(self):
        series = self.getSelectedSeries()
        self.DTIQueue.put((series))
        pass

    def DTImodelSaveThread(self):
        series = self.getSelectedSeries()
        self.DTISave.put((series))
        pass

    def DKImodelSaveThread(self):
        series = self.getSelectedSeries()
        self.DKISave.put((series))
        pass

    def ADCMonoEmodelSaveThread(self):
        series = self.getSelectedSeries()
        self.ADCMonoSave.put((series))
        pass

    def SEmodelSaveThread(self):
        series = self.getSelectedSeries()
        self.SESave.put((series))
        pass

    def IVIMmodelSaveThread(self):
        series = self.getSelectedSeries()
        self.IVIMSave.put((series))
        pass
    def IVIMmodelCalThread(self):
        series = self.getSelectedSeries()
        self.IVIMQueue.put((series))
        pass

    def IVIMmodelCalThread(self):
        series = self.getSelectedSeries()
        self.IVIMQueue.put((series))
        pass

    def GRFmodelCalThread(self):
        series = self.getSelectedSeries()
        phasetimegap = self.PhaseTimeGap.value()
        AifData = self.AifData

        self.GFRQueue.put((series,phasetimegap,AifData))
        #t = ThreadWithReturnValue(target=GFRmodelCalSeries,args = (series,phasetimegap,AifData))
        #t.daemon = True
        #t.start()
        #m_GRF = t.join()
#
        #self.imageview.setImage((m_GRF[:, :, 0]).T, autoRange=True, autoLevels=self.autoLevelsCheck.isChecked())
        #self._fillTagView()

    def GRFmodelCal(self):
        series = self.getSelectedSeries()
        phasetimegap = self.PhaseTimeGap.value()

        m_GRF = GFRmodelCalSeries(series,phasetimegap,self.AifData)
        self.imageview.setImage((m_GRF[:,:,0]).T, autoRange=True, autoLevels=self.autoLevelsCheck.isChecked())
        self._fillTagView()



        #TotalNum = len(series.dcms)
        ## timelist = series.acquiretimelist
        #sliceNum = series.sliceNum
        #phaseNum = int(TotalNum / self.sliceNum)
#
        #rows = series.getTagObject(0).Rows
        #columns = series.getTagObject(0).Columns
#
        ##data = np.zeros((rows,columns,sliceNum,phaseNum))
#
        ##for i in range(TotalNum):data[:,:,int(i%sliceNum),int(i/sliceNum)]=series.getPixelData(i)
#
        #dcms = np.zeros((rows,columns,phaseNum))
        #Croi = np.zeros((phaseNum,1))
        #A = np.zeros((phaseNum,2))
        #C = np.zeros((phaseNum,1))
#
        #phasetimegap = self.PhaseTimeGap.value()
#
        #deltaT = 4./60.
        #m_GRF = np.zeros((rows,columns,sliceNum))
#
        #for s in range(sliceNum):
        #    for p in range(phaseNum): dcms[:,:,p]=series.getPixelData(p*sliceNum+s)
        #    for i in range(rows):
        #        for j in range(columns):
        #            for p in range(phaseNum):
        #                if(p==0):
        #                    Croi[p] = dcms[i,j,p]
        #                else:
        #                    Croi[p] = dcms[i,j,p]-Croi[0]
        #            cumAif =0
        #            for p in range(phaseNum):
        #                cumAif += self.AifData[p]*deltaT
        #                A[p,0] = cumAif
        #                A[p,1] = self.AifData[p]
        #                C[p] = Croi[p]
        #            B = np.dot(np.linalg.pinv(A),C)
        #            m_GRF[i,j,s] = B[0]
#

    def saveGRFmodelResult(self):
        import time,os
        dir = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
        os.mkdir(dir)



        series = self.getSelectedSeries()
        dcm = dicomio.read_file(series.filenames[0])

        rslope = float(dcm.get('RescaleSlope', 1))
        rinter = float(dcm.get('RescaleIntercept', 0))
        #GRF = np.zeros((192,256,1),dtype='uint16')
        sliceNum = self.m_GRF.shape[2]
        for s in range(sliceNum):
            img = self.m_GRF[:,:,s]
            img = img*100
            #img.dtype = 'uint16'
            #img2 = dcm.pixel_array
            dcm.pixel_array.flat = img.flat
            dcm.PixelData = dcm.pixel_array.tostring()
            dcm.save_as(os.path.join(dir,str(s)))

        self.addSourceDir(dir)
        #if dcm.pixel_array is not None:
        #    rslope = float(dcm.get('RescaleSlope', 1))
        #    rinter = float(dcm.get('RescaleIntercept', 0))
        #    img = dcm.pixel_array * rslope + rinter

    def saveDKIResult(self):
        import time,os
        dir = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
        os.mkdir(dir)
        series = self.getSelectedSeries()
        dcm = dicomio.read_file(series.filenames[0])
        rslope = float(dcm.get('RescaleSlope', 1))
        rinter = float(dcm.get('RescaleIntercept', 0))
        #GRF = np.zeros((192,256,1),dtype='uint16')
        sliceNum = self.FA.shape[2]
        self.statusSignalDKISave.emit("", 0, 0)
        self.statusProgressBarDKISave.setVisible(1)

        RangeSize = 4096.
        dcm.SeriesDescription = 'DKIParaMap'

        for s in range(sliceNum):
            img = self.FA[:,:,s]

            rslope = RangeSize / float(img.max() - img.min())
            rinter = rslope * img.min()
            img = img * rslope - rinter

            dcm.RealWorldValueMappingSequence[0].RealWorldValueSlope = rslope
            dcm.RealWorldValueMappingSequence[0].RealWorldValueIntercept = rinter

            #img.dtype = 'uint16'
            #img2 = dcm.pixel_array
            dcm.pixel_array.flat = img.flat
            dcm.PixelData = dcm.pixel_array.tostring()
            dcm.save_as(os.path.join(dir,str(s)+'_FA'))

            img = self.MD[:, :, s]
            rslope = RangeSize / float(img.max() - img.min())
            rinter = rslope * img.min()
            img = img * rslope - rinter

            dcm.RealWorldValueMappingSequence[0].RealWorldValueSlope = rslope
            dcm.RealWorldValueMappingSequence[0].RealWorldValueIntercept = rinter
            # img.dtype = 'uint16'
            # img2 = dcm.pixel_array
            dcm.pixel_array.flat = img.flat
            dcm.PixelData = dcm.pixel_array.tostring()
            dcm.save_as(os.path.join(dir, str(s) + '_MD'))

            img = self.AD[:, :, s]
            rslope = RangeSize / float(img.max() - img.min())
            rinter = rslope * img.min()
            img = img * rslope - rinter

            dcm.RealWorldValueMappingSequence[0].RealWorldValueSlope = rslope
            dcm.RealWorldValueMappingSequence[0].RealWorldValueIntercept = rinter
            # img.dtype = 'uint16'
            # img2 = dcm.pixel_array
            dcm.pixel_array.flat = img.flat
            dcm.PixelData = dcm.pixel_array.tostring()
            dcm.save_as(os.path.join(dir, str(s) + '_AD'))

            img = self.RD[:, :, s]
            rslope = RangeSize / float(img.max() - img.min())
            rinter = rslope * img.min()
            img = img * rslope - rinter

            dcm.RealWorldValueMappingSequence[0].RealWorldValueSlope = rslope
            dcm.RealWorldValueMappingSequence[0].RealWorldValueIntercept = rinter
            # img.dtype = 'uint16'
            # img2 = dcm.pixel_array
            dcm.pixel_array.flat = img.flat
            dcm.PixelData = dcm.pixel_array.tostring()
            dcm.save_as(os.path.join(dir, str(s) + '_RD'))

            img = self.MK[:, :, s]
            rslope = RangeSize / float(img.max() - img.min())
            rinter = rslope * img.min()
            img = img * rslope - rinter

            dcm.RealWorldValueMappingSequence[0].RealWorldValueSlope = rslope
            dcm.RealWorldValueMappingSequence[0].RealWorldValueIntercept = rinter
            # img.dtype = 'uint16'
            # img2 = dcm.pixel_array
            dcm.pixel_array.flat = img.flat
            dcm.PixelData = dcm.pixel_array.tostring()
            dcm.save_as(os.path.join(dir, str(s) + '_MK'))

            img = self.AK[:, :, s]
            rslope = RangeSize / float(img.max() - img.min())
            rinter = rslope * img.min()
            img = img * rslope - rinter

            dcm.RealWorldValueMappingSequence[0].RealWorldValueSlope = rslope
            dcm.RealWorldValueMappingSequence[0].RealWorldValueIntercept = rinter
            # img.dtype = 'uint16'
            # img2 = dcm.pixel_array
            dcm.pixel_array.flat = img.flat
            dcm.PixelData = dcm.pixel_array.tostring()
            dcm.save_as(os.path.join(dir, str(s) + '_AK'))

            img = self.RK[:, :, s]
            rslope = RangeSize / float(img.max() - img.min())
            rinter = rslope * img.min()
            img = img * rslope - rinter

            dcm.RealWorldValueMappingSequence[0].RealWorldValueSlope = rslope
            dcm.RealWorldValueMappingSequence[0].RealWorldValueIntercept = rinter
            # img.dtype = 'uint16'
            # img2 = dcm.pixel_array
            dcm.pixel_array.flat = img.flat
            dcm.PixelData = dcm.pixel_array.tostring()
            dcm.save_as(os.path.join(dir, str(s) + '_RK'))


            import pandas as pd
            data_df = pd.DataFrame(self.FA[:,:,s])
            writer = pd.ExcelWriter(os.path.join(dir, str(s) + '_FA'+'.xlsx'))
            data_df.to_excel(writer, 'page_1', float_format='%.5f')  # float_format 控制精度
            writer.save()
            data_df = pd.DataFrame(self.MD[:, :, s])
            writer = pd.ExcelWriter(os.path.join(dir, str(s) + '_MD' + '.xlsx'))
            data_df.to_excel(writer, 'page_1', float_format='%.5f')  # float_format 控制精度
            writer.save()
            data_df = pd.DataFrame(self.RD[:, :, s])
            writer = pd.ExcelWriter(os.path.join(dir, str(s) + '_RD' + '.xlsx'))
            data_df.to_excel(writer, 'page_1', float_format='%.5f')  # float_format 控制精度
            writer.save()
            data_df = pd.DataFrame(self.AD[:, :, s])
            writer = pd.ExcelWriter(os.path.join(dir, str(s) + '_AD' + '.xlsx'))
            data_df.to_excel(writer, 'page_1', float_format='%.5f')  # float_format 控制精度
            writer.save()
            data_df = pd.DataFrame(self.MK[:, :, s])
            writer = pd.ExcelWriter(os.path.join(dir, str(s) + '_MK' + '.xlsx'))
            data_df.to_excel(writer, 'page_1', float_format='%.5f')  # float_format 控制精度
            writer.save()
            data_df = pd.DataFrame(self.RK[:, :, s])
            writer = pd.ExcelWriter(os.path.join(dir, str(s) + '_RK' + '.xlsx'))
            data_df.to_excel(writer, 'page_1', float_format='%.5f')  # float_format 控制精度
            writer.save()
            data_df = pd.DataFrame(self.AK[:, :, s])
            writer = pd.ExcelWriter(os.path.join(dir, str(s) + '_AK' + '.xlsx'))
            data_df.to_excel(writer, 'page_1', float_format='%.5f')  # float_format 控制精度
            writer.save()

            self.statusSignalDKISave.emit("DKI save", s + 1, sliceNum)
        time.sleep(1)
        self.statusProgressBarDKISave.setVisible(0)
        self.addSourceDir(dir)

    def saveIVIMResult(self):
        import time,os
        dir = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
        os.mkdir(dir)
        series = self.getSelectedSeries()
        dcm = dicomio.read_file(series.filenames[0])
        rslope = float(dcm.get('RescaleSlope', 1))
        rinter = float(dcm.get('RescaleIntercept', 0))
        #GRF = np.zeros((192,256,1),dtype='uint16')
        sliceNum = self.S0_predicted.shape[2]

        self.statusSignalIVIMSave.emit("", 0, 0)
        self.statusProgressBarIVIMSave.setVisible(1)

        RangeSize = 4096.
        dcm.SeriesDescription = 'IVIMParaMap'
        for s in range(sliceNum):
            img = self.S0_predicted[:,:,s]
            rslope = RangeSize / float(img.max() - img.min())
            rinter = rslope * img.min()
            img = img * rslope - rinter

            dcm.RealWorldValueMappingSequence[0].RealWorldValueSlope = rslope
            dcm.RealWorldValueMappingSequence[0].RealWorldValueIntercept = rinter
            #img.dtype = 'uint16'
            #img2 = dcm.pixel_array
            dcm.pixel_array.flat = img.flat
            dcm.PixelData = dcm.pixel_array.tostring()
            dcm.save_as(os.path.join(dir,str(s)+'_S0_predicted'))

            img = self.perfusion_fraction[:, :, s]
            rslope = RangeSize / float(img.max() - img.min())
            rinter = rslope * img.min()
            img = img * rslope - rinter

            dcm.RealWorldValueMappingSequence[0].RealWorldValueSlope = rslope
            dcm.RealWorldValueMappingSequence[0].RealWorldValueIntercept = rinter
            # img.dtype = 'uint16'
            # img2 = dcm.pixel_array
            dcm.pixel_array.flat = img.flat
            dcm.PixelData = dcm.pixel_array.tostring()
            dcm.save_as(os.path.join(dir, str(s) + '_perfusion_fraction'))

            img = self.D_star[:, :, s]
            rslope = RangeSize / float(img.max() - img.min())
            rinter = rslope * img.min()
            img = img * rslope - rinter

            dcm.RealWorldValueMappingSequence[0].RealWorldValueSlope = rslope
            dcm.RealWorldValueMappingSequence[0].RealWorldValueIntercept = rinter
            # img.dtype = 'uint16'
            # img2 = dcm.pixel_array
            dcm.pixel_array.flat = img.flat
            dcm.PixelData = dcm.pixel_array.tostring()
            dcm.save_as(os.path.join(dir, str(s) + '_D_star'))

            img = self.D[:, :, s]
            rslope = RangeSize / float(img.max() - img.min())
            rinter = rslope * img.min()
            img = img * rslope - rinter

            dcm.RealWorldValueMappingSequence[0].RealWorldValueSlope = rslope
            dcm.RealWorldValueMappingSequence[0].RealWorldValueIntercept = rinter
            # img.dtype = 'uint16'
            # img2 = dcm.pixel_array
            dcm.pixel_array.flat = img.flat
            dcm.PixelData = dcm.pixel_array.tostring()
            dcm.save_as(os.path.join(dir, str(s) + '_D'))

            import pandas as pd
            data_df = pd.DataFrame(self.S0_predicted[:,:,s])
            writer = pd.ExcelWriter(os.path.join(dir, str(s) + '_S0_predicted'+'.xlsx'))
            data_df.to_excel(writer, 'page_1', float_format='%.5f')  # float_format 控制精度
            writer.save()
            data_df = pd.DataFrame(self.perfusion_fraction[:, :, s])
            writer = pd.ExcelWriter(os.path.join(dir, str(s) + '_perfusion_fraction' + '.xlsx'))
            data_df.to_excel(writer, 'page_1', float_format='%.5f')  # float_format 控制精度
            writer.save()
            data_df = pd.DataFrame(self.D_star[:, :, s])
            writer = pd.ExcelWriter(os.path.join(dir, str(s) + '_D_star' + '.xlsx'))
            data_df.to_excel(writer, 'page_1', float_format='%.5f')  # float_format 控制精度
            writer.save()
            data_df = pd.DataFrame(self.D[:, :, s])
            writer = pd.ExcelWriter(os.path.join(dir, str(s) + '_D' + '.xlsx'))
            data_df.to_excel(writer, 'page_1', float_format='%.5f')  # float_format 控制精度
            writer.save()
            self.statusSignalIVIMSave.emit("IVIM save", s + 1, sliceNum)
        time.sleep(1)
        self.statusProgressBarIVIMSave.setVisible(0)
        self.addSourceDir(dir)

    def saveDTIResult(self):
        import time,os
        dir = time.strftime('%Y%m%d%H%M%S',time.localtime(time.time()))
        os.mkdir(dir)
        series = self.getSelectedSeries()
        dcm = dicomio.read_file(series.filenames[0])
        rslope = float(dcm.get('RescaleSlope', 1))
        rinter = float(dcm.get('RescaleIntercept', 0))
        #GRF = np.zeros((192,256,1),dtype='uint16')
        sliceNum = self.DTI_FA.shape[2]
        self.statusSignalDTISave.emit("", 0, 0)
        self.statusProgressBarDTISave.setVisible(1)
        RangeSize = 4096.
        dcm.SeriesDescription = 'DTIParaMap'
        for s in range(sliceNum):
            img = self.DTI_FA[:,:,s]
            rslope = RangeSize/float(img.max()-img.min())
            rinter = rslope*img.min()
            img = img*rslope+rinter

            dcm.RealWorldValueMappingSequence[0].RealWorldValueSlope = rslope
            dcm.RealWorldValueMappingSequence[0].RealWorldValueIntercept= rinter

            #img.dtype = 'uint16'
            #img2 = dcm.pixel_array
            dcm.pixel_array.flat = img.flat
            dcm.PixelData = dcm.pixel_array.tostring()
            dcm.save_as(os.path.join(dir,str(s)+'_DTI_FA'))

            img = self.DTI_MD[:, :, s]
            rslope = RangeSize / float(img.max() - img.min())
            rinter = rslope * img.min()
            img = img * rslope - rinter
            dcm.RealWorldValueMappingSequence[0].RealWorldValueSlope = rslope
            dcm.RealWorldValueMappingSequence[0].RealWorldValueIntercept = rinter
            # img.dtype = 'uint16'
            # img2 = dcm.pixel_array
            dcm.pixel_array.flat = img.flat
            dcm.PixelData = dcm.pixel_array.tostring()
            dcm.save_as(os.path.join(dir, str(s) + '_DTI_MD'))

            img = self.DTI_AD[:, :, s]
            rslope = RangeSize / float(img.max() - img.min())
            rinter = rslope * img.min()
            img = img * rslope- rinter
            dcm.RealWorldValueMappingSequence[0].RealWorldValueSlope = rslope
            dcm.RealWorldValueMappingSequence[0].RealWorldValueIntercept = rinter
            # img.dtype = 'uint16'
            # img2 = dcm.pixel_array
            dcm.pixel_array.flat = img.flat
            dcm.PixelData = dcm.pixel_array.tostring()
            dcm.save_as(os.path.join(dir, str(s) + '_DTI_AD'))

            img = self.DTI_RD[:, :, s]
            rslope = RangeSize / float(img.max() - img.min())
            rinter = rslope * img.min()
            img = img * rslope - rinter
            dcm.RealWorldValueMappingSequence[0].RealWorldValueSlope = rslope
            dcm.RealWorldValueMappingSequence[0].RealWorldValueIntercept = rinter
            # img.dtype = 'uint16'
            # img2 = dcm.pixel_array
            dcm.pixel_array.flat = img.flat
            dcm.PixelData = dcm.pixel_array.tostring()
            dcm.save_as(os.path.join(dir, str(s) + '_DTI_RD'))




            import pandas as pd
            data_df = pd.DataFrame(self.DTI_FA[:,:,s])
            writer = pd.ExcelWriter(os.path.join(dir, str(s) + '_DTI_FA'+'.xlsx'))
            data_df.to_excel(writer, 'page_1', float_format='%.5f')  # float_format 控制精度
            writer.save()
            data_df = pd.DataFrame(self.DTI_MD[:, :, s])
            writer = pd.ExcelWriter(os.path.join(dir, str(s) + '_DTI_MD' + '.xlsx'))
            data_df.to_excel(writer, 'page_1', float_format='%.5f')  # float_format 控制精度
            writer.save()
            data_df = pd.DataFrame(self.DTI_RD[:, :, s])
            writer = pd.ExcelWriter(os.path.join(dir, str(s) + '_DTI_RD' + '.xlsx'))
            data_df.to_excel(writer, 'page_1', float_format='%.5f')  # float_format 控制精度
            writer.save()
            data_df = pd.DataFrame(self.DTI_AD[:, :, s])
            writer = pd.ExcelWriter(os.path.join(dir, str(s) + '_DTI_AD' + '.xlsx'))
            data_df.to_excel(writer, 'page_1', float_format='%.5f')  # float_format 控制精度
            writer.save()
            self.statusSignalDTISave.emit("DTI save",s+1,sliceNum)
        time.sleep(1)
        self.statusProgressBarDTISave.setVisible(0)



        self.addSourceDir(dir)

    def saveADCMonoResult(self):
        import time, os
        dir = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        os.mkdir(dir)
        series = self.getSelectedSeries()
        dcm = dicomio.read_file(series.filenames[0])
        rslope = float(dcm.get('RescaleSlope', 1))
        rinter = float(dcm.get('RescaleIntercept', 0))
        # GRF = np.zeros((192,256,1),dtype='uint16')
        sliceNum = self.ADCm.shape[2]
        self.statusSignalMonoESave.emit("", 0, 0)
        self.statusProgressBarMonoESave.setVisible(1)
        RangeSize = 4096.
        dcm.SeriesDescription = 'MonoEParaMap'
        for s in range(sliceNum):
            img = self.ADCm[:, :, s]
            rslope = RangeSize / float(img.max() - img.min())
            rinter = rslope * img.min()
            img = img * rslope - rinter

            dcm.RealWorldValueMappingSequence[0].RealWorldValueSlope = rslope
            dcm.RealWorldValueMappingSequence[0].RealWorldValueIntercept = rinter

            # img.dtype = 'uint16'
            # img2 = dcm.pixel_array
            dcm.pixel_array.flat = img.flat
            dcm.PixelData = dcm.pixel_array.tostring()
            dcm.save_as(os.path.join(dir, str(s) + '_MonoE_ADC'))


            import pandas as pd
            data_df = pd.DataFrame(self.ADCm[:, :, s])
            writer = pd.ExcelWriter(os.path.join(dir, str(s) + '_MonoE_ADC' + '.xlsx'))
            data_df.to_excel(writer, 'page_1', float_format='%.5f')  # float_format 控制精度
            writer.save()

            self.statusSignalMonoESave.emit("MonoE save", s + 1, sliceNum)
        time.sleep(1)
        self.statusProgressBarMonoESave.setVisible(0)

        self.addSourceDir(dir)

    def saveSEResult(self):
        import time, os
        dir = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        os.mkdir(dir)
        series = self.getSelectedSeries()
        dcm = dicomio.read_file(series.filenames[0])
        rslope = float(dcm.get('RescaleSlope', 1))
        rinter = float(dcm.get('RescaleIntercept', 0))
        # GRF = np.zeros((192,256,1),dtype='uint16')
        sliceNum = self.ADCs.shape[2]
        self.statusSignalSESave.emit("", 0, 0)
        self.statusProgressBarSESave.setVisible(1)
        RangeSize = 4096.
        dcm.SeriesDescription = 'SEParaMap'
        for s in range(sliceNum):
            img = self.ADCs[:, :, s]
            rslope = RangeSize / float(img.max() - img.min())
            rinter = rslope * img.min()
            img = img * rslope - rinter

            dcm.RealWorldValueMappingSequence[0].RealWorldValueSlope = rslope
            dcm.RealWorldValueMappingSequence[0].RealWorldValueIntercept = rinter

            # img.dtype = 'uint16'
            # img2 = dcm.pixel_array
            dcm.pixel_array.flat = img.flat
            dcm.PixelData = dcm.pixel_array.tostring()
            dcm.save_as(os.path.join(dir, str(s) + '_SE_ADC'))

            img = self.Alpha[:, :, s]
            rslope = RangeSize / float(img.max() - img.min())
            rinter = rslope * img.min()
            img = img * rslope - rinter

            dcm.RealWorldValueMappingSequence[0].RealWorldValueSlope = rslope
            dcm.RealWorldValueMappingSequence[0].RealWorldValueIntercept = rinter

            # img.dtype = 'uint16'
            # img2 = dcm.pixel_array
            dcm.pixel_array.flat = img.flat
            dcm.PixelData = dcm.pixel_array.tostring()
            dcm.save_as(os.path.join(dir, str(s) + '_SE_Alpha'))


            import pandas as pd
            data_df = pd.DataFrame(self.ADCs[:, :, s])
            writer = pd.ExcelWriter(os.path.join(dir, str(s) + '_SE_ADC' + '.xlsx'))
            data_df.to_excel(writer, 'page_1', float_format='%.5f')  # float_format 控制精度
            writer.save()

            data_df = pd.DataFrame(self.Alpha[:, :, s])
            writer = pd.ExcelWriter(os.path.join(dir, str(s) + '_SE_Alpha' + '.xlsx'))
            data_df.to_excel(writer, 'page_1', float_format='%.5f')  # float_format 控制精度
            writer.save()

            self.statusSignalSESave.emit("SE save", s + 1, sliceNum)
        time.sleep(1)
        self.statusProgressBarSESave.setVisible(0)

        self.addSourceDir(dir)














#import matplotlib
#matplotlib.use("Qt5Agg")  # 声明使用QT5
#from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
#from matplotlib.figure import Figure
#class Figure_Canvas(FigureCanvas):   # 通过继承FigureCanvas类，使得该类既是一个PyQt5的Qwidget，又是一个matplotlib的FigureCanvas，这是连接pyqt5与matplot                                          lib的关键
#    def __init__(self, parent=None, width=5, height=3, dpi=300):
#
#        fig = Figure(figsize=(width, height), dpi=50)  # 创建一个Figure，注意：该Figure为matplotlib下的figure，不是matplotlib.pyplot下面的figure
#
#        FigureCanvas.__init__(self, fig) # 初始化父类
#        self.setParent(parent)
#
#        self.axes = fig.add_subplot(111) # 调用figure下面的add_subplot方法，类似于matplotlib.pyplot下面的subplot方法
#    def line(self,x,y):
#        self.axes.plot(x, y)

def main(args=[],app=None):
    '''
    Default main program which starts Qt based on the command line arguments `args', sets the stylesheet if present, then
    creates the window object and shows it. The `args' list of command line arguments is also passed to the window object
    to pick up on specified directories. The `app' object would be the QApplication object if this was created elsewhere,
    otherwise it's created here. Returns the value of QApplication.exec_() if this object was created here otherwise 0.
    '''

    if not app:
        app = QtGui.QApplication(args)
        app.setAttribute(Qt.AA_DontUseNativeMenuBar) # in OSX, forces menubar to be in window
        app.setStyle('Plastique')
        
        # load the stylesheet included as a Qt resource
        with closing(QtCore.QFile(':/css/DefaultUIStyle.css')) as f:
            if f.open(QtCore.QFile.ReadOnly):
                app.setStyleSheet(bytes(f.readAll()).decode('UTF-8'))
            else:
                print('Failed to read %r'%f.fileName())

    browser=DicomBrowser(args)
    browser.show()
    #input()
    #while True:
    #    app.processEvents()
    #return app.processEvents() if app else 0
    return app.exec_() if app else 0


def mainargv():
    '''setuptools compatible entry point.'''
    #The Scripts below make pyinstaller --onefile mode work
    try:
        # Python 3.4+
        if sys.platform.startswith('win'):
            import multiprocessing.popen_spawn_win32 as forking
        else:
            import multiprocessing.popen_fork as forking
    except ImportError:
        import multiprocessing.forking as forking

    if sys.platform.startswith('win'):
        # First define a modified version of Popen.
        class _Popen(forking.Popen):
            def __init__(self, *args, **kw):
                if hasattr(sys, 'frozen'):
                    # We have to set original _MEIPASS2 value from sys._MEIPASS
                    # to get --onefile mode working.
                    os.putenv('_MEIPASS2', sys._MEIPASS)
                try:
                    super(_Popen, self).__init__(*args, **kw)
                finally:
                    if hasattr(sys, 'frozen'):
                        # On some platforms (e.g. AIX) 'os.unsetenv()' is not
                        # available. In those cases we cannot delete the variable
                        # but only set it to the empty string. The bootloader
                        # can handle this case.
                        if hasattr(os, 'unsetenv'):
                            os.unsetenv('_MEIPASS2')
                        else:
                            os.putenv('_MEIPASS2', '')

                            # Second override 'Popen' class with our modified version.

        forking.Popen = _Popen
    freeze_support()
    sys.exit(main(sys.argv))


if __name__=='__main__':
    import multiprocessing
    #multiprocessing.freeze_support()
    mainargv()
    