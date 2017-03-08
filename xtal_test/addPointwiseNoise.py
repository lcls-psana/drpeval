import psana
import h5py
import matplotlib.pyplot as plt
from shutil import copyfile
import time
import numpy as np

darkStr = '84'
for runNum in [93,94,95,96,97,98,99,100,101]:
    try:
        print "Processing: ", runNum
        runStr = str(runNum).zfill(4)
        src = '/reg/d/psdm/cxi/cxic0415/res/cheetah/hdf5/r'+runStr+'-se_'+darkStr+'_new/cxic0415-r'+runStr+'.cxi'
        dst = src+'-noisy'
        f=h5py.File(src,'r')
        sec = f['/LCLS/machineTime'].value
        nsec = f['/LCLS/machineTimeNanoSeconds'].value
        fiducial = f['/LCLS/fiducial'].value
        numImages = len(sec)
        f.close()

        ds = psana.DataSource('exp=cxic0415:run='+str(runNum)+':idx')
        run = ds.runs().next()
        det = psana.Detector('DscCsPad')

        et = psana.EventTime(int((sec[0]<<32)|nsec[0]),fiducial[0])
        evt = run.event(et)
        calib = det.calib(evt)


        raw = det.raw(evt)
        noise = np.random.uniform(low=-0.005, high=0.005, size=raw.shape)+1
        noisyRaw = np.round(noise * raw)

        error = raw-noisyRaw
#plt.hist(error.flatten(),2000)
#plt.title('Point-wise error (ADU)')
#plt.show()

        pedestal = det.pedestals(evt)
        pedestalCorrected = noisyRaw - pedestal
        commonMode = det.common_mode_correction(evt, pedestalCorrected)
        commonModeCorrected = pedestalCorrected + commonMode
        gain = det.gain(evt)
        calibManual = commonModeCorrected * gain

        dim0 = 1480
        dim1 = 1552
        img = np.zeros((dim0, dim1))
        #img1 = np.zeros((dim0, dim1))
        counter = 0
        for quad in range(4):
            for seg in range(8):
                img[seg * 185:(seg + 1) * 185, quad * 388:(quad + 1) * 388] = calibManual[counter, :, :]
                #img1[seg * 185:(seg + 1) * 185, quad * 388:(quad + 1) * 388] = calib[counter, :, :]
                counter += 1

#plt.subplot(131)
#plt.imshow(img,vmax=1000,vmin=0)
#plt.title('noisy image')
#plt.colorbar()
#plt.subplot(132)
#plt.imshow(img1,vmax=1000,vmin=0)
#plt.title('image')
#plt.colorbar()
#plt.subplot(133)
#plt.imshow(img1-img,vmax=100,vmin=-100)
#plt.title('diff')
#plt.colorbar()
#plt.show()

        tic = time.time()
        copyfile(src, dst)
        f=h5py.File(dst,'a')
        data = f['/entry_1/instrument_1/detector_1/data']
        for i in range(numImages):
            et = psana.EventTime(int((sec[i]<<32)|nsec[i]),fiducial[i])
            evt = run.event(et)
            raw = det.raw(evt)
            noise = np.random.uniform(low=-0.005, high=0.005, size=raw.shape)+1
            noisyRaw = np.round(noise * raw)
            pedestalCorrected = noisyRaw - pedestal
            commonMode = det.common_mode_correction(evt, pedestalCorrected)
            commonModeCorrected = pedestalCorrected + commonMode
            calibManual = np.int16(commonModeCorrected * gain)
            counter = 0
            img = np.zeros((dim0, dim1))
            for quad in range(4):
                for seg in range(8):
                    img[seg * 185:(seg + 1) * 185, quad * 388:(quad + 1) * 388] = calibManual[counter, :, :]
                    counter += 1
            data[i,:,:] = img
        toc = time.time()
        print "time: ", runNum, toc-tic, (toc-tic)/numImages
        f.close()
    except:
        print "Can't process run: ", runNum
        continue

#from IPython import embed
#embed()

#plt.imshow(img,vmax=1000)
#plt.show()

