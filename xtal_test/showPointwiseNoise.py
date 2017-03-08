import psana
import h5py
import matplotlib.pyplot as plt
from shutil import copyfile
import time
import numpy as np

darkStr = '44'
for runNum in [51]:
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
        plt.hist(error.flatten(),2000)
        plt.title('Point-wise error (ADU)')
        plt.savefig("HistogramPointwiseError.png")
        plt.show()

        pedestal = det.pedestals(evt)
        pedestalCorrected = noisyRaw - pedestal
        commonMode = det.common_mode_correction(evt, pedestalCorrected)
        commonModeCorrected = pedestalCorrected + commonMode
        gain = det.gain(evt)
        calibManual = commonModeCorrected * gain
        
        assembledImg = det.image(evt)
        assembledNoisyImg = det.image(evt,nda_in=calibManual)
        plt.subplot(222)
        plt.imshow(assembledNoisyImg,vmax=4000,vmin=0)
        plt.title('noisy image')
        #plt.colorbar()
        plt.subplot(221)
        plt.imshow(assembledImg,vmax=4000,vmin=0)
        plt.title('original image')
        #plt.colorbar()
        plt.subplot(223)
        commonModeAssem = det.image(evt, nda_in=commonMode)
        plt.imshow(commonModeAssem)#,vmax=4000,vmin=0)
        plt.title('common mode image')
        plt.colorbar()
        plt.subplot(224)
        diff = assembledImg-assembledNoisyImg
        plt.imshow(diff,vmax=50,vmin=-50)
        #plt.title('diff: '+str(np.max(diff)+','+str(np.min(diff))))
        plt.colorbar()
        plt.savefig("ArtificialNoiseAssem.png")
        plt.show()

        dim0 = 1480
        dim1 = 1552
        img = np.zeros((dim0, dim1))
        img1 = np.zeros((dim0, dim1))
        counter = 0
        for quad in range(4):
            for seg in range(8):
                img[seg * 185:(seg + 1) * 185, quad * 388:(quad + 1) * 388] = calibManual[counter, :, :]
                img1[seg * 185:(seg + 1) * 185, quad * 388:(quad + 1) * 388] = calib[counter, :, :]
                counter += 1

        plt.subplot(132)
        plt.imshow(img,vmax=4000,vmin=0)
        plt.title('noisy image')
        #plt.colorbar()
        plt.subplot(131)
        plt.imshow(img1,vmax=4000,vmin=0)
        plt.title('original image')
        #plt.colorbar()
        plt.subplot(133)
        diff = img1-img
        plt.imshow(diff,vmax=50,vmin=-50)
        #plt.title('diff: '+str(np.max(diff)+','+str(np.min(diff))))
        plt.colorbar()
        plt.savefig("ArtificialNoise.png")
        plt.show()

    except:
        print "Can't process run: ", runNum
        continue

#from IPython import embed
#embed()

#plt.imshow(img,vmax=1000)
#plt.show()

