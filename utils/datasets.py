import numpy as np
import h5py
import matplotlib.pyplot as plt
import numpy as np
from skimage.segmentation import mark_boundaries

# generate a batch of images, returns images and targets
def generate_fake_samples(g_model, samples, patch_shape):
	# generate fake instance
	X = g_model.predict(samples)
	# create 'fake' class labels (0)
	y = np.zeros((len(X), patch_shape, patch_shape, 1))
	return X, y

# select a batch of random samples, returns images and target
def generate_real_samples(dataset, n_samples, patch_shape):
	# unpack dataset
	trainA, trainB = dataset
	# choose random instances
	ix = np.random.randint(0, trainA.shape[0], n_samples)
	# retrieve selected images
	X1, X2 = trainA[ix], trainB[ix]
	# generate 'real' class labels (1)
	y = np.ones((n_samples, patch_shape, patch_shape, 1))
	return [X1, X2], y

def create_dataset():
    fid = r"C:\Users\Ryan\Documents\DREAM3D-6.5.167-Win64\Data\Output\Synthetic\02_HexagonalSingleEquiaxedOut.dream3d"
    f = h5py.File(fid, 'r')
    
    ipf = f['DataContainers/SyntheticVolumeDataContainer/CellData/IPFColor'][:]
    ids = f['DataContainers/SyntheticVolumeDataContainer/CellData/FeatureIds'][:].squeeze()
    
    batchA = []
    batchB = []
    for x in range(ipf.shape[0]):
        ipfSlice = ipf[x,:,:,:]
        idsSlice = ids[x,:,:]
        
        white_rgb = np.ones((128,128,3), dtype='uint8') * 255
        edges = mark_boundaries(image = white_rgb, label_img = idsSlice, color=(0,0,0))
        
        batchA.append(ipfSlice)
        batchB.append(edges)
        

    X = resize(np.array(a), (256,256,256,3), preserve_range=True).astype('uint8')
    Y = resize(np.array(b)*255, (256,256,256), preserve_range=True).astype('uint8')
    np.save(r"C:\Users\Ryan\Documents\GitHub\vec2pix\datasets\ipf_256x256x256x3.npy", X)
    np.save(r"C:\Users\Ryan\Documents\GitHub\vec2pix\datasets\ids_256x256x256.npy", Y)
    print(r"Saved dataset to C:\Users\Ryan\Documents\GitHub\vec2pix\datasets")

def load_dataset():
	x = np.load(r"C:\Users\Ryan\Documents\GitHub\vec2pix\datasets\ipf_256x256x256x3.npy") 
	y = np.load(r"C:\Users\Ryan\Documents\GitHub\vec2pix\datasets\ids_256x256x256.npy")

	#Scale dataset
	x = (x - 127.5) / 127.5
	y = (y - 127.5) / 127.5
	
	return x,y