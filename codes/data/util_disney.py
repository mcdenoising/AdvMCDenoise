import OpenEXR
import Imath
import numpy as np
import time
import data.util_exr as exr_utils
import os

def _crop(img, pos, size):
    ow, oh = img.shape[0], img.shape[1]  
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):        
        # return img.crop((x1, y1, x1 + tw, y1 + th)) #CHANGED
        return img[x1:(x1 + tw), y1:(y1 + th), :]
    return img


def get_distinct_prefix(dir_path):
    names = set()
    for f in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, f)):
            names.add(f.split(".")[0].rsplit("-",1)[0])
    return list(names)

# Divide variance by mean^2 to get relative variance
def CalcRelVar(data, var, calcLog, calcLum=True, calcMean=False):

    if calcLum:
      denom = np.expand_dims(CalcLuminance(data), axis=2)
    elif calcMean:
      denom = np.expand_dims(CalcMean(data), axis=2)
    else:
        denom = data
    var = var / ((denom * denom) + 1.0e-5)
    if calcLog:
        var = LogTransform(var)
    return var

# Calculate log transform (with an offset to map zero to zero)
def LogTransform(data):
    assert(np.sum(data < 0) == 0)
    return np.log(data + 1.0)


# Calculate luminance (3 channels in and 1 channel out)
def CalcLuminance(data):
    return (0.2126*data[:,:,0] + 0.7152*data[:,:,1] + 0.0722*data[:,:,2])


# Calculate mean (3 channels in and 1 channel out)
def CalcMean(data):
    return (0.3333*data[:,:,0] + 0.3333*data[:,:,1] + 0.3333*data[:,:,2])



# for shading

def loadDisneyEXR_feature_shading(path, FEATURE_LIST):
	# time0 = time.time()
	
	prefix = path.split(".")[0]

	# color_path = prefix + "_color.exr"
	variance_path = prefix + "_variance.exr"
	normal_path = prefix + "_normal.exr"
	depth_path = prefix + "_depth.exr"
	texture_path = prefix + "_texture.exr"
	visibility_path = prefix + "_visibility.exr"
	diffuse_path = prefix + "_diffuse.exr"
	specular_path = prefix + "_specular.exr"


	# inFile = exr_utils.open(variance_path)
	# variance = inFile.get_all()["default"]
	if "normal" in FEATURE_LIST:
		try:
			inFile = exr_utils.open(normal_path)
			normal = inFile.get_all()["default"]
			normal = _crop(normal, (1,1), 128)
		except Exception:
			normal = np.zeros((128,128,3))	

	if "depth" in FEATURE_LIST:		
		try:	
			inFile = exr_utils.open(depth_path)
			depth = inFile.get_all()["default"]
			depth = _crop(depth, (1,1), 128)
		except Exception:
			depth = np.zeros((128,128,1))	

	# if "albedo" in FEATURE_LIST:		//always load in albedo
	try:	
		inFile = exr_utils.open(texture_path)
		texture = inFile.get_all()["default"]
		texture = _crop(texture, (1,1), 128)
	except Exception:
		texture = np.zeros((128,128,3))

	if "visibility" in FEATURE_LIST:		
		try:		
			inFile = exr_utils.open(visibility_path)
			visibility = inFile.get_all()["default"]
			visibility = _crop(visibility, (1,1), 128)
		except Exception:
			visibility = np.zeros((128,128,1))

	if "diffuse" in FEATURE_LIST:		
		try:			
			inFile = exr_utils.open(diffuse_path)
			diffuse = inFile.get_all()["default"]
			diffuse = _crop(diffuse, (1,1), 128)
		except Exception:
			diffuse = np.zeros((128,128,3))	

	if "specular" in FEATURE_LIST:		
		try:		
			inFile = exr_utils.open(specular_path)
			specular = inFile.get_all()["default"]
			specular = _crop(specular, (1,1), 128)
		except Exception:
			specular = np.zeros((128,128,3))	

	# variance = CalcRelVar( (1+ color.copy()) , variance, False, False, True )

	if "diffuse" in FEATURE_LIST:
		diffuse[diffuse < 0.0] = 0.0
		diffuse = diffuse / (texture + 0.00316)
		diffuse = LogTransform(diffuse)
		color = diffuse
	if "specular" in FEATURE_LIST:	
		specular[specular < 0.0] = 0.0
		specular = LogTransform(specular)
		color = specular

	feature_tuple = ()
	if "normal" in FEATURE_LIST:
		normal = np.nan_to_num(normal)
		if "specular" in FEATURE_LIST:
			normal = (normal + 1.0)*0.5
			normal = np.maximum(np.minimum(normal,1.0),0.0)
		feature_tuple += (normal,)
	if "depth" in FEATURE_LIST:	
		# Normalize current frame depth to [0,1]
		maxDepth = np.max(depth)
		if maxDepth != 0:
			depth /= maxDepth
		feature_tuple += (depth,)
	if "albedo" in FEATURE_LIST:
			# texture = np.clip(texture,0.0,1.0)
		feature_tuple += (texture, )
	if "visibility" in FEATURE_LIST:
		feature_tuple += (visibility, )		


	if len(feature_tuple) == 0:
		return color, np.zeros(color.shape)
	feautres = np.concatenate(feature_tuple, axis=2)	 #

	return color, feautres



def loadDisneyEXR_multi_ref_shading(path, FEATURE_LIST):
	# time0 = time.time()
	
	prefix = path.split(".")[0]

	color_path = prefix + "_color.exr"
	diffuse_path = prefix + "_diffuse.exr"
	specular_path = prefix + "_specular.exr"
	texture_path = prefix + "_texture.exr"

	if "diffuse" in FEATURE_LIST:	
		try:	
			inFile = exr_utils.open(diffuse_path)
			diffuse = inFile.get_all()["default"]
			diffuse = _crop(diffuse, (1,1), 128)
		except Exception:
			diffuse = np.zeros((128,128,3))	
	if "specular" in FEATURE_LIST:		
		try:		
			inFile = exr_utils.open(specular_path)
			specular = inFile.get_all()["default"]
			specular = _crop(specular, (1,1), 128)
		except Exception:
			specular = np.zeros((128,128,3))

	try:	
		inFile = exr_utils.open(texture_path)
		texture = inFile.get_all()["default"]
		texture = _crop(texture, (1,1), 128)
	except Exception:
		texture = np.zeros((128,128,3))	

	if "diffuse" in FEATURE_LIST:
		diffuse[diffuse < 0.0] = 0.0
		diffuse = diffuse / (texture + 0.00316)
		diffuse = LogTransform(diffuse)
		color = diffuse
	if "specular" in FEATURE_LIST:	
		specular[specular < 0.0] = 0.0
		specular = LogTransform(specular)
		color = specular
	return color

def loadDisneyEXR_ref(path):
	inFile = exr_utils.open(path)
	data = inFile.get_all()["default"]
	data = LogTransform(data)
	return data






# def loadDisneyEXR_feature_from_whole(path, channel=3):
# 	image = OpenEXR.InputFile(path)
# 	dataWindow = image.header()['dataWindow']
# 	size = (dataWindow.max.x - dataWindow.min.x + 1, dataWindow.max.y - dataWindow.min.y + 1)
# 	FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)

# 	channel_to_extract = ["B","G","R",'colorVariance.Z','normal.B',"normal.G","normal.R",'depth.Z','albedo.B',"albedo.G","albedo.R",'visibility.Z']
# 	time0 = time.time()
# 	data =  np.array([np.fromstring(image.channel(c, FLOAT), dtype=np.float32) for c in channel_to_extract])
# 	data = np.moveaxis(data, 0, -1)
# 	data = data.reshape(size[1], size[0], -1)

# 	time1 = time.time()
# 	color = data[:,:,:3]
# 	variance = data[:,:,3:4]
# 	normal = data[:,:,4:7]
# 	depth = data[:,:, 7:8]
# 	texture = data[:,:, 8:11]
# 	visibility = data[:,:, 11:12]

# 	time2 = time.time()
# 	variance = CalcRelVar( (1+ color.copy()) , variance, False, False, True )
# 	color = LogTransform(color)
# 	normal = (normal + 1.0)*0.5
# 	# Normalize current frame depth to [0,1]
# 	maxDepth = np.max(depth)
# 	if maxDepth != 0:
# 		depth /= maxDepth

# 	features = np.concatenate((variance,normal,depth,texture,visibility), axis=2)
# 	time3 = time.time()
# 	print("time 0 =%f, time1 = %f, time2 = %f " %(time1-time0, time2-time1,time3-time2))
# 	return color, features



# def loadDisneyEXR_feature(path, FEATURE_LIST):
# 	# time0 = time.time()
	
# 	prefix = path.split(".")[0]

# 	color_path = prefix + "_color.exr"
# 	variance_path = prefix + "_variance.exr"
# 	normal_path = prefix + "_normal.exr"
# 	depth_path = prefix + "_depth.exr"
# 	texture_path = prefix + "_texture.exr"
# 	# visibility_path = prefix + "_visibility.exr"

# 	diffuse_path = prefix + "_diffuse.exr"
# 	specular_path = prefix + "_specular.exr"
# 	try:
# 		inFile = exr_utils.open(color_path)
# 		color = inFile.get_all()["default"]
# 		color = _crop(color, (1,1), 128)
# 	except Exception:
# 		color = np.zeros((128,128,3))	
# 	# inFile = exr_utils.open(variance_path)
# 	# variance = inFile.get_all()["default"]
# 	try:
# 		inFile = exr_utils.open(normal_path)
# 		normal = inFile.get_all()["default"]
# 		normal = _crop(normal, (1,1), 128)
# 	except Exception:
# 		normal = np.zeros((128,128,3))	
# 	try:	
# 		inFile = exr_utils.open(depth_path)
# 		depth = inFile.get_all()["default"]
# 		depth = _crop(depth, (1,1), 128)
# 	except Exception:
# 		depth = np.zeros((128,128,1))	
# 	try:	
# 		inFile = exr_utils.open(texture_path)
# 		texture = inFile.get_all()["default"]
# 		texture = _crop(texture, (1,1), 128)
# 	except Exception:
# 		texture = np.zeros((128,128,3))	
# 	# try:		
# 	# 	inFile = exr_utils.open(visibility_path)
# 	# 	visibility = inFile.get_all()["default"]
# 	# 	visibility = _crop(visibility
# 	# 		, (1,1), 128)
# 	# except Exception:
# 	# 	visibility = np.zeros((128,128,1))
# 	try:			
# 		inFile = exr_utils.open(diffuse_path)
# 		diffuse = inFile.get_all()["default"]
# 		diffuse = _crop(diffuse, (1,1), 128)
# 	except Exception:
# 		diffuse = np.zeros((128,128,3))	
# 	try:		
# 		inFile = exr_utils.open(specular_path)
# 		specular = inFile.get_all()["default"]
# 		specular = _crop(specular, (1,1), 128)
# 	except Exception:
# 		specular = np.zeros((128,128,3))	

# 	# variance = CalcRelVar( (1+ color.copy()) , variance, False, False, True )
# 	color[color < 0.0] = 0.0
# 	color = LogTransform(color)
# 	diffuse[diffuse < 0.0] = 0.0
# 	diffuse = LogTransform(diffuse)
# 	specular[specular < 0.0] = 0.0
# 	specular = LogTransform(specular)
# 	normal = np.nan_to_num(normal)
# 	normal = (normal + 1.0)*0.5
# 	normal = np.maximum(np.minimum(normal,1.0),0.0)
# 	# Normalize current frame depth to [0,1]
# 	maxDepth = np.max(depth)
# 	if maxDepth != 0:
# 		depth /= maxDepth

# 	# texture = np.clip(texture,0.0,1.0)
	
# 	# feautres = np.concatenate((variance,  normal, depth, texture, visibility), axis=2)	
# 	feautres = np.concatenate((normal, depth, texture), axis=2)	 #visibility

# 	return color, diffuse, specular, feautres
# 	# return np.concatenate((color, normal, depth, texture), axis=2)



# def loadDisneyEXR_multi_ref(path, FEATURE_LIST):
# 	# time0 = time.time()
	
# 	prefix = path.split(".")[0]

# 	color_path = prefix + "_color.exr"
# 	diffuse_path = prefix + "_diffuse.exr"
# 	specular_path = prefix + "_specular.exr"
# 	try:
# 		inFile = exr_utils.open(color_path)
# 		color = inFile.get_all()["default"]
# 		color = _crop(color, (1,1), 128)
# 	except Exception:
# 		color = np.zeros((128,128,3))	
# 	try:	
# 		inFile = exr_utils.open(diffuse_path)
# 		diffuse = inFile.get_all()["default"]
# 		diffuse = _crop(diffuse, (1,1), 128)
# 	except Exception:
# 		diffuse = np.zeros((128,128,3))	
# 	try:		
# 		inFile = exr_utils.open(specular_path)
# 		specular = inFile.get_all()["default"]
# 		specular = _crop(specular, (1,1), 128)
# 	except Exception:
# 		specular = np.zeros((128,128,3))		

# 	color[color<0.0] = 0.0
# 	color = LogTransform(color)
# 	diffuse[diffuse < 0.0] = 0.0
# 	diffuse = LogTransform(diffuse)
# 	specular[specular < 0.0] = 0.0
# 	specular = LogTransform(specular)
# 	return color, diffuse, specular
