import os
import numpy as np
from mpstool.img import *
from copy import deepcopy
from tqdm import tqdm
SAMPLE_RATE=10000

def sample_from_png(filename, param):
    assert(".png" in filename)
    nb_channels = param["DATA_DIM"][2]
    channel_mode = "RGB" if nb_channels in [3,4] else "Gray"
    source = Image.fromPng(filename, channel_mode=channel_mode)
    if param["VERBOSE"]:
        print("Source image shape : ", source.shape)

    if nb_channels==1:
        output = [source.get_sample(param["DATA_DIM"], normalize=True) for i in range(param["EPOCH_SIZE"])]
    else:
        output = [source.get_sample(param["DATA_DIM"], normalize=True, var_name=["R","G","B"]) for i in range(param["EPOCH_SIZE"])]

    if param["VERBOSE"]:
        for i in range(int(len(output)//SAMPLE_RATE)):
            to_output = deepcopy(output[i])
            to_output.exportAsPng("output/test_example{}.png".format(i))

    output = np.array([img.asArray() for img in output])
    return output


def sample_from_gslib(filename, param):
    assert(".gslib" in filename)
    source = Image.fromGslib(filename)
    if param["VERBOSE"]:
        print("source image shape : ", source.shape)
    output = [source.get_sample(param["DATA_DIM"], normalize=True) for i in range(param["EPOCH_SIZE"])]
    if param["VERBOSE"]:
        for i in range(int(len(output)//SAMPLE_RATE)):
            to_output = deepcopy(output[i])
            to_output.exportAsVox("output/test_example{}.vox".format(i))
    output = np.array([img.asArray() for img in output])
    return output

def single_sample_from_path(path, param):
    if path.endswith('.png'):
        return sample_from_png(path, param)  # Assumes sample_from_png is a predefined function
    elif path.endswith('.gslib'):
        return sample_from_gslib(path, param)  # Assumes sample_from_gslib is a predefined function
    else:
        raise ValueError(f"Unsupported file format: {path}")

def load_examples_from_folder(folder_path, extension, loader_function):
    if not os.path.exists(folder_path):
        raise FileNotFoundError(f"The specified folder does not exist: {folder_path}")

    file_paths = [f for f in os.listdir(folder_path) if f.endswith(extension)]

    if not file_paths:
        raise ValueError(f"No {extension} files found in {folder_path}")

    examples = []
    for file_name in file_paths:
        print(f"Reading file {file_name} ...")
        file_path = os.path.join(folder_path, file_name)
        image = loader_function(file_path)
        examples.append(image)
    return np.array(examples)

def get_train_examples(param):

    path = os.path.join(os.getcwd(),param["SOURCE"])
    
    try:
        if path.endswith(('.png', '.gslib')):
                return single_sample_from_path(path,param)

        print("Using examples from the '{}' folder".format(param["SOURCE"]))
        path_png = os.path.join(path,"png")
        path_gslib = os.path.join(path,"gslib")

        # Loading from png subfolder
        if os.path.exists(path_png):
                return load_examples_from_folder(path_png, '.png', 
                                                lambda p: Image.fromPng(p, 
                                                                        normalize=True, 
                                                                        channel_mode="Gray").asArray())

        # Loading from gslib subfolder
        if os.path.exists(path_gslib):
                return load_examples_from_folder(path_gslib, '.gslib', 
                                                lambda p: Image.fromGslib(p, 
                                                                        normalize=True).asArray())

        raise ValueError(f"No supported files found in the source folder: {path}")

    except Exception as e:
        print(f"An error occurred: {e}")
        raise

def sample_cuts_from_gslib(filename, param, nb_cuts):
    # Assumes the gslib represents some 3D data
    assert(".gslib" in filename)
    source = Image.fromGslib(filename)
    if param["VERBOSE"]:
        print("source image shape : ", source.shape)
    output = []
    size =  param["DATA_DIM"][1]
    xs,ys,zs = source.shape
    source = source.asArray()
    for i in range(param["EPOCH_SIZE"]):
        sample = []
        x = np.random.randint(xs-size)
        y = np.random.randint(ys-size)
        z = np.random.randint(zs-size)
        sample.append(source[x, y:y+size, z:z+size])
        sample.append(source[x:x+size, y, z:z+size])
        if nb_cuts==3:
             sample.append(source[x:x+size, y:y+size, z])
        sample = Image.fromArray(np.array(sample))
        sample.normalize()
        output.append(sample.asArray())
    output = np.array(output)
    if param["VERBOSE"]:
        print("Example tensor shape :", output.shape)
        for i in range(int(len(output)//SAMPLE_RATE)):
            img = deepcopy(output[i*SAMPLE_RATE,:,:,:])
            img = np.concatenate(img, axis=0)
            img = img.reshape(img.shape+(1,))
            Image.fromArray(img).exportAsPng("output/test_example{}.png".format(i))
    return output


def get_train_examples_cuts(param, nb_cuts):
    path = param["SOURCE"]
    assert nb_cuts==2 or nb_cuts==3
    if '.gslib' in path:
        return sample_cuts_from_gslib(path, param, nb_cuts)

    # Sampling from png training image
    source_x = Image.fromPng(os.path.join(path,"x.png"))
    source_y = Image.fromPng(os.path.join(path,"y.png"))
    source_z = Image.fromPng(os.path.join(path,"z.png"))
    output=[]
    print('source_x',os.path.join(path,"x.png"))
    print('source_y',os.path.join(path,"y.png"),)
    print('source_z',os.path.join(path,"z.png"))
    extract_dim = param["DATA_DIM"][1:]+(1,)
    for i in tqdm(range(param["EPOCH_SIZE"])):

        sample_x = source_x.get_sample(extract_dim, normalize=True).asArray()
        sample_x = sample_x.reshape(sample_x.shape[:-1])
        
        sample_y = source_y.get_sample(extract_dim,  normalize=True).asArray()
        sample_y = sample_y.reshape(sample_y.shape[:-1])
        sample = [sample_x, sample_y]
        if nb_cuts==3:
            sample_z = source_z.get_sample(extract_dim, normalize=True).asArray()
            sample_z = sample_z.reshape(sample_z.shape[:-1])
            sample.append(sample_z)
        sample = np.array(sample)
        output.append(sample)
    output = np.array(output)
    if param["VERBOSE"]:
        print("Example tensor shape :", output.shape)
        for i in range(int(len(output)//SAMPLE_RATE)):
            img = deepcopy(output[i*SAMPLE_RATE,:,:])
            img = np.concatenate(img, axis=0)
            img = img.reshape(img.shape+(1,))
            Image.fromArray(img).exportAsPng("output/test_example{}.png".format(i))
    return output
