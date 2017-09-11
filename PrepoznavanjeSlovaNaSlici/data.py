import os
import cPickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from glob import glob
import string
from matplotlib import pyplot as plt
import sys
from random import seed, sample
from pprint import pprint
from datetime import datetime
from sklearn.base import BaseEstimator
from skimage.feature import hog
from skimage import color
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import confusion_matrix
from nolearn.decaf import ConvNetFeatures
from sklearn.metrics import accuracy_score

class OcrData():
    """
    Ova klasa omogucuje kreiranje i odbradjivanje objekata u slici
    """
    
    def __init__(self, config):

        #konstruktor za preuzimanje parametara iz ocr-config.py fajlova
        self.config = self._load_config(config)
        self.folder_labels = self.config['folder_labels']
        self.folder_data = self.config['folder_data']
        self.verbose = self.config['verbose']
        self.img_size = self.config['img_size']
        self.limit = self.config['limit']
        self.pickle_data = self.config['pickle_data']
        self.from_pickle = self.config['from_pickle']
        self.automatic_split = self.config['automatic_split']
        self.plot_evaluation = self.config['plot_evaluation']
        self.split = self.config['percentage_of_test_set']
        self.cross_val_models = self.set_models()
        self.load()

        # ucitava sliku i deli je na test i train
        if self.automatic_split:
            self.split_train_test()


    def _load_config(self, filename):

        # dobijamo parametre iz ocr-config fajla
        return eval(open(filename).read())


    def load(self):

        if self.from_pickle:
            try:
                # ako je from_pickle == True u ocr-config.py ucitane podatke ponovo vracamo u isti folder
                with open(os.path.join(self.folder_data, self.pickle_data), 'rb') as fin:
                    self.ocr = cPickle.load(fin)
                    if self.limit == 0:
                        pass
                    else:
                        self.ocr = {
                            'images': self.ocr['images'][:self.limit],
                            'data': self.ocr['data'][:self.limit],
                            'target': self.ocr['target'][:self.limit]
                        }
                    if self.verbose:
                        print 'Loaded {} images each {} pixels'.format(self.ocr['images'].shape[0], self.img_size)
                    return self.ocr

            except IOError:
                print 'You have not provided a .pickle file to load data from!'
                sys.exit(0)
        else:
            #ako je from_pickle == False u ocr-config.py pruzimamo relativnue putanje slika i njihove labele
            image_paths = self.getRelativePath()
            image_labels = self.getLabels()

            #zipujemo putanju slike i labelu
            if self.limit == 0:
                complete = zip(image_paths, image_labels)
            else:
                complete = zip(image_paths[:self.limit], image_labels[:self.limit])
            n_images = len(complete)
            im = np.zeros((n_images,) + self.img_size)
            labels = []
            i = 0

            #ucitavmo sliku u grayscale i promeni velicinu
            for couple in complete:
                image = imread(os.path.join(self.folder_data, couple[0] + '.png'), as_grey=True)
                sh = image.shape
                if ((sh[0] * sh[1]) >= (self.img_size[0] * self.img_size[1])):
                    im[i] = resize(image, self.img_size)
                    i += 1
                    labels.append(couple[1])
            im = im[:len(labels)]

            #preuzima podatke
            seed(10)
            k = sample(range(len(im)), len(im))
            im_shuf = im[k]
            labels_shuf = np.array(labels)[k]

            if self.verbose:
                print 'Loaded {} images each {} pixels'.format(len(labels), self.img_size)

            #postavlja novodobijene bodatke u fajl
            self.ocr = {
                'images': im_shuf,
                'data': im_shuf.reshape((im_shuf.shape[0], -1)),  # / 255.0
                'target': labels_shuf
            }

            now = str(datetime.now()).replace(':', '-')
            fname_out = 'images-{}-{}-{}.pickle'.format(len(labels), self.img_size, now)
            full_name = os.path.join(self.folder_data, fname_out)
            with open(full_name, 'wb') as fout:
                cPickle.dump(self.ocr, fout, -1)

            return self.ocr


    def getRelativePath(self):

        # uzima relativnu putanju slika i vraca ih u listu
        mfiles = [os.path.join(self.folder_labels,mfile) for mfile in glob(os.path.join(self.folder_labels,'*.m'))]

        self.images = []

        #ucitavaju se slike Englishfnt, Englishhnd i Englishimg dataseta
        for mfile in mfiles:
            m = open(mfile, "r")
            lines = m.readlines()
            for index, line in enumerate(lines):
                if line.startswith('list.ALLnames'):
                    start_index = index
                    start_image = line[18:].strip()[:-1 ]
                    if 'Img' in mfile:
                        self.images.append(os.path.join(*(['Englishimg','Img'] + start_image.split('/'))))
                    elif 'Fnt' in mfile:
                        self.images.append(os.path.join(*(['EnglishFnt','Fnt'] + start_image.split('/'))))
                    elif 'Hnd' in mfile:
                        self.images.append(os.path.join(*(['EnglishHnd','Hnd'] + start_image.split('/'))))
                elif line.startswith('list.classlabels'):
                    end_index = index - 1
            if 'Img' in mfile:
                self.images += [os.path.join(*(['Englishimg','Img'] + line.strip()[1:-1].split('/'))) for line in lines[start_index+1:end_index]]
            elif 'Fnt' in mfile:
                self.images += [os.path.join(*(['EnglishFnt','Fnt'] + line.strip()[1:-1].split('/'))) for line in lines[start_index+1:end_index]]
            elif 'Hnd' in mfile:
                self.images += [os.path.join(*(['EnglishHnd','Hnd'] + line.strip()[1:-1].split('/'))) for line in lines[start_index+1:end_index]]
            m.close()
        
        if self.verbose:
            print 'Found {} images.'.format(len(self.images))
    
        return self.images


    def getLabels(self):

        # uzima labele slika i vraca ih u listu, da bi slika bila ispravno oznacena
        mfiles = [os.path.join(self.folder_labels,mfile) for mfile in glob(os.path.join(self.folder_labels,'*.m'))]

        self.labels = []        
 
        for mfile in mfiles:
            m = open(mfile, "r")
            lines = m.readlines()
            for index, line in enumerate(lines):
                if line.startswith('list.ALLlabels'):
                    start_index = index
                    start_label = line[18:].strip()[:-1 ]
                    self.labels.append(start_label)
                    
                elif line.startswith('list.ALLnames'):
                    end_index = index - 1
            self.labels += [line.strip()[:-1] for line in lines[start_index+1:end_index]]
            m.close()

        #posmatramo mala i velika slova jednako
        keys = range(1,63)
        values = map(str, range(10)) + list(string.ascii_lowercase) + list(string.ascii_lowercase) 
        
        classes = dict(zip(keys, values))
        self.labels = map(lambda x: classes[int(x)], self.labels)
        
        if self.verbose:
            print 'Found {} labels.'.format(len(self.labels))        
           
        return self.labels 


    def split_train_test(self):

        #imamo 2 dataseta: train i test
        seed(10)
        total = len(self.ocr['target'])
        population = range(total)
        if self.split==0:
            self.images_train = self.ocr['images']
            self.data_train = self.ocr['data']
            self.labels_train = self.ocr['target']
            
            return self.images_train, self.data_train, self.labels_train
        else:
            k = int(np.floor(total * self.split))
            test = sample(population, k)
            train = [i for i in population if i not in test]
            
            self.images_train = self.ocr['images'][train]
            self.data_train = self.ocr['data'][train]
            self.labels_train = self.ocr['target'][train]
            
            self.images_test = self.ocr['images'][test]
            self.data_test = self.ocr['data'][test]
            self.labels_test = self.ocr['target'][test]
    
            return self.images_train, self.data_train, self.labels_train, self.images_test, self.data_test, self.labels_test


    def set_models(self):

        #postavljanje parametara za CV, i dobijamo podatke koje koristima za input u GridSearchCV
        models = {
            'linearsvc': (
                LinearSVC(),
                {'C':  list(np.arange(0.01,1.5,0.01))}, 
                ),

            'linearsvc-hog': (
                Pipeline([
                    ('hog', HOGFeatures(
                        orientations=2,
                        pixels_per_cell=(2, 2),
                        cells_per_block=(2, 2),
                        size = self.img_size
                        )), ('clf', LinearSVC(C=1.0))]),

                {
                    'hog__orientations': [2, 4, 5, 10],
                    'hog__pixels_per_cell': [(2,2), (4,4), (5,5)],
                    'hog__cells_per_block': [(2,2), (4,4), (5,5)],
                    'clf__C': [0.01, 0.05, 0.1, 0.5, 1, 1.5, 2, 5, 10],
                    },
                ),
            }

        return models


class HOGFeatures(BaseEstimator):

    #implementacija Histogram Of Gradients
    def __init__(self,
                 size,
                 orientations=8,
                 pixels_per_cell=(10, 10),
                 cells_per_block=(1, 1)):

        super(HOGFeatures, self).__init__()
        self.orientations = orientations
        self.pixels_per_cell = pixels_per_cell
        self.cells_per_block = cells_per_block
        self.size = size

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X = X.reshape((X.shape[0], self.size[0], self.size[1]))
        result = []
        for image in X:
            #image = rgb2gray(image)
            features = hog(
                image,
                orientations=self.orientations,
                pixels_per_cell=self.pixels_per_cell,
                cells_per_block=self.cells_per_block,
                )
            result.append(features)
        return np.array(result)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    