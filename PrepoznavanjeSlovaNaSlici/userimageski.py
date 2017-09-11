import numpy as np
from skimage.io import imread
from skimage.filter import threshold_otsu
from skimage.transform import resize
import cPickle
from matplotlib import pyplot as plt
from skimage.morphology import closing, square
from skimage.measure import regionprops
from skimage import restoration
from skimage import measure
from skimage.color import label2rgb
import matplotlib.patches as mpatches

class UserData():
    """
    Klasa za procesiranje i obradu izabrane slike.
    """

    def __init__(self, image_file):

        # cita sliku kao grayscale i preprocesira je
        self.image = imread(image_file, as_grey=True)
        self.preprocess_image()


    def plot_preprocessed_image(self):

        # cuva i prikazuje stanje slike pre nego sto je precesirana
        image = restoration.denoise_tv_chambolle(self.image, weight=0.1)
        thresh = threshold_otsu(image)
        bw = closing(image > thresh, square(2))
        cleared = bw.copy()

        label_image = measure.label(cleared)
        borders = np.logical_xor(bw, cleared)

        label_image[borders] = -1
        image_label_overlay = label2rgb(label_image, image=image)

        fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(12, 12))
        ax.imshow(image_label_overlay)

        for region in regionprops(label_image):
            if region.area < 10:
                continue

            minr, minc, maxr, maxc = region.bbox
            rect = mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                      fill=False, edgecolor='red', linewidth=2)
            ax.add_patch(rect)

        plt.show()

    def preprocess_image(self):

        # postavlja kontrast slike
        image = restoration.denoise_tv_chambolle(self.image, weight=0.01)
        thresh = threshold_otsu(image)
        self.bw = closing(image > thresh, square(2))
        self.cleared = self.bw.copy()
        return self.cleared


    def get_text_candidates(self):

        # prepoznavanje konture objekta u slici
        label_image = measure.label(self.cleared)   
        borders = np.logical_xor(self.bw, self.cleared)
        label_image[borders] = -1
        
        coordinates = []
        i=0

        # iscrtavanje kvadrata oko pronadjenih objekata i cuvanje njihovih koordinata
        for region in regionprops(label_image):
            if region.area > 10:
                minr, minc, maxr, maxc = region.bbox
                margin = 3
                minr, minc, maxr, maxc = minr-margin, minc-margin, maxr+margin, maxc+margin
                roi = self.image[minr:maxr, minc:maxc]
                if roi.shape[0]*roi.shape[1] == 0:
                    continue
                else:
                    if i==0:
                        # promenjena velicina objekta na 20x20 piksela
                        samples = resize(roi, (20,20))
                        coordinates.append(region.bbox)
                        i+=1
                    elif i==1:
                        roismall = resize(roi, (20,20))
                        samples = np.concatenate((samples[None,:,:], roismall[None,:,:]), axis=0)
                        coordinates.append(region.bbox)
                        i+=1
                    else:
                        roismall = resize(roi, (20,20))
                        samples = np.concatenate((samples[:,:,:], roismall[None,:,:]), axis=0)
                        coordinates.append(region.bbox)

        # cuvanje svakog iscrtanog kvadrata sa objektom kao posebnu sliku
        self.candidates = {
            'total': samples,
            '3d': samples.reshape((samples.shape[0], -1)),
            'coordinates': np.array(coordinates)
            }
        
        print 'Slika pronadjenih objekata na slici'
        print 'Ukupno: ', self.candidates['total'].shape
        print 'Koordinate konura: ', self.candidates['coordinates'].shape
        print '============================================================'
        
        return self.candidates

    def select_text_among_candidates(self, model_filename2):

        #koristeci .pickle model odredjujemo da li objekat sadrzi tekst
        with open(model_filename2, 'rb') as fin:
            model = cPickle.load(fin)
            
        is_text = model.predict(self.candidates['3d'])
        
        self.to_be_classified = {
            'total': self.candidates['total'][is_text == '1'],
            '3d': self.candidates['3d'][is_text == '1'],
            'coordinates': self.candidates['coordinates'][is_text == '1']
        }

        print 'Slika nakon detekcije objekta'
        print 'Ukupno: ', self.to_be_classified['total'].shape
        print 'Koordinate konura: ', self.to_be_classified['coordinates'].shape
        print 'Pronadjeni kvadrai koji ne sadrze tekst '+str(self.candidates['coordinates'].shape[0]-self.to_be_classified['coordinates'].shape[0])+' od '+str(self.candidates['coordinates'].shape[0])
        print '============================================================'

        return self.to_be_classified

    def classify_text(self, model_filename36):

        # prepoznavanje slova koriscenjem argumenata iz .pickle modela
        with open(model_filename36, 'rb') as fin:
            model = cPickle.load(fin)
            
        which_text = model.predict(self.to_be_classified['3d'])
        
        self.which_text = {
            'total': self.to_be_classified['total'],
            '3d': self.to_be_classified['3d'],
            'coordinates': self.to_be_classified['coordinates'],
            'predicted_char': which_text
        }

        return self.which_text

    def plot_to_check(self, what_to_plot, title):

        # ispisuje slovo ispred odgovarajuceg kvadrate pre nego sto se prikaze u 2d prostoru
        n_images = what_to_plot['total'].shape[0]

        fig = plt.figure(figsize=(12, 12))

        if n_images <= 100:
            if n_images < 100:
                total = range(n_images)
            elif n_images == 100:
                total = range(100)

            for i in total:
                ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
                ax.imshow(what_to_plot['total'][i], cmap="Greys_r")
                if 'predicted_char' in what_to_plot:
                    ax.text(-6, 8, str(what_to_plot['predicted_char'][i]), fontsize=22, color='red')
            plt.suptitle(title, fontsize=20)
            plt.show()
        else:
            total = list(np.random.choice(n_images, 100))
            for i, j in enumerate(total):
                ax = fig.add_subplot(10, 10, i + 1, xticks=[], yticks=[])
                ax.imshow(what_to_plot['total'][j], cmap="Greys_r")
                if 'predicted_char' in what_to_plot:
                    ax.text(-6, 8, str(what_to_plot['predicted_char'][j]), fontsize=22, color='red')
            plt.suptitle(title, fontsize=20)

        plt.show()

    def realign_text(self):

        # procesira pronadjeno slovo i ispisuje ga u matplotlib sliku
        max_maxrow = max(self.which_text['coordinates'][:,2])
        min_mincol = min(self.which_text['coordinates'][:,1])
        subtract_max = np.array([max_maxrow, min_mincol, max_maxrow, min_mincol]) 
        flip_coord = np.array([-1, 1, -1, 1])
        
        coordinates = (self.which_text['coordinates'] - subtract_max) * flip_coord
        
        ymax = max(coordinates[:,0])
        xmax = max(coordinates[:,3])
        
        coordinates = [list(coordinate) for coordinate in coordinates]
        predicted = [list(letter) for letter in self.which_text['predicted_char']]
        to_realign = zip(coordinates, predicted)
        
        fig = plt.figure()
        ax = fig.add_subplot(111)
        for char in to_realign:
            ax.text(char[0][1], char[0][2], char[1][0], size=16)
        ax.set_ylim(-10,ymax+10)
        ax.set_xlim(-10,xmax+10)
        
        plt.show()

        
            
        
        
        
        
    