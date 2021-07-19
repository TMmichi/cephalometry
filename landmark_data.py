# Produces landmark data out of each images
from dataset_function import *
import cv2 as cv
import os
import xlrd
import re
####################################################################
#SWITCH
Training = True                                #Make Training set
Evaluating = False                             #Make Evaluatng set
isISBI = True
isSeive = True

# CONSTANTS
#LANDMARK_CONSIDERATION = [1,6,7,2,8,11,10,19,9,18,23,20,46,50,43,53,12,13,16]
#LANDMARK_CONSIDERATION = ["Sella", "Nasion", "Orbitale", "Porion", "A_point", "B_point", \
#                            "Pogonion", "Menton", "Gnathion", "Gonion", "Lower_Incisal_incision", \
#                            "Upper_Incisal_incision", "Upper_lip", "Lower_lip", "Subnasale", "Soft_Tissue_pogonion", \
#                            "Posterior_Nasal_Spine", "Anterior_Nasal_Spine", "Articulate"]
#LANDMARK_CONSIDERATION = ["Sella", "Nasion", "Orbitale", "Porion", "A_point"]
#LANDMARK_CONSIDERATION = ["B_point", "Pogonion", "Menton","Gnathion", "Gonion"]
#LANDMARK_CONSIDERATION = ["Lower_Incisal_incision", "Upper_Incisal_incision", "Upper_lip", "Lower_lip", "Subnasale"]
#LANDMARK_CONSIDERATION = ["Soft_Tissue_pogonion", "Posterior_Nasal_Spine","Anterior_Nasal_Spine", "Articulate"]

#LANDMARK_CONSIDERATION = ["Sella", "Nasion", "Orbitale"]
#LANDMARK_CONSIDERATION = ["Porion","A_point", "B_point"]
#LANDMARK_CONSIDERATION = ["Pogonion","Menton","Gnathion"]
#LANDMARK_CONSIDERATION = ["Gonion","Lower_Incisal_incision","Upper_Incisal_incision"]
#LANDMARK_CONSIDERATION = ["Upper_lip","Lower_lip","Subnasale"]
LANDMARK_CONSIDERATION = ["Soft_Tissue_pogonion", "Posterior_Nasal_Spine","Anterior_Nasal_Spine", "Articulate"]
IMAGE_SIZE = 91                                 # pixel
DOWNSAMPLEFACTOR = 3                            # Downsampling factor
if not isISBI:
    IMAGE_NUMBERS = 5200                        # Total number of image data
    PRED_INIT_NUMBER = 5000                     # Initial image number for Prediction
else:
    IMAGE_NUMBERS = 400                         # Total number of image data
    PRED_INIT_NUMBER = 151                      # Initial image number for Prediction

if not isSeive:
    FALSE_SAMPLE_NUMBER = 500                       # Number of false sample images
    True_multiplier = int(500/25 * 0.4)
    TRUE_SAMPLE_NUMBER = 25 * True_multiplier
    FALSE_MAX_REF_RADIUS = 40                        #MM
    FALSE_MIN_REF_RADIUS = 2.1                       #MM
    TRUE_REF_RADIUS = 1
else:
    FALSE_SAMPLE_NUMBER = 1000                       # Number of false sample images
    TRUE_SAMPLE_NUMBER = 400
    TRUE_REF_RADIUS = 18
    FALSE_MAX_REF_RADIUS = -1                        #MM
    FALSE_MIN_REF_RADIUS = 2.1                       #MM
TOTAL_SAMPLE_NUMBER = FALSE_SAMPLE_NUMBER + TRUE_SAMPLE_NUMBER  # False sample number + True sample number(25)
EVAL_REF_DIAMETER = 40

#FOLDER DIRECTORY
if isISBI:
    Raw_data_folder_directory = "/home/ljh/Projects/cephalometry/data/ISBI/raw_data/"
else:
    Raw_data_folder_directory = "/home/ljh/Projects/cephalometry/data/EWHA/raw_data/"
if isSeive:
    if isISBI:
        Training_data_folder_directory = "/home/ljh/Projects/cephalometry/data/ISBI/seive_training_set/"
    else:
        Training_data_folder_directory = "/home/ljh/Projects/cephalometry/data/EHWA/seive_training_set/"
else:
    if isISBI:
        Training_data_folder_directory = "/home/ljh/Projects/cephalometry/data/ISBI/training_set/"
    else:
        Training_data_folder_directory = "/home/ljh/Projects/cephalometry/data/EWHA/training_set/"
Eval_data_folder_directory = "/home/ljh/Projects/cephalometry/data/eval_data"
Averaged_landmark_position_directory = 'Average_landmark_position_ISBI.txt'
####################################################################

class landmark_data(object):
    def __init__(self, image_index, landmark_datapath, isISBI, downsamplefactor=3):
        self.path = landmark_datapath
        self.downsample = downsamplefactor
        self.position = []
        self.label = []
        self.image_index = image_index

    def processdata(self):
        if not isISBI:
            def data_read():
                rawdata = []
                wb = xlrd.open_workbook(self.path)
                sh = wb.sheet_by_name('Tracing coordination')
                for rownum in range(sh.nrows):
                    rawdata.append(sh.row_values(rownum))
                return rawdata

            data = data_read()
            data_index = 0
            for iter in range(len(LANDMARK_CONSIDERATION)):
                for lines in data:
                    character = lines[0]
                    if type(character) == float:
                        if int(character) == LANDMARK_CONSIDERATION[data_index]:
                            data_index += 1
                            # image downsampling
                            lines[2] = int(int(lines[2])/self.downsample)
                            lines[3] = int(int(lines[3])/self.downsample)
                            self.position.append([lines[3], lines[2]])
                            self.label.append(int(lines[0]))
                            break
                        else:
                            pass
                    else:
                        pass
        else:
            p = re.compile('\d*\d')
            with open(self.path,'r') as file:
                data = file.readlines()
            for i in range(len(LANDMARK_CONSIDERATION)):
                position_string = data[i]
                m = p.findall(position_string)
                self.position.append([int(int(m[1])/self.downsample),int(int(m[0])/self.downsample)])
                self.label.append(LANDMARK_CONSIDERATION[i])

        return [self.position, self.label]

# Produces training image sets (25 positive, 200 negative) of a single lateral ceph of each landmarks
# input: landmark_data (position, label)
# output: N by N 525 images of each landmarks

class trainingset(object):

    def __init__(self, image_index, landmark_datum, image_dimension, downsamplefactor=3):
        self.image_index = image_index
        self.landmark_positions = landmark_datum.position  # landmark_positions
        self.landmark_labels = landmark_datum.label  # landmark_labels
        self.Raw_image = None
        self.image_size = image_dimension
        self.dsf = downsamplefactor

    def make_imageset(self):
        print("Image " + str(self.image_index)+" manipulation")
        for lan_num in range(0, len(self.landmark_labels)):
            batch_number = 0
            # Raw_image call by read_image
            if not isISBI:
                raw_image_path = Raw_data_folder_directory+"/"+str(image_index)+".jpg"
            else:
                temp_index = format(self.image_index,"03d")
                raw_image_path = Raw_data_folder_directory+"/"+str(temp_index)+".bmp"
            self.Raw_image = read_image(raw_image_path, self.dsf)
            height, width = self.Raw_image.shape[:2]
            print(self.landmark_positions[lan_num])

            # image_type = true
            # Extract sample_positions from position of i-landmark
            position_data = sample_training_position(self.landmark_positions[lan_num], 1, FALSE_SAMPLE_NUMBER, FALSE_MAX_REF_RADIUS, FALSE_MIN_REF_RADIUS, TRUE_SAMPLE_NUMBER, TRUE_REF_RADIUS, self.dsf, width, height)

            # EX) Directory : training_set/image3/Landmark_Sella
            batch_directory = Training_data_folder_directory+"/image"+str(self.image_index)+"/Landmark_"+self.landmark_labels[lan_num]
            try:
                os.makedirs(batch_directory)
            except OSError:
                pass

            # produce sample images from each sample_training_position -> append [image,label] to image_true set
            for positions in position_data:
                batch_number += 1
                if positions[0] >= height or positions[1] >= width:
                    print("Warning, None proper true sampled position found in image " + str(self.image_index)
                          + ", landmark " + str(LANDMARK_CONSIDERATION[lan_num]) + ", batch " + str(batch_number))
                batch_image, label_temp = produce_image(self.Raw_image, positions, self.landmark_labels[lan_num], self.image_size)
                # EX) Batch image name: image3_Sella_true_17
                batch_name = "image"+str(self.image_index)+"_"+""+self.landmark_labels[lan_num] + "_true_"\
                             + str(batch_number)+".jpg"
                cv.imwrite(batch_directory+"/"+batch_name, batch_image)

            position_dir = batch_directory + "/True_sample_set positions.txt"
            file = open(position_dir, 'w')
            file.write(str(position_data))
            file.close()

            batch_number = 0

            # image_type = false
            # Extract 500 sample_positions from position of i-landmark
            position_data = sample_training_position(self.landmark_positions[lan_num], 0, FALSE_SAMPLE_NUMBER, FALSE_MAX_REF_RADIUS, FALSE_MIN_REF_RADIUS, TRUE_SAMPLE_NUMBER, TRUE_REF_RADIUS, self.dsf, width, height)
            # produce sample images from each sample_training_position -> append [image,label] to image_true set
            for positions in position_data:
                batch_number += 1
                if positions[0] >= height or positions[1] >= width:
                    print("Warning, None proper false sampled position found in image "
                          + str(self.image_index) +", landmark " + str(LANDMARK_CONSIDERATION[lan_num]) + ", batch " + str(batch_number))
                batch_image, label_temp = produce_image(self.Raw_image, positions, self.landmark_labels[lan_num], self.image_size)
                # EX) Batch image name: image3_Sella_false_17
                batch_name = "image"+str(self.image_index)+"_"+""+self.landmark_labels[lan_num] + "_false_"\
                             + str(batch_number)+".jpg"
                cv.imwrite(batch_directory+"/"+batch_name, batch_image)

            position_dir = batch_directory + "/False_sample_set positions.txt"
            file = open(position_dir, 'w')
            file.write(str(position_data))
            file.close()

            label_dir = batch_directory + "/Landmark_"+str(LANDMARK_CONSIDERATION[lan_num]) + ".txt"
            file = open(label_dir, 'w')
            for j in range(0, TOTAL_SAMPLE_NUMBER):
                if j < TRUE_SAMPLE_NUMBER:
                    file.write(str(1)+",")
                elif TRUE_SAMPLE_NUMBER <= j < (TOTAL_SAMPLE_NUMBER-1):
                    file.write(str(0)+",")
                else:
                    file.write(str(0))
            file.close()
            print("landmark data_" + str(LANDMARK_CONSIDERATION[lan_num]) + " formed")
        else:
            return


if __name__ == '__main__':
    image_size = IMAGE_SIZE
    for image_index in range(150, 151):
        if isISBI:
            index_temp = format(image_index,'03d')
            landmark_data_path = Raw_data_folder_directory + "/" + str(index_temp) + ".txt"
        else:
            landmark_data_path = Raw_data_folder_directory + "/" + str(image_index) + ".xlsx"
        landmark_temp = landmark_data(image_index, landmark_data_path, True,DOWNSAMPLEFACTOR)
        position, label = landmark_temp.processdata()
        train_data = trainingset(image_index, landmark_temp, IMAGE_SIZE, DOWNSAMPLEFACTOR)
        train_data.make_imageset()
