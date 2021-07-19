import math
import numpy as np
import xlrd
import os
import random
import datetime
import cv2 as cv
import scipy.stats as stats
import re
#import matplotlib.pyplot as plt
####################################################################
#CONSTANTS
gpu_num = "0"
MAX_SCORE = 2500
ALPHA = 0.85 #The higher, the denser the original image
####################################################################

def sample_training_position(position, image_type, false_sample_number, false_max_radius, false_min_radius, true_sample_number, true_max_radius, dsf, width, height):
    '''
    #input: position, image_type(true or false), false sample number, MAX and MIN ref diameter
    #output: position vectors(25*True_multiplier positions if true, false_sample_number if false)
    #true image -> {5x5 images(+- 0.6mm)} surrounding input location including particular pixel
    #false image -> Out of [{267x267 images(+- 40mm)} - {15x15 images(+- 2.1mm)}], choose 500 among them randomly.
    '''
    sample_positions = []

    #if position == True
    if image_type:
        current = datetime.datetime.now()
        random.seed(current+datetime.timedelta(0))
        true_max_radius_pixel = int(true_max_radius/0.1/dsf)
        for i in range(0, true_sample_number):
            if position[1] + true_max_radius >= width:
                x_max = width-1
            else:
                x_max = position[1] + true_max_radius
            if position[1] - true_max_radius <= 0:
                x_min = 0
            else:
                x_min = position[1] - true_max_radius
            x_position = random.randint(x_min, x_max)
            if position[0] + int(math.sqrt(true_max_radius_pixel**2-(x_position - position[1])**2)) >= height:
                y_max = height-1
            else:
                y_max = position[0] + int(math.sqrt(true_max_radius_pixel**2-(x_position - position[1])**2))
            if position[0] - int(math.sqrt(true_max_radius_pixel**2-(x_position - position[1])**2)) < 0:
                y_min = 0
            else:
                y_min = position[0] - int(math.sqrt(true_max_radius_pixel**2-(x_position - position[1])**2))
            y_position = random.randint(y_min, y_max)
            sample_positions.append([y_position, x_position])

    #if position == False
    else:
        current = datetime.datetime.now()
        random.seed(current+datetime.timedelta(0))
        false_max_radius_pixel = int(false_max_radius/0.1/dsf)
        false_min_radius_pixel = int(false_min_radius/0.1/dsf)
        true_max_radius_pixel = int(true_max_radius/0.1/dsf)
        for i in range(0, false_sample_number):
            if false_max_radius > 0:
                if position[1] + false_max_radius_pixel >= width:
                    x_max = width-1
                else:
                    x_max = position[1] + false_max_radius_pixel
                if position[1] - false_max_radius_pixel < 0:
                    x_min = 0
                else:
                    x_min = position[1] - false_max_radius_pixel
                x_position = random.randint(x_min, x_max)

                y_rad = int(math.sqrt(false_max_radius_pixel**2-(x_position - position[1])**2))
                if position[0] + y_rad >= height:
                    y_max = height-1
                else:
                    y_max = position[0] + y_rad
                if position[0] - y_rad < 0:
                    y_min = 0
                else:
                    y_min = position[0] - y_rad
                if abs(x_position - position[1]) > false_min_radius_pixel:
                    y_position = random.randint(y_min,y_max)
                else:
                    y_list = np.linspace(y_min,y_max,y_max-y_min+1)
                    false_rad = int(math.sqrt(false_min_radius_pixel**2 - (x_position - position[1])**2))
                    remove_max = position[0] + false_rad
                    remove_min = position[0] - false_rad
                    remove_num = remove_max - remove_min + 1
                    remove_list = np.linspace(remove_min, remove_max, remove_num)
                    y_list = np.delete(y_list,remove_list)
                    y_position = int(random.choice(y_list))
                sample_positions.append([y_position, x_position])
            else:
                x_max = width-1
                x_min = 0
                x_position = random.randint(x_min, x_max)
                if not position[1] - true_max_radius_pixel < x_position < position[1] + true_max_radius_pixel:
                    y_max = height-1
                    y_min = 0
                    y_position = random.randint(y_min, y_max)
                else:
                    y_rad = int(math.sqrt(true_max_radius_pixel**2-(x_position - position[1])**2))
                    if position[0] - true_max_radius_pixel < 0:
                        y_min = position[0] + y_rad
                        y_max = height-1
                        y_position = random.randint(y_min, y_max)
                    elif position[0] + true_max_radius_pixel >= height:
                        y_min = 0
                        y_max = position[0] - y_rad
                        y_position = random.randint(y_min, y_max)
                    else:
                        y_list = np.linspace(0,height-1,height)
                        remove_min = position[0] - y_rad
                        remove_max = position[0] + y_rad
                        remove_num = remove_max - remove_min + 1
                        remove_list = np.linspace(remove_min, remove_max, remove_num)
                        y_list = np.delete(y_list,remove_list)
                        y_position = int(random.choice(y_list))
                sample_positions.append([y_position, x_position])
    return sample_positions

def sample_predicting_position(position, ref_diameter, dsf):
    #ref_diameter in mm
    sample_positions = []
    sample_size = int(ref_diameter*10/dsf)
    print(ref_diameter)
    print(sample_size)
    if sample_size % 2 == 0:
        sample_size += 1
    sample_radius = int((sample_size-1)/2)
    for size in range(0,sample_radius):
        if size == 0:
            #Height, Width
            temp = [int(position[0]),int(position[1])]
            sample_positions.append(temp)
        else:
            for height in range(-1*size+1,size):
                if (height == -1*size+1) or (height == size-1):
                    for width in range(-1*size+1,size):
                        temp = [int(position[0]+height),int(position[1]+width)]
                        sample_positions.append(temp)
                else:
                    width = -1*size+1
                    temp = [int(position[0]+height),int(position[1]+width)]
                    sample_positions.append(temp)
                    width = size-1
                    temp = [int(position[0]+height),int(position[1]+width)]
                    sample_positions.append(temp)
    return sample_positions

def sample_seiving_position(height, width, step_size):
    y = np.linspace(0,height,int(height/step_size),endpoint=False)
    x = np.linspace(0,width,int(width/step_size),endpoint=False)
    sample_positions = []
    for x_position in x:
        for y_position in y:
            sample_positions.append([int(y_position),int(x_position)])
    return sample_positions

def read_image(raw_image_path, dsf):
    '''
    #input: path of an image, downsample factor
    #output: downsampled Raw image vector
    #need to downsample image
    '''
    raw_image = cv.imread(raw_image_path,0)
    height, width = raw_image.shape[:2]
    new_raw_image = cv.resize(raw_image, (int(width/dsf), int(height/dsf)), interpolation=cv.INTER_AREA)
    #image[i][j] -> image[height index][width index] // From left upper corner to down, right direction
    return new_raw_image

def produce_image(raw_image, position, label, image_size):
    '''
    #input: Raw_image, 1 position, label data from sample_position, image_size
    #output: 1 image of NxN pixels (N = image_size), label
    #padding should be included depending on whether CNN model from tf compensates or not
    '''
    image = np.array(raw_image)
    height, width = np.shape(image)
    if image_size % 2 == 0:
        raise NameError("Warning: input image size should be an odd number")
    y_start = int(position[0] - (image_size-1)/2)
    y_end = int(position[0] + (image_size-1)/2)
    x_start = int(position[1] - (image_size-1)/2)
    x_end = int(position[1] + (image_size-1)/2)
    #If pixel positions not in boundary of raw_image
    if y_start < 0 or y_end >= height or x_start < 0 or x_end >= width:
        x_pad1 = 0
        x_pad2 = 0
        y_pad1 = 0
        y_pad2 = 0
        if y_start < 0:
            y_pad1 = abs(y_start)
            y_start = 0
        if y_end >= height:
            y_pad2 = min(abs(height - y_end)+1 , 91)
            y_end = height-1
        if x_start < 0:
            x_pad1 = abs(x_start)
            x_start = 0
        if x_end >= width:
            x_pad2 = min(abs(width - x_end)+1 , 91)
            x_end = width-1
        image1 = image[y_start:y_end+1, x_start:x_end+1]
        crop_image = cv.copyMakeBorder(image1, y_pad1, y_pad2, x_pad1, x_pad2, cv.BORDER_CONSTANT, value = 0)
        #print(image1.shape[:2], crop_image.shape[:2])
    #If pixel position is within boundary of raw_image -> image croped from start point to end
    else:
        crop_image = image[y_start:y_end+1, x_start:x_end+1] #+1 due to: end point of index not included
    height, width = crop_image.shape[:2]
    #if height != image_size or width != image_size:
    #    print(height, width)
    return [crop_image, label]

def bring_inputset(training_data_folder_directory, image_start_num, image_end_num, landmark_num, false_sample_number, true_sample_number):
    print("Importing Input sets for landmark "+str(landmark_num))
    image = []
    labels = []
    for i in range(image_start_num, image_end_num+1):
        file_directory = training_data_folder_directory+"/"+"image"+str(i)+"/Landmark_"+str(landmark_num)+"/"
        for j in range(0, true_sample_number):
            batch_name = "image"+str(i)+"_"+str(landmark_num)+"_true_"+str(j+1)+".jpg"
            image_temp = cv.imread(file_directory+batch_name, 0)
            height, width = image_temp.shape[:2]
            image_temp = np.reshape(image_temp, [height*width])
            image.append(image_temp)
        for j in range(0, false_sample_number):
            batch_name = "image"+str(i)+"_"+str(landmark_num)+"_false_"+str(j+1)+".jpg"
            image_temp = cv.imread(file_directory+batch_name, 0)
            height, width = image_temp.shape[:2]
            image_temp = np.reshape(image_temp, [height*width])
            image.append(image_temp)
        print("Input sets for image "+str(i)+" imported")

        file = open(file_directory+"/Landmark_"+str(landmark_num)+".txt", 'r')
        rawdata = file.readlines()
        file.close()
        label_str = rawdata[0]
        label_str_split = label_str.split(',')
        for numbers in label_str_split:
            labels.append(int(numbers))
    return np.asarray(image, dtype=np.float32), np.asarray(labels, dtype=np.int32)

def bring_evalset_make(image_index, landmark_number, Raw_data_folder_directory, ref_diameter, dsf, position, image_size, isISBI):
    print("Importing Eval sets for landmark"+str(landmark_number))
    iter_index = 0
    image = []
    sample_positions = sample_predicting_position(position,ref_diameter, dsf)
    if not isISBI:
        raw_image_path = Raw_data_folder_directory + "/" + str(image_index) +".jpg"
    else:
        raw_image_path = Raw_data_folder_directory + "/" + str(image_index) +".bmp"
    raw_image = read_image(raw_image_path, dsf)
    for i in range(0, len(sample_positions)):
        iter_index += 1
        position_temp = sample_positions[i]
        image_temp, _ = produce_image(raw_image, position_temp, 1, image_size)
        height, width = image_temp.shape[:2]
        image_temp = np.reshape(image_temp, [height*width])
        image.append(image_temp)
        if iter_index % 1000 == 0:
            print("Eval set "+str(i+1)+" formed")
    print("last_index = ", iter_index)
    return np.asarray(image, dtype=np.float32), sample_positions

def bring_predset(raw_image_path, position, ref_diameter, dsf, image_size, landmark_number):
    raw_image = read_image(raw_image_path, dsf)
    sample_positions = sample_predicting_position(position, ref_diameter, dsf)
    image = []
    iter_index = 0
    print("# of Pred sets: "+str(len(sample_positions)))
    for i in range(0, len(sample_positions)):
        iter_index += 1
        position_temp = sample_positions[i]
        image_temp, junk = produce_image(raw_image, position_temp, 1, image_size)
        cv.imwrite("data/PRED_IMAGE/"+str(landmark_number)+"/"+str(position_temp)+".jpg",image_temp)
        height, width = image_temp.shape[:2]
        image_temp = np.reshape(image_temp, [height*width])
        image.append(image_temp)
        if iter_index % 1000 == 0:
            print("Pred set "+str(i+1)+" appended")
    return np.asarray(image, dtype=np.float32), np.asarray(sample_positions, dtype=np.uint16)

def bring_seive_make(image_index, landmark_number, Raw_data_folder_directory, dsf, image_size, isISBI):
    print("Importing Eval sets for landmark"+str(landmark_number))
    iter_index = 0
    image = []
    if not isISBI:
        raw_image_path = Raw_data_folder_directory + "/" + str(image_index) +".jpg"
    else:
        raw_image_path = Raw_data_folder_directory + "/" + str(image_index) +".bmp"
    raw_image = read_image(raw_image_path, dsf)
    height, width = raw_image.shape[:2]
    sample_positions = sample_seiving_position(height, width, step_size=10)
    for i in range(0, len(sample_positions)):
        position_temp = sample_positions[i]
        image_temp, _ = produce_image(raw_image, position_temp, 1, image_size)
        height1, width1 = image_temp.shape[:2]
        image_temp = np.reshape(image_temp, [height1*width1])
        image.append(image_temp)
    print("Seive set for landmark_" +str(landmark_number)+" formed")
    return np.asarray(image, dtype=np.float32), sample_positions

def average_position(pos_dict, position, label):
    if pos_dict["num"] == 0:
        for i in range(0, len(label)):
            pos_dict[label[i]] = position[i]
        pos_dict["num"] += 1
    else:
        for j in range(0, len(label)):
            accum_position = pos_dict[label[j]]
            add_position = position[j]
            temp2 = pos_dict["num"]+1
            tempx = (accum_position[0] * pos_dict["num"] + add_position[0]) / temp2
            tempy = (accum_position[1] * pos_dict["num"] + add_position[1]) / temp2
            new_position = (tempx, tempy)
            pos_dict[label[j]] = new_position
        pos_dict["num"] += 1
    return pos_dict

def write_position_dict(position_dict,Averaged_landmark_position_directory):
    label = position_dict.keys()
    file = open(Averaged_landmark_position_directory, 'w')
    for member in label:
        file.write(str(member))
        file.write('/')
        file.writelines(str(position_dict[member]))
        file.write("_")
    file.close()

def read_position_dict(dict_path):
    file = open(dict_path, 'r')
    position = file.readlines()
    file.close()
    position = position[0]
    position = position.split('_')
    position_dict = {}
    for members in position:
        temp = members.split('/')
        numb = temp[0]
        if numb.isdigit():
            position_dict[temp[0]] = temp[1]

    return position_dict

def get_pred_landmark_position(landmark_position_directory, landmark_number):
    pos_dict = read_position_dict(landmark_position_directory)
    landmark_position = pos_dict[str(landmark_number)]
    landmark_position = eval(landmark_position)
    print("Predicted landmark"+str(landmark_number)+ " position: "+str(landmark_position))
    return landmark_position

def position_process(eval_position, evaluated_classes, dsf, evaluated_softmax):
    eval_set = []
    true_set = []
    sum_position = [0, 0]
    num_true_image = 0
    if len(eval_position) != len(evaluated_classes):
        print("# of pred positions mismatches the # of predicted classes")
    else:
        for j in range(0, len(eval_position)):
            temp = [eval_position[j][1]*dsf, eval_position[j][0]*dsf, evaluated_classes[j], evaluated_softmax[j]]
            if evaluated_classes[j] == 1 and eval_position[j][0]*dsf <= 2510 and eval_position[j][1]*dsf <= 2000:
                sum_position[0] += eval_position[j][1] * dsf
                sum_position[1] += eval_position[j][0] * dsf
                num_true_image += 1
                true_set.append(temp)
            eval_set.append(temp)
    return eval_set, true_set, sum_position, num_true_image

def bayes_prob_averaging(eval_softmax_set, BAYES_ITER):
    mean_sum = np.zeros(len(eval_softmax_set[0]))
    stv_sum = np.zeros(len(eval_softmax_set[0]))
    for i in range(0,BAYES_ITER):
        mean_sum += eval_softmax_set[i]
    eval_mean = mean_sum/BAYES_ITER

    for i in range(0,BAYES_ITER):
        stv_sum = (eval_softmax_set[i]-eval_mean)**2
    eval_stv = 1/(BAYES_ITER-1)*np.sqrt(stv_sum)
    return [eval_mean, eval_stv]

def bayes_position_process_kde(eval_softmax_set, eval_position, DOWNSAMPLEFACTOR, Eval_log_folder_directory, image_number, landmark_number):
    position = []
    x_pos = []
    y_pos = []
    index_local_max = []
    probability = []
    #eval_position = [Height, Width]
    for i in range(len(eval_softmax_set)):
        index = np.argmax(eval_softmax_set[i])
        index_local_max.append(index)
        probability.append(eval_softmax_set[i][index])
        x_real = eval_position[index][1] * DOWNSAMPLEFACTOR
        y_real = eval_position[index][0] * DOWNSAMPLEFACTOR
        position.append([x_real,y_real])
        x_pos.append(position[i][0])
        y_pos.append(position[i][1])
    x_pos = np.reshape(np.asarray(x_pos),(len(x_pos),1))
    y_pos = np.reshape(np.asarray(y_pos),(len(y_pos),1))
    position = np.array(position)

    #eval_position = [Height, Width]
    sum_probability = np.sum(np.asarray(probability))
    print("value of sum_probability = ",sum_probability)
    probability = np.reshape(np.asarray(probability),(len(probability),1))
    norm_probability = probability / sum_probability
    weighted_position = position * norm_probability
    mean_position = np.sum(weighted_position, axis=0)
    bayes_eval_position_kde = (int(mean_position[0]),int(mean_position[1]))
    print("mean position = ", mean_position)
    cov_xx = np.sum((x_pos-mean_position[0]) * (x_pos-mean_position[0]) * norm_probability)
    cov_xy = np.sum((x_pos-mean_position[0]) * (y_pos-mean_position[1]) * norm_probability)
    cov_yy = np.sum((y_pos-mean_position[1]) * (y_pos-mean_position[1]) * norm_probability)
    norm_coef = 1/(1-np.sum(norm_probability*norm_probability))
    cov_matrix = norm_coef * np.array([[cov_xx,cov_xy],[cov_xy, cov_yy]])
    print("cov_matrix = ", cov_matrix)

    eigval, eigvec = np.linalg.eig(cov_matrix)
    if abs(eigval[0]) > abs(eigval[1]):
        large_eigvec = eigvec[:,0]
        large_eigval = eigval[0]
        small_eigval = eigval[1]
    else:
        large_eigvec = eigvec[:,1]
        large_eigval = eigval[1]
        small_eigval = eigval[0]
    print("eigvec = ", eigvec)
    print("eigval = ", eigval)
    print("large eigvec = ", large_eigvec)

    center = bayes_eval_position_kde
    angle = math.atan(large_eigvec[1]/large_eigvec[0])
    if angle < 0:
        angle += math.pi * 2
    angle = int(180 * angle / math.pi)
    print(angle)
    width = int(2*math.sqrt(5.991*abs(large_eigval)))
    height = int(2*math.sqrt(5.991*abs(small_eigval)))
    print(width,height)
    ellipse_data = [center, angle, width, height]
    return bayes_eval_position_kde, position, ellipse_data, index_local_max
    #return bayes_eval_position_kde, position, grid_position, probability_grid, index_local_max

def bayes_position_process_score(threshold_prob, eval_position, dsf, eval_mean, eval_stv, index_max = []):
    max_score = []
    mod_set = []
    square_sum_stv = 0
    for i in range(len(eval_mean)):
        if eval_mean[i] >= threshold_prob:
            temp = [eval_position[i][1]*dsf, eval_position[i][0]*dsf, eval_mean[i], eval_stv[i],0]
            mod_set.append(temp)
            square_sum_stv += (eval_stv[i]) ** 2
    if len(mod_set) >= 1:
        norm_stv = math.sqrt(square_sum_stv)
        score_sum = 0
        x_weighted_pos_sum = 0
        y_weighted_pos_sum = 0
        for i in range(len(mod_set)):
            inv_stv = norm_stv/(mod_set[i][3])
            sat_stv = math.tanh(inv_stv/100)
            score = (math.exp(mod_set[i][2]*10)-1)*sat_stv
            mod_set[i][4] = score
            score_sum += score
            x_weighted_pos_sum += mod_set[i][0] * score
            y_weighted_pos_sum += mod_set[i][1] * score
        bayes_eval_position = [x_weighted_pos_sum/score_sum,y_weighted_pos_sum/score_sum]
        if len(index_max) != 0:
            for j in range(len(index_max)):
                inv_stv = 1/(eval_stv[index_max[j]])
                sat_stv = math.tanh(inv_stv/100)
                score = (math.exp(eval_mean[index_max[j]]*10)-1)*sat_stv
                max_score.append(score)
            return bayes_eval_position, mod_set, max_score
        else:
            return bayes_eval_position, mod_set
    else:
        if len(index_max) != 0:
            for j in range(len(index_max)):
                inv_stv = 1/(eval_stv[index_max[j]])
                sat_stv = math.tanh(inv_stv/100)
                score = (math.exp(eval_mean[index_max[j]])-1)*sat_stv
                max_score.append(score)
            print("No positions have mean probability over the threshold_prob. Returning NONE\n")
            return [0,0], [], max_score
        else:
            return [0,0], []

def get_True_position(eval_data_directory, landmark_number, isISBI, landmark_index):
    if not isISBI:
        wb = xlrd.open_workbook(eval_data_directory)
        sh = wb.sheet_by_name('Tracing coordination')
        true_landmark_position = [0,0]
        for rownum in range(sh.nrows):
            temp = sh.row_values(rownum)
            if type(temp[0]) == float:
                if int(temp[0]) == landmark_number:
                    true_landmark_position[0] = temp[2]
                    true_landmark_position[1] = temp[3]
    else:
        p = re.compile('\d*\d')
        file = open(eval_data_directory)
        data = file.readlines()
        file.close()
        position_string = data[landmark_index]
        m = p.findall(position_string)
        true_landmark_position = [int(m[0]),int(m[1])]
    return true_landmark_position

def position_averaging(num_true_image,sum_position,true_landmark_position=[0,0]):
    if num_true_image >= 1:
        #Predicted Position -> Averaged
        average_position = [sum_position[0]/num_true_image, sum_position[1]/num_true_image]
        #Check whether the predicted position is located within 2mm from true landmark
        length = math.sqrt((true_landmark_position[0]-average_position[0])**2
                           + (true_landmark_position[0]-average_position[0])**2) * 0.1
    else:
        length = 0
        average_position = [0,0]
    return average_position, length

def write_log_eval_file(eval_log_directory, eval_set, true_set, true_landmark_position, average_eval_position, length):
    file = open(eval_log_directory, 'w')
    #file.write("X, Y, classes, probabilities\n")
    #for lines in eval_set:
        #file.writelines(str(lines)+"\n")
    #file.writelines(str("\n"))
    #file.writelines(str("\n"))
    file.write("X, Y, classes, probabilities\n")
    for lines in true_set:
        file.writelines(str(lines)+"\n")
    file.writelines(str("\n"))
    file.writelines(str("\n"))
    file.writelines("True Landmark Location = "+str(true_landmark_position)+"\n")
    file.writelines("Predicted Landmark Location = "+str(average_eval_position)+"\n")
    file.writelines("Distance = "+str(length)+"\n")
    file.close()

def write_log_bayes_eval_file(eval_log_directory, BAYES_ITER, eval_set, eval_position, dsf, eval_classes_set, eval_softmax_set, true_landmark_position, bayes_eval_position_score, bayes_eval_position_kde, mod_set):
    file = open(eval_log_directory, 'w')
    #file.write("index, position_X, position_Y")
    #for i in range(BAYES_ITER):
    #    string_class = ", classes"+str(i+1)
    #    string_softmax = ", softmax"+str(i+1)
    #    file.write(string_class)
    #    file.write(string_softmax)
    #file.write(", mean, stv\n")
    #for i in range(len(eval_set[0])):
    #    string1 = str(i+1)+str(", ")+str(eval_position[i][1]*dsf)+str(", ")+str(eval_position[i][0]*dsf)+str(", ")
    #    file.write(string1)
    #    for j in range(BAYES_ITER):
    #        string2 = str(eval_classes_set[j][i]) + str(", ") + str(eval_softmax_set[j][i]) + str(", ")
    #        file.write(string2)
    #    string3 = str(eval_set[0][i]) + str(", ") + str(eval_set[1][i])
    #    file.write(string3)
    #    file.write("\n")
    #mod_set: temp = [eval_position[i][1]*dsf, eval_position[i][0]*dsf, eval_mean[i], eval_stv[i],0]
    file.write("Mod_position X, Mod_position Y, eval_mean, eval_stv, score\n")
    if len(mod_set) >= 1:
        for i in range(len(mod_set)):
            string4 = str(i+1)+": "+str(mod_set[i][0])+", "+str(mod_set[i][1])+", "+str(mod_set[i][2])+", "+str(mod_set[i][3])+", "+str(mod_set[i][4])
            file.write(string4)
            file.write("\n")
    else:
        file.write("No positions have mean probability over the threshold_prob.\n")

    file.writelines("Evaluated Landmark Location (score) = "+str(bayes_eval_position_score)+"\n")
    file.writelines("Evaluated Landmark Location (kde) = "+str(bayes_eval_position_kde)+"\n")
    file.writelines("True Landmark Location = "+str(true_landmark_position)+"\n")
    length_score = math.sqrt((true_landmark_position[0]-bayes_eval_position_score[0])**2+(true_landmark_position[1]-bayes_eval_position_score[1])**2) * 0.1
    length_kde = math.sqrt((true_landmark_position[0]-bayes_eval_position_kde[0])**2+(true_landmark_position[1]-bayes_eval_position_kde[1])**2) * 0.1
    file.writelines("Length (kde) = "+str(length_kde)+" ")
    file.writelines("Length (score) = "+str(length_score))
    file.close()
    return length_kde, length_score

def write_log_pred_file(pred_log_directory, pred_set, true_set, average_pred_position):
    file = open(pred_log_directory, 'w')
    for lines in pred_set:
        file.writelines(str(lines)+"\n")
    file.writelines(str("\n"))
    file.writelines(str("\n"))
    for lines in true_set:
        file.writelines(str(lines)+"\n")
    file.writelines(str("\n"))
    file.writelines(str("\n"))
    file.writelines("Predicted Landmark Location = "+str(average_pred_position)+"\n")
    file.close()

def write_log_bayes_pred_file(pred_log_directory, pred_mean, pred_stv, pred_position):
    file = open(pred_log_directory, 'w')
    for i in range(len(pred_mean)):
        file.write("mean = ")
        file.write(str(pred_mean[i]))
        file.write(", stv = ")
        file.write(str(pred_stv[i]))
        file.write(" // position = ")
        file.write(str(pred_position[i][1])+' '+str(pred_position[i][0]))
        file.write("\n")
    file.close()

def draw_picture(Result_Image, Result_picture_directory, true_set, true_landmark_position, average_eval_position, landmark_number):
    print("Plotting evaluated result to the lateral cephalograph")
    color_code = {"1":[0,64,0],"6":[0,255,64],"7":[255,0,64],"2":[255,64,0],"8":[128,64,128],"11":[64,0,64],"10":[0,64,64],\
                  "19":[128,64,64],"9":[64,64,128],"18":[0,64,128],"23":[64,0,128],"20":[64,255,0],"46":[255,192,0],"50":[192,255,255],\
                  "43":[128,0,64],"53":[0,255,255],"12":[64,0,0],"13":[0,0,64],"16":[255,192,255]}
    if landmark_number not in color_code.keys():
        print(landmark_number)
        print("No code matching for the landmark_number",landmark_number)
        print("Please make matching color code.")
        return None
    if landmark_number == 1:
        insert_legend(Result_Image, color_code)
    mark_radius = 3
    for position in true_set:
        #Evaluated position sets = White
        status = 0
        for i in range(-1*mark_radius,mark_radius+1):
            for j in range(-1*mark_radius,mark_radius+1):
                x_position = int(position[0] + i)
                y_position = int(position[1] + j)
                mark_position = [x_position,y_position]
                mark_image(Result_Image,status,mark_position, color_code, landmark_number)

    #True landmark position = Blue
    status = 1
    for i in range(-1*mark_radius,mark_radius+1):
        for j in range(-1*mark_radius,mark_radius+1):
            x_position = int(true_landmark_position[0] + i)
            y_position = int(true_landmark_position[1] + j)
            mark_position = [x_position,y_position]
            mark_image(Result_Image,status,mark_position, color_code, landmark_number)

    #Evaluated landmark position = Red
    status = 2
    for i in range(-1*mark_radius,mark_radius+1):
        for j in range(-1*mark_radius,mark_radius+1):
            x_position = int(average_eval_position[0] + i)
            y_position = int(average_eval_position[1] + j)
            mark_position = [x_position,y_position]
            mark_image(Result_Image,status,mark_position, color_code, landmark_number)

    cv.imwrite(Result_picture_directory,Result_Image)

def draw_bayes_picture_score(Result_Image, Result_picture_directory, mod_set, true_landmark_position, bayes_eval_position):
    if len(mod_set) >= 1:
        for i in range(len(mod_set)):
            mark_image(Result_Image , 3, [mod_set[i][0], mod_set[i][1]], 0, 0, mod_set[i][4])
        mark_image(Result_Image, 1, [int(true_landmark_position[0]),int(true_landmark_position[1])], 0, 0)
        mark_image(Result_Image, 2, [int(bayes_eval_position[0]),int(bayes_eval_position[1])], 0, 0)
        cv.imwrite(Result_picture_directory,Result_Image)

def draw_bayes_picture_kde(Result_Image, Result_picture_directory, mod_set, true_landmark_position, bayes_eval_position, ellipse_data):
#def draw_bayes_picture_kde(Result_Image, Result_picture_directory, mod_set, true_landmark_position, bayes_eval_position, grid_position, probability_grid):
    #ellipse_data = [center, angle, width, height]
    center = ellipse_data[0]
    angle = ellipse_data[1]
    width = ellipse_data[2]
    height = ellipse_data[3]
    cv.ellipse(Result_Image, center, (width,height), angle, 0, 360, (0,69,255), 3)
    for i in range(len(mod_set)):
        mark_image(Result_Image, 4, [int(mod_set[i][0]), int(mod_set[i][1])], 0, 0)
    mark_image(Result_Image, 1, [int(true_landmark_position[0]),int(true_landmark_position[1])], 0, 0)
    mark_image(Result_Image, 2, [int(bayes_eval_position[0]),int(bayes_eval_position[1])], 0, 0)
    cv.imwrite(Result_picture_directory,Result_Image)

def draw_GT_picture(Result_Image, Result_picture_directory,  Evaluated_P_data, true_landmark_position, GT_positions):
    color_code = {"1":[0,64,0],"6":[0,255,64],"7":[255,0,64],"2":[255,64,0],"8":[128,64,128],"11":[64,0,64],"10":[0,64,64],\
                  "19":[128,64,64],"9":[64,64,128],"18":[0,64,128],"23":[64,0,128],"20":[64,255,0],"46":[255,192,0],"50":[192,255,255],\
                  "43":[128,0,64],"53":[0,255,255],"12":[64,0,0],"13":[0,0,64],"16":[255,192,255]}
    insert_legend(Result_Image, color_code)
    for i in range(len(Evaluated_P_data)):
        mark_image(Result_Image, 1, [int(true_landmark_position[i][0]),int(true_landmark_position[i][1])], 0, 0)
        mark_image(Result_Image, 0, [int(Evaluated_P_data[i][0][0]),int(Evaluated_P_data[i][0][1])], color_code, 1)
        mark_image(Result_Image, 0, [int(Evaluated_P_data[i][1][0]),int(Evaluated_P_data[i][1][1])], color_code, 6)
        mark_image(Result_Image, 0, [int(GT_positions[i][0]),int(GT_positions[i][1])], color_code, 7)
    cv.imwrite(Result_picture_directory,Result_Image)

def mark_image(Result_Image, status, mark_position, color_code, landmark_number, score = 0):
    height, width = Result_Image.shape[:2]
    if status == 0: #Status == Eval
        for i in range(-2,3):
            for j in range(-2,3):
                try:
                    Result_Image[mark_position[1]+i][mark_position[0]+j][0] = color_code[str(landmark_number)][0]
                    Result_Image[mark_position[1]+i][mark_position[0]+j][1] = color_code[str(landmark_number)][1]
                    Result_Image[mark_position[1]+i][mark_position[0]+j][2] = color_code[str(landmark_number)][2]
                except Exception:
                    pass
    if status == 1: #Status == True
        for i in range(-2,3):
            for j in range(-2,3):
                try:
                    Result_Image[mark_position[1]+i][mark_position[0]+j][0] = 255
                    Result_Image[mark_position[1]+i][mark_position[0]+j][1] = 255
                    Result_Image[mark_position[1]+i][mark_position[0]+j][2] = 255
                except Exception:
                    pass
    if status == 2: #Status == Averaged
        for i in range(-2,3):
            for j in range(-2,3):
                try:
                    Result_Image[mark_position[1]+i][mark_position[0]+j][0] = 0
                    Result_Image[mark_position[1]+i][mark_position[0]+j][1] = 0
                    Result_Image[mark_position[1]+i][mark_position[0]+j][2] = 0
                except Exception:
                    pass
    if status == 3: #Status == Bayes score plot
        for i in range(-1,2):
            for j in range(-1,2):
                if score < 0.25*MAX_SCORE:
                    score_B = 255
                    score_G = score/0.25*MAX_SCORE*255
                    score_R = 0
                elif score >= 0.25*MAX_SCORE and score <= 0.5*MAX_SCORE:
                    score_B = (0.25*MAX_SCORE - score)/0.25*MAX_SCORE*255 + 255
                    score_G = 255
                    score_R = 0
                elif score >= 0.5*MAX_SCORE and score <= 0.75*MAX_SCORE:
                    score_B = 0
                    score_G = 255
                    score_R = (score-0.5*MAX_SCORE)/0.25*MAX_SCORE*255
                elif score >= 0.75*MAX_SCORE and score <= MAX_SCORE:
                    score_B = 0
                    score_G = (0.75*MAX_SCORE - score)/0.25*MAX_SCORE*255 + 255
                    score_R = 255
                else:
                    score_B = 0
                    score_G = 0
                    score_R = 255
                try:
                    Result_Image[mark_position[1]+i][mark_position[0]+j][0] = ALPHA*Result_Image[mark_position[1]+i][mark_position[0]+j][0] + (1-ALPHA)*score_B
                    Result_Image[mark_position[1]+i][mark_position[0]+j][1] = ALPHA*Result_Image[mark_position[1]+i][mark_position[0]+j][1] + (1-ALPHA)*score_G
                    Result_Image[mark_position[1]+i][mark_position[0]+j][2] = ALPHA*Result_Image[mark_position[1]+i][mark_position[0]+j][2] + (1-ALPHA)*score_R
                except Exception:
                    pass
    if status == 4: #Status == max_softmax
        for i in range(-1,2):
            for j in range(-1,2):
                try:
                    Result_Image[mark_position[1]+i][mark_position[0]+j][0] = 0
                    Result_Image[mark_position[1]+i][mark_position[0]+j][1] = 255
                    Result_Image[mark_position[1]+i][mark_position[0]+j][2] = 0
                except Exception:
                    pass

def insert_legend(Result_Image, color_code):
    print("Legend Inserting...")
    for i in range(0,100):
        for j in range(0,30):
            for k in [0,5,6]:
                Result_Image[j+k*40+1650][1750+i][0] = color_code[str(k+1)][0]
                Result_Image[j+k*40+1650][1750+i][1] = color_code[str(k+1)][1]
                Result_Image[j+k*40+1650][1750+i][2] = color_code[str(k+1)][2]

def get_length(true_P_data, GT_positions):
    GTlength_data = []
    print(true_P_data)
    print(GT_positions)
    print("Length of true_P_data = " + str(len(true_P_data))+", Length of GT_position = " +str(len(GT_positions)))
    for i in range(len(GT_positions)):
        length_temp = math.sqrt((true_P_data[i][0] - GT_positions[i][0])**2 + (true_P_data[i][1] - GT_positions[i][1])**2)*0.1
        GTlength_data.append(length_temp)
    return GTlength_data

def write_GT_file(eval_log_directory, GT_length,true_P_data,GT_positions,length_data,Evaluated_P_data, U_2, target_P_data):
    with open(eval_log_directory,'w') as file:
        for i in range(len(GT_length)):
            file.write("Landmark: " + str(i+1))
            file.write("\n")
            file.write("KDE = ")
            file.write(str(length_data[i][0]))
            file.write("\n")
            file.write(" SCORE = ")
            file.write(str(length_data[i][1]))
            file.write("\n")
            file.write(" GT = ")
            file.write(str(GT_length[i]))
            file.write("\n")
            file.write("Seived_position: ")
            file.write(str(int(target_P_data[i][0])))
            file.write(" ")
            file.write(str(int(target_P_data[i][1])))
            file.write("\n")
            file.write(" KDE_position: ")
            file.write(str(int(Evaluated_P_data[i][0][0])))
            file.write(" ")
            file.write(str(int(Evaluated_P_data[i][0][1])))
            file.write("\n")
            file.write(" SCORE_position: ")
            file.write(str(int(Evaluated_P_data[i][1][0])))
            file.write(" ")
            file.write(str(int(Evaluated_P_data[i][1][1])))
            file.write("\n")
            file.write(" GT_position: ")
            file.write(str(int(GT_positions[i][0])))
            file.write(" ")
            file.write(str(int(GT_positions[i][1])))
            file.write("\n")
        file.write("U_2: ")
        file.write(str(U_2[0]))
        file.write("\n")
