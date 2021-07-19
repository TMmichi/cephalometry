from trainmodel import *
from landmark_data import *
import math
import numpy as np
import xlrd
import os

####################################################################
#CONSTANTS
PRED_REF_DIAMETER = 60 #MM
BAYES_ITER = 40
THRESHOLD_PROB = 0.0
TRAINING_IMAGE_START = 1
TRAINING_IMAGE_END = 150
EVAL_START = 152
EVAL_END = 152
data_type = "mean"
training_step = 300000

#SWITCH
TRAINING = True
EVALUATING = False
PREDICTING = False
Eval_Preimage = False
isISBI = True

#FOLDER DIRECTORY
Model_directory = "/home/ljh/Projects/cephalometry/model/"
if isISBI:
    model_seive_name = "ModelISBI_seive18"
    model_name = "ModelISBI_mean"
else:
    model_seive_name = "ModelEWHA_seive18"
    model_name = "ModelEWHA_mean"
Eval_log_folder_directory = "/home/ljh/Projects/data/Eval_log_data/ISBI_mean_seived_thr_0.0_it40"
Pred_folder_directory = "/home/ljh/Projects/data/Pred_data"
####################################################################


def main(_):
    ###################################################################
    #TRAINING
    ###################################################################
    TRAIN_IMAGE_NUMBER = TOTAL_SAMPLE_NUMBER * int(PRED_INIT_NUMBER-1)
    if TRAINING:
        for landmark_number in LANDMARK_CONSIDERATION:
            #Estimator Setup
            run_config = tf.estimator.RunConfig()
            config = tf.ConfigProto(log_device_placement=True)
            config.gpu_options.visible_device_list = '3'
            config.gpu_options.allow_growth = True
            run_config = run_config.replace(save_checkpoints_steps=10000, keep_checkpoint_max=None, session_config=config, save_summary_steps=5000)
            if not isSeive:
                model_dir = Model_directory + model_name + "/model_LM" + str(landmark_number)
            else:
                model_dir = Model_directory + model_seive_name + "/model_LM" + str(landmark_number)
         
            landmark_classifier = tf.estimator.Estimator(
                model_fn=modeling, model_dir=model_dir,config=run_config)

            ##Input data importing
            start_num = TRAINING_IMAGE_START
            end_num = TRAINING_IMAGE_END
            input_images, input_labels = bring_inputset(Training_data_folder_directory,start_num,end_num,landmark_number, FALSE_SAMPLE_NUMBER, TRUE_SAMPLE_NUMBER)

            #Training input formation
            train_images = input_images[:TRAIN_IMAGE_NUMBER]
            train_labels = np.asarray(input_labels[:TRAIN_IMAGE_NUMBER], dtype=np.int32)
            print("Training datasets imported")

            #Training Model input Setting
            train_input_fn = tf.estimator.inputs.numpy_input_fn(
                x={"x": train_images},
                y=train_labels,
                batch_size=128,
                num_epochs=None,
                shuffle=True)

            steps=training_step
            #Model training
            landmark_classifier.train(
              input_fn=train_input_fn,
              steps=steps)


###################################################################
#EVALUATING
###################################################################

    if EVALUATING:
        #for image_index in range(1060, 1061+1):
        for image_index in range(EVAL_START,EVAL_END+1):
            time_image_init = datetime.datetime.now()
            P_data = []
            E_data = []
            true_P_data = []
            target_P_data = []
            Evaluated_P_data = []
            length_data = []
            for landmark_number in LANDMARK_CONSIDERATION:
                landmark_index = LANDMARK_CONSIDERATION.index(landmark_number)
                time_landmark_init = datetime.datetime.now()
                print("Evaluation for image "+str(image_index)+", landmark "+str(landmark_number))

                #Target Landmark position seiving
                ###################################################################
                run_config = tf.estimator.RunConfig()
                config = tf.ConfigProto(device_count={'GPU':4})
                config.gpu_options.allow_growth = True
                run_config = run_config.replace(session_config=config)
                seive_model_dir = Model_directory + model_seive_name + "/model_LM" + str(landmark_number)
                landmark_classifier = tf.estimator.Estimator(
                    model_fn=modeling, model_dir=seive_model_dir, config=run_config)
                #Correspoding evaluation set importing
                seive_images, seive_position = bring_seive_make(image_index, landmark_number, Raw_data_folder_directory, DOWNSAMPLEFACTOR, IMAGE_SIZE, isISBI)
                print(len(seive_images))
                #Importing True landmark position from the pred data
                if not isISBI:
                    seive_data_directory = Raw_data_folder_directory+"/"+str(image_index)+".xlsx"
                else:
                    index_temp = format(image_index,"03d")
                    seive_data_directory = Raw_data_folder_directory+"/"+str(index_temp)+".txt"
                #Raw [X,Y]
                true_landmark_position = get_True_position(seive_data_directory, landmark_number, isISBI, landmark_index)
                true_P_data.append(true_landmark_position)
                #Input Evaluating
                seive_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": seive_images},
                    num_epochs=1,
                    shuffle=False)
                seive_model = list(landmark_classifier.predict(input_fn=seive_input_fn))
                evaluated_classes = [p["classes"] for p in seive_model]
                evaluated_softmax = [p["probabilities"][1] for p in seive_model]
                #Evaluated position extracting
                _, _, sum_position, num_true_image = position_process(seive_position, evaluated_classes, DOWNSAMPLEFACTOR, evaluated_softmax)
                #Predicted position averaging
                target_landmark_position, _ = position_averaging(num_true_image,sum_position,true_landmark_position)
                target_P_data.append(target_landmark_position)
                print(target_landmark_position)
                target_landmark_position_mod = [int(target_landmark_position[1]/DOWNSAMPLEFACTOR), int(target_landmark_position[0]/DOWNSAMPLEFACTOR)]
                print(true_landmark_position)
                ###################################################################

                #Correspoding evaluation set importing
                eval_images, eval_position = bring_evalset_make(image_index, landmark_number, Raw_data_folder_directory, EVAL_REF_DIAMETER, DOWNSAMPLEFACTOR, target_landmark_position_mod, IMAGE_SIZE, isISBI)
                print("Importing eval datasets completed\n")

                #Model setup for Evaluation
                model_dir=Model_directory+model_name+"/model_LM"+str(landmark_number)
                model_function = modeling
                landmark_classifier = tf.estimator.Estimator(
                    model_fn=model_function,
                    model_dir=model_dir,
                    config=run_config)

                #Input Evaluating
                eval_input_fn = tf.estimator.inputs.numpy_input_fn(
                    x={"x": eval_images},
                    num_epochs=1,
                    shuffle=False)

                eval_softmax_set = []
                eval_classes_set = []
                for iter in range(0,BAYES_ITER):
                    print("\n")
                    print("Evaluation iter = ",iter+1)
                    print("Image number = ", image_index)
                    eval_model = list(landmark_classifier.predict(input_fn=eval_input_fn))
                    evaluated_softmax = [p["probabilities"][1] for p in eval_model]
                    evaluated_class = [p["classes"] for p in eval_model]
                    eval_softmax_set.append(np.asarray(evaluated_softmax))
                    eval_classes_set.append(np.asarray(evaluated_class))

                #Evaluation result logging
                ###################################################################

                #Evaluated position extracting
                eval_set = bayes_prob_averaging(eval_softmax_set, BAYES_ITER)
                bayes_eval_position_kde, position_candidate_kde, ellipse_data, index_max = bayes_position_process_kde(eval_softmax_set, eval_position, DOWNSAMPLEFACTOR, Eval_log_folder_directory, image_index, landmark_number)
                bayes_eval_position_score, position_candidate_score, score_max = bayes_position_process_score(THRESHOLD_PROB, eval_position, DOWNSAMPLEFACTOR, eval_set[0], eval_set[1], index_max)

                P_data.append(position_candidate_kde)
                E_data.append(score_max)
                Evaluated_P_data.append([bayes_eval_position_kde, bayes_eval_position_score])

                #Write to a file
                log_folder_directory = Eval_log_folder_directory + "/Image"+str(image_index)
                try:
                    os.makedirs(log_folder_directory)
                except OSError:
                    pass
                eval_log_directory = log_folder_directory+"/Bayes_eval_result image"+str(image_index)+" LM"+str(landmark_number)+".txt"
                length_kde, length_score = write_log_bayes_eval_file(eval_log_directory, BAYES_ITER, eval_set, eval_position, DOWNSAMPLEFACTOR, eval_classes_set, eval_softmax_set, true_landmark_position, bayes_eval_position_score, bayes_eval_position_kde, position_candidate_score)
                length_data.append([length_kde, length_score])

                #Plot a score Graph
                Result_picture_directory_kde = Eval_log_folder_directory+"/Image"+str(image_index)+"_Bayes_Result_kde.jpg"
                Result_picture_directory_kde_each = Eval_log_folder_directory+"/Image"+str(image_index)+"landmark"+str(landmark_number)+"_Bayes_Result_kde.jpg"
                Result_picture_directory_score = Eval_log_folder_directory+"/Image"+str(image_index)+"_Bayes_Result_score.jpg"
                if os.path.isfile(Result_picture_directory_kde):
                    Result_Image_kde = cv.imread(Result_picture_directory_kde,1)
                else:
                    if not isISBI:
                        Result_Image_kde = cv.imread(Raw_data_folder_directory+"/"+str(image_index)+".jpg",1)
                    else:
                        Result_Image_kde = cv.imread(Raw_data_folder_directory+"/"+str(image_index)+".bmp",1)
                if os.path.isfile(Result_picture_directory_score):
                        Result_Image_score = cv.imread(Result_picture_directory_score,1)
                else:
                    if not isISBI:
                        Result_Image_score = cv.imread(Raw_data_folder_directory+"/"+str(image_index)+".jpg",1)
                    else:
                        Result_Image_score = cv.imread(Raw_data_folder_directory+"/"+str(image_index)+".bmp",1)
                Result_Image_kde_each = cv.imread(Raw_data_folder_directory+"/"+str(image_index)+".bmp",1)
                draw_bayes_picture_kde(Result_Image_kde, Result_picture_directory_kde, position_candidate_kde, true_landmark_position, bayes_eval_position_kde, ellipse_data)
                draw_bayes_picture_kde(Result_Image_kde_each, Result_picture_directory_kde_each, position_candidate_kde, true_landmark_position, bayes_eval_position_kde, ellipse_data)
                draw_bayes_picture_score(Result_Image_score, Result_picture_directory_score, position_candidate_score, true_landmark_position, bayes_eval_position_score)
                time_landmark_end = datetime.datetime.now()
                print("Landmark calculation time: ",time_landmark_end - time_landmark_init)
                file = open("Time required_image" + str(image_index) +".txt",'a')
                file.write("Required time for image"+str(image_index)+", landmark"+str(landmark_number)+" = "+str(time_landmark_end - time_landmark_init))
                file.write("\n")
                file.close()

            #process = GTF(E_data, P_data)
            #GT_positions, U_2 = process.Postprocess()
            #GT_length = get_length(true_P_data, GT_positions)
            GT_positions = []
            for i in range(19):
                GT_positions.append([0,0])
            GT_length = np.zeros(19)
            U_2 = [0]
            #Result_Image = cv.imread(Raw_data_folder_directory+"/"+str(image_index)+".bmp",1)
            #Result_picture_directory = Eval_log_folder_directory+"/Image"+str(image_index)+"_Bayes_Result_overall.jpg"
            #draw_GT_picture(Result_Image, Result_picture_directory,  Evaluated_P_data, true_P_data, GT_positions)
            eval_log_directory = log_folder_directory+"/Bayes_eval_result image"+str(image_index)+"_Overall.txt"
            write_GT_file(eval_log_directory, GT_length,true_P_data,GT_positions,length_data,Evaluated_P_data, U_2, target_P_data)

            time_image_end = datetime.datetime.now()
            print("Image calculation time: ",time_image_end - time_image_init)
            file = open("Time required_image" + str(image_index) +".txt",'a')
            file.write("Required time for image"+str(image_index)+": "+str(time_image_end - time_image_init))
            file.write("\n")
            file.close()

###################################################################
#PREDICTING
###################################################################

    if PREDICTING:

        #Target Landmark position seiving
        ###################################################################
        landmark_classifier = tf.estimator.Estimator(
            model_fn=modeling, model_dir="ModelISBI_seive/model_LM"+str(landmark_number))
        #Correspoding evaluation set importing
        eval_images, eval_position = bring_seive_make(image_index, landmark_number, Raw_data_folder_directory, DOWNSAMPLEFACTOR, IMAGE_SIZE, isISBI)
        #Input Evaluating
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_images},
            num_epochs=1,
            shuffle=False)
        eval_model = list(landmark_classifier.predict(input_fn=eval_input_fn))
        evaluated_classes = [p["classes"] for p in eval_model]
        evaluated_softmax = [p["probabilities"][1] for p in eval_model]
        #Evaluated position extracting
        eval_set, true_set, sum_position, num_true_image = position_process(eval_position, evaluated_classes, DOWNSAMPLEFACTOR, evaluated_softmax)
        #Predicted position averaging
        target_landmark_position, _ = position_averaging(num_true_image,sum_position,true_landmark_position)
        target_landmark_position = [int(target_landmark_position[0]), int(target_landmark_position[1])]
        ###################################################################

        #Prediction Model input Setting
        pred_images, pred_position = bring_predset(Pred_folder_directory+"/pred.jpg", target_landmark_position, PRED_REF_DIAMETER, DOWNSAMPLEFACTOR, IMAGE_SIZE, landmark_number)
        print("Pred datasets imported")

        #Input Predicting
        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": pred_images},
            num_epochs=1,
            shuffle=False)

        #Input Predicting
        pred_softmax_set = []
        for iter in range(0,BAYES_ITER):
            print("Prediciton iter = ",iter+1)
            predict_model = list(landmark_classifier.predict(input_fn=predict_input_fn))
            predicted_softmax = [p["probabilities"][1] for p in predict_model]
            print(predicted_softmax)
            pred_softmax_set.append(np.asarray(predicted_softmax))

        #Prediction result logging
        ###################################################################

        #Predicted position extracting
        prob_set = bayes_prob_averaging(pred_softmax_set, BAYES_ITER)

        #Write to a file
        pred_log_directory = Pred_folder_directory+"/Bayes_pred_result LM"+str(landmark_number)+".txt"
        write_log_bayes_pred_file(pred_log_directory, prob_set, pred_position)

if __name__ == "__main__":
    tf.app.run()
