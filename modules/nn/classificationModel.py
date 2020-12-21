from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.layers import Dropout
from keras.metrics import AUC
import datetime
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle
from sklearn.metrics import roc_auc_score,roc_curve


def nested_dict_to_df(dict_x):
    output = pd.DataFrame()
    for k, v in dict_x.items():
        if type(v) == dict:
            df_i = pd.DataFrame.from_dict(v, orient='index').T
            df_i.columns = k + '_' + df_i.columns
            output = pd.concat([output, df_i], axis=1)
        else:
            output[k] = v
    return output

def time_string():
    t=datetime.datetime.now()
    return t.strftime("%Y%m%d_%H%M%S")

def evalute_hyper_random_search():
    number_of_users=2000
    number_of_items=3000
    ## ---- get data ---- ##
    rd = recommender_data(conf['train_file'], conf['validation_file'], conf['test_file'])
    tr_user, tr_items, tr_rating, val_rating = \
        rd.training_validation_split(number_of_users=number_of_users, number_of_items=number_of_items, split_quant=0.2,
                                     selection_type='random')

    output = pd.DataFrame()

    ## ---- loop over candidates---- ##
    for i in range(20):
        candidate=get_candidate(conf)
        model = recommender(R=tr_rating, train_ind=(tr_user, tr_items), V=val_rating, val_ind=(tr_user, tr_items),
                            latent_dim=candidate['latent_dim'],
                            epochs=60,
                            lrs=candidate['lrs'],
                            regularizers=candidate['regularizers'],
                            optimizer='sgd')
        start_run = time.time()
        m=model.fit()
        end_run = time.time()

        ## ---- collecting results--- ##
        candidate['model'] = i
        candidate['naive_error']=model.naive_error()
        candidate['min_MSE'] = min(model.loss_curve['validation'])
        candidate['best_epoch'] = np.argmin(model.loss_curve['validation'])
        candidate['% improvement']=1- candidate['min_MSE']/candidate['naive_error']
        candidate['elpased']=str(datetime.timedelta(seconds=end_run-start_run))
        fig=model.plot_learning_curve()
        plt.savefig(f'export/model_{i}_{time_string()}.png')

        ## ---- saving run results --- ##
        output=output.append(nested_dict_to_df(candidate))

    #printing to file
    output.to_csv(f'export/u_{number_of_users}_i_{number_of_items}_{time_string()}.csv')

def auroc(y_true, y_pred):
    return tf.py_func(roc_auc_score, (y_true, y_pred), tf.double)

class classificationModel:
    def __init__(self,dataz,CONF):
        self.X_train = dataz[0]
        self.X_val = dataz[1]
        self.X_test = dataz[2]
        self.Y_train = dataz[3]
        self.Y_val = dataz[4]
        self.Y_test = dataz[5]
        self.CONF = CONF
        self.loss = []
        self.accuracy = []
        self.valLoss = []
        self.valacc = []
        self.lr = []
        self.evaluation = []
        self.iteration = 0
        self.elapsed = 0

    def NN_model(self,inputNeurons=16,hiddenLayers=1,neuronLayers=8):
        if hiddenLayers == 1:
            NN_model = Sequential([
                Dense(inputNeurons, input_dim=14, activation='relu')
                , Dropout(0.1)
                , Dense(neuronLayers, activation='relu')
                , Dropout(0.1)
                , Dense(1, activation='sigmoid')
            ])
        elif hiddenLayers == 2:
            NN_model = Sequential([
                Dense(inputNeurons, input_dim=14, activation='relu')
                , Dropout(0.1)
                , Dense(neuronLayers, activation='relu')
                , Dropout(0.1)
                , Dense(neuronLayers, activation='relu')
                , Dropout(0.1)
                , Dense(1, activation='sigmoid')
            ])
        elif hiddenLayers == 3:
            NN_model = Sequential([
                Dense(inputNeurons, input_dim=14, activation='relu')
                , Dropout(0.1)
                , Dense(neuronLayers, activation='relu')
                , Dropout(0.1)
                , Dense(neuronLayers, activation='relu')
                , Dropout(0.1)
                , Dense(neuronLayers, activation='relu')
                , Dropout(0.1)
                , Dense(1, activation='sigmoid')
            ])
        elif hiddenLayers == 4:
            NN_model = Sequential([
                Dense(inputNeurons, input_dim=14, activation='relu')
                , Dropout(0.1)
                , Dense(neuronLayers, activation='relu')
                , Dropout(0.1)
                , Dense(neuronLayers, activation='relu')
                , Dropout(0.1)
                , Dense(neuronLayers, activation='relu')
                , Dropout(0.1)
                , Dense(neuronLayers, activation='relu')
                , Dropout(0.1)
                , Dense(1, activation='sigmoid')
            ])
        return NN_model

    def select_random_option(self,x):
        i = np.random.randint(0, len(x))
        return x[i]

    def get_candidate(self,CONF):
        candidate = {}
        for param_type in CONF['candidate'].keys():
            if type(CONF['candidate'][param_type]) == dict:
                param_sub = {}
                for sub_val in CONF[param_type].keys():
                    x = self.select_random_option(CONF['candidate'][param_type])
                    param_sub[sub_val] = x
                candidate[param_type] = param_sub
            else:
                x = self.select_random_option(CONF['candidate'][param_type])
                candidate[param_type] = x
        return candidate

    def step_decay_schedule(self,initial_lr=1e-3, decay_factor=0.75, step_size=10):
        '''
        Wrapper function to create a LearningRateScheduler with step decay schedule.
        '''

        def schedule(epoch):
            return initial_lr * (decay_factor ** np.floor(epoch / step_size))
        return LearningRateScheduler(schedule)

    def fit(self,NN_model,batchz,initial_lr,max_epochs):
        auc = AUC(name='auc')
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, baseline=0.4)
        lr_sched = self.step_decay_schedule(initial_lr=initial_lr, decay_factor=0.75, step_size=2)

        NN_model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy',auc])

        NN_results = NN_model.fit(self.X_train, self.Y_train, epochs=max_epochs, batch_size=batchz,
                                  validation_data=(self.X_val,self.Y_val.squeeze()),callbacks=[es,lr_sched],
                                  shuffle=True, verbose=1)
        return NN_results

    def evaluate(self,NN_model,batchz=128):
        print("Evaluate on test data")
        results = NN_model.evaluate(self.X_test, self.Y_test, batch_size=batchz)
        print("test loss, test acc:", results)
        return results

    # def decider(self,NewCONF,optLoss,optAcc,historyz,evaluateResultsLoss,evaluateResultsAcc,LossERR,AccERR,numzi,inputzi,nueronzi,batchi):
    #     if NewCONF['optimal_nn']['optLoss'] > optLoss:
    #         NewCONF["optimal_nn"]["hidden_layers"] = numzi
    #         NewCONF["optimal_nn"]["inputNeurons"] = inputzi
    #         NewCONF["optimal_nn"]["hiddenLayerNeurons"] = nueronzi
    #         NewCONF["optimal_nn"]["batches"] = batchi
    #         NewCONF["optimal_nn"]["optLoss"] = optLoss
    #         NewCONF["optimal_nn"]["optAcc"] = optAcc
    #         NewCONF["optimal_nn"]["historyz"] = historyz
    #         NewCONF["optimal_nn"]["evaluateResultsLoss"] = evaluateResultsLoss
    #         NewCONF["optimal_nn"]["evaluateResultsAcc"] = evaluateResultsAcc
    #         NewCONF["optimal_nn"]["lossERR"] = LossERR
    #         NewCONF["optimal_nn"]["AccERR"] = AccERR
    #     return NewCONF

    ############# Still need to work on predictor
    # def predict(self):
    #     print("Generate predictions for 3 samples")
    #     predictions = model.predict(x_test[:3])
    #     print("predictions shape:", predictions.shape)

    def plotGraphs2(self):
        CONF = self.CONF
        if CONF["plot_graphs"]:
            figs, axs = plt.subplots(2, 2, figsize=(15, 10))
            n_epoch=len(self.valLoss)

            #-- plotting loss---#
            axs[0, 0].plot(range(n_epoch), self.loss, color='blue', label='Training Loss')
            axs[0, 0].plot(range(n_epoch), self.valLoss, color='orange', label='Validation Loss')
            axs[0, 0].legend()

            # -- plotting accuracy--#
            axs[0, 1].plot(range(n_epoch), self.accuracy, color='blue', label='Training Accuracy')
            axs[0, 1].plot(range(n_epoch), self.valacc , color='orange', label='Validation Accuracy')
            axs[0, 1].legend()

            # -- plotting roc val--#
            fpr, tpr, thresholds = self.val_roc_info
            axs[1, 0].plot(fpr, tpr, lw=1, color='orange', label=f'validation ROC')
            axs[1, 0].set_xlabel('False Positive Rate')
            axs[1, 0].set_ylabel('True Positive Rate (Recall)')
            axs[1, 0].legend()

            # -- plotting roc val--#
            fpr, tpr, thresholds = self.test_roc_info
            axs[1, 1].plot(fpr, tpr, lw=1, color='red', label=f'test ROC')
            axs[1, 1].set_xlabel('False Positive Rate')
            axs[1, 1].set_ylabel('True Positive Rate (Recall)')
            axs[1, 1].legend()

        return axs

    def plotGraphs(self):
        CONF = self.CONF
        if CONF["plot_graphs"]:
            self.plotNN_Accuracy()
            self.plotNN_Loss()

    def plotNN_Loss(self):
        loss = self.loss
        val_loss = self.valLoss
        plt.plot(loss)
        plt.plot(val_loss)
        plt.title('Model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        #plt.show()


    def plotNN_Accuracy(self):
        acc = self.accuracy
        valAcc = self.valacc
        plt.plot(acc)
        plt.plot(valAcc)
        plt.title('Model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Test'], loc='upper left')
        #plt.show()


    def orcastrator(self,i,dir_results):

        #create candidate
        batchez = self.CONF['batchez']
        candidate = self.get_candidate(self.CONF)
        start_run = time.time()

        #Running the model
        NN_model = self.NN_model(candidate['inputNeuron'],candidate['numberHiddenLayers'],candidate['hiddenLayerNeurons'])
        NN_results = self.fit(NN_model, batchez, candidate['lr'],max_epochs=self.CONF['max_epochs'])
        evaluateResults = self.evaluate(NN_model)
        end_run = time.time()


        ## ---- collecting results--- ##
        self.test_roc_info = roc_curve(self.Y_test,
                                         NN_model.predict(self.X_test), pos_label=1)
        self.val_roc_info = roc_curve(self.Y_val,
                                       NN_model.predict(self.X_val), pos_label=1)

        self.iteration = i
        self.evaluation = evaluateResults
        self.loss = NN_results.history['loss']
        self.accuracy = NN_results.history['accuracy']
        self.valLoss = NN_results.history['val_loss']
        self.valacc = NN_results.history['val_accuracy']
        self.lr = NN_results.history['lr']
        self.elapsed = str(datetime.timedelta(seconds=end_run - start_run))

        candidate['best_training_accuracy']=max(NN_results.history['accuracy'])
        candidate['best_validation_accuracy'] = max(NN_results.history['val_accuracy'])
        candidate['best_val_auc']= max(NN_results.history['val_auc'])
        candidate['best_epoch_auc']=np.argmax(NN_results.history['val_auc'])+1
        candidate['test_auc'] = evaluateResults[2]

        self.plotGraphs2()  # Plotting the result
        plt.savefig(f'{dir_results}/model_{i}_{time_string()}.png')

        return candidate

        # numberHideenLayers = self.CONF['hyperparameters']['numberHiddenLayers']
        # inputNeuron = self.CONF['hyperparameters']['inputNeuron']
        # hiddenLayerNeurons = self.CONF['hyperparameters']['hiddenLayerNeurons']
        # initial_lr = self.CONF['hyperparameters']['lr']
        # lenz = len(numberHideenLayers) * len(inputNeuron) * len(hiddenLayerNeurons) * len(initial_lr)
        # print('\n','\n',"There are ",lenz," conditions, estimates run time is ",'{:.2f}'.format(lenz * 0.5)," minutes")

        # NeuronsInput = 1600
        # NumLayers = 2
        # NeuronsHidden = 800
        # batchez = 128
        # initial_lr = initial_lr[1]

        # NN_model = self.NN_model(NeuronsInput, NumLayers, NeuronsHidden)
        # NN_results = self.fit(NN_model,batchez,initial_lr)
        # evaluateResults = self.evaluate(NN_model)

        # self.evaluation = evaluateResults
        # self.loss = NN_results.history['loss']
        # self.accuracy = NN_results.history['accuracy']
        # self.valLoss = NN_results.history['val_loss']
        # self.valacc = NN_results.history['val_accuracy']
        # self.lr = NN_results.history['lr']
        # self.plotGraphs()  # Plotting the result


        ### Addign the sensitivity dimension
        # for NumLayers in numberHideenLayers:
        #     for NeuronsInput in inputNeuron:
        #         for NeuronsHidden in hiddenLayerNeurons:
        #             i += 1
        #             NN_model = self.NN_model(NeuronsInput, NumLayers, NeuronsHidden)
        #             NN_results = self.fit(NN_model)
        #             evaluateResults = self.evaluate(NN_model)
        #             optLoss = NN_results.history['loss'][-1]
        #             optAcc = NN_results.history['accuracy'][-1]
        #             historyz = NN_results.history
        #             evaluateResultsLoss = evaluateResults[0]
        #             evaluateResultsAcc = evaluateResults[1]
        #             LossERR = abs((optLoss - evaluateResultsLoss))
        #             AccERR = abs((evaluateResultsAcc - optAcc))
        #             NewCONF = self.decider(NewCONF,optLoss, optAcc, historyz, evaluateResultsLoss, evaluateResultsAcc,
        #                          LossERR, AccERR,NumLayers,NeuronsInput,NeuronsHidden)
        #             print("This is iteration number: ",i," out of ",lenz,'\n',"CURRENT RUN TIME:", '{:.2f}'.format((time.time() - start_time) / 60), "minutes.")
        # return NN_results


if __name__=='__main__':
    evalute_hyper_random_search()