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



if __name__=='__main__':
    evalute_hyper_random_search()
