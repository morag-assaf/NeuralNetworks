import bayes

class dataloader_classification(object):
    def __init__(self,df,target):
        self.X=df[df.columns!=target]
        self.Y=df[target]

    def load_data(self):

    def preprocess(self):
        #for example, use the bayes model discretization, scaling, binarization

    def split_train_test(self):

    def weight_classification(self):
        #if we use this approach to solve the non balance data


