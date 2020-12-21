import os
print(os.path.abspath(''))
import yaml as yml
import pandas as pd
import pickle
from modules.nn import dataPrep
from modules.nn.dataPrep import dataPrep
from modules.nn import classificationModel
from modules.nn.classificationModel import classificationModel
from itertools import chain, combinations
# from IPython.display import display, HTML
import json
import statistics as st
import time
from pathlib import Path
import datetime


start_time = time.time() # tracking the duration it takes the code to execute

ymlz = 'modules/nn/config_nn.yaml'

'''
            Check list for things that still needs to be done:
                #### check early stopping ---> Done
                #### batches keep it stagnant ---> Done
                #### grid search or random search ---> Not sure why

'''

def loadData():
    with open(ymlz) as file:
        CONF = yml.load(file, Loader=yml.FullLoader)
    if CONF['use_synth_data']:
        data_file_path = CONF['syncData']
    else:
        data_file_path = CONF['rawData']
    df = pd.read_csv(Path(data_file_path))
    df.dropna(inplace=True)
    return df,CONF

def time_string():
    t=datetime.datetime.now()
    return t.strftime("%Y%m%d_%H%M%S")

def sendToModel(df,CONF):
    if CONF['use_pickle']:
        filename = CONF['picklePath']
        infile = open(filename, 'rb')
        model = pickle.load(infile)
        model.plotGraphs()
        infile.close()
    else:
        iterationz = CONF['randGredNum']
        NN_data = dataPrep(df)  # Initializing the model
        dataz = NN_data.returnData()
        output=pd.DataFrame()
        dir_results = f"export/opt_{time_string()}"
        os.mkdir(dir_results)
        for i in range(iterationz):
            pickle_path = f"{dir_results}/model_{str(i)}"
            model = classificationModel(dataz,CONF)
            if CONF['rawDataAnalysis']:
                model.correl() # This was used in order to assess the correlation between the features
                model.correlNormalized () # This was used in order to assess the correlation between the features

            candidate = model.orcastrator(i,dir_results)
            output=output.append(pd.DataFrame.from_dict(candidate, orient='index').T,ignore_index=True)

            #saving model
            outfile = open(pickle_path, 'wb')
            pickle.dump(model, outfile)
            outfile.close()
        output.to_csv(f"{dir_results}/models_stats.csv")

    return model

def printToScreenAdjustments():
    CSS = """
    .output {
        align-items: left;
    }
    """

    # HTML('<style>{}</style>'.format(CSS))

    pd.set_option('display.max_rows', None)
    pd.options.display.width = 0
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print('Display modified successfully','\n')


def main():
    print(os.getcwd())
    printToScreenAdjustments()                      # Adjusting dataframe print to screen
    df,CONF = loadData()                            # Loading model's data and Yaml configuration
    sendToModel(df, CONF)     # Sending to model and receiving results and updated Configuration JSON file
    print("TOTAL RUN TIME:", '{:.2f}'.format((time.time() - start_time) / 60), "minutes.")


if __name__=='__main__':
    main()