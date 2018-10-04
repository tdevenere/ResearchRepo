import os
import pandas as pd
import random

def main():
    path = os.path.join(os.getcwd(), 'Users_time_aligned')
    trainp = os.path.join(os.getcwd(), 'RNN-Train-Aligned')
    testp = os.path.join(os.getcwd(), 'RNN-Test-Aligned')
    files = os.listdir(path)
    for file in files:
        extension = file[-4:]
        if extension == '.csv':
            name = file[:-4]
            filepath = os.path.join(path, file)
            df = pd.read_csv(filepath)

            #train = pd.DataFrame()
            trains = []
            tests = []
            #test = pd.DataFrame()

            for i in range(1,9):
                dir_i = df.loc[df['Direction'] == i]
                sample_vals = set(dir_i['Sample'])
                for s in sample_vals:
                    curr_sample = dir_i.loc[dir_i['Sample'] == s]
                    r = random.random()
                    if r < 0.5:
                        tests.append(curr_sample)
                    else:
                        trains.append(curr_sample)


            train = pd.concat(trains)
            test = pd.concat(tests)

            trainname = name + '-train.csv'
            trainpath = os.path.join(trainp, trainname)
            train.to_csv(trainpath, index=False)
            testname = name + '-test.csv'
            testpath = os.path.join(testp, testname)
            test.to_csv(testpath, index=False)







if __name__ == '__main__':
    main()