import csv 
import random
import time
import numpy as np
import argparse

fieldnames = ["subject", "ppg", "ecg", "abp"]
data_path = '../data/'

x = np.load(data_path+'x_test.npy', allow_pickle=True)
y = np.load(data_path+'y_test.npy', allow_pickle=True)

parser = argparse.ArgumentParser(description='Subject to gen')
parser.add_argument('-s', type=int, dest='subject', default=0,
                    help='Subject Number')

args = parser.parse_args()

print(args.subject)

with open('data.csv', 'w') as csv_file:
    csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    csv_writer.writeheader()

for index in range(y[args.subject].shape[0]):
    ppg = x[args.subject][0][index]
    ecg = x[args.subject][1][index]
    abp = y[args.subject][index]
    with open('data.csv', 'a') as csv_file:
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        info = {
            "subject": args.subject+1,
            "ppg": ppg,
            "ecg": ecg,
            "abp": abp
        }

        csv_writer.writerow(info)
        print(args.subject+1, ppg, ecg, abp)

    time.sleep(0.1)
print('Done...')