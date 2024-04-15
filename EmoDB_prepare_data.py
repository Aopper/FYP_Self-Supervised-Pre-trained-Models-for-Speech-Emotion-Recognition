import os
from collections import defaultdict
import pandas as pd
import torchaudio

# Assuming `data_dir` is the directory containing your .wav files
data_dir = '/home/felix/Aopp/0.0/EmoDB/wav'
out_path = '/home/felix/Aopp/FINAL/data/EmoDB'
files = os.listdir(data_dir)

# Organize files by speaker
data_by_speaker = defaultdict(list)
for file in files:
    if file.endswith('.wav'):
        speaker_id = file[:2]  # Extract the speaker ID
        data_by_speaker[speaker_id].append(file)

for test_speaker in data_by_speaker.keys():

    test_files = data_by_speaker[test_speaker]
    
    # if len(test_files) <= 55:
    #     continue

    train_files = []
    for speaker_id, files in data_by_speaker.items():
        if speaker_id != test_speaker:
            train_files.extend(files)
    
    
    # print(f"Training with all speakers except {test_speaker}. Total training files: {len(train_files)}")
    # print(f"Testing with speaker {test_speaker}. Total testing files: {len(test_files)}")


    # lable_dic = {"W": 'anger', "L": 'boredom', "E": 'disgust', "A": 'anxiety/fear', "F": 'happiness', "T": 'sadness', "N": 'neutral version'}
    lable_dic = {"W": 'NH', "L": 'NL', "E": 'NH', "A": 'NH', "F": 'P', "T": 'NL', "N": 'Ne'}

    train_list, test_list = [],[]

    for file in train_files:
        try:
            torchaudio.load(os.path.join(data_dir, file))
            train_list.append(
                {
                    "name": file.split('.')[0],
                    "path": os.path.join(data_dir,file),
                    "emotion": lable_dic[file[5]]        
                }
            )
        except Exception as e:
            print(os.path.join(data_dir, file), e)
            pass

    for file in test_files:
        try:
            torchaudio.load(os.path.join(data_dir, file))
            test_list.append(
                {
                    "name": file.split('.')[0],
                    "path": os.path.join(data_dir,file),
                    "emotion": lable_dic[file[5]]        
                }
            )
        except Exception as e:
            print(os.path.join(data_dir, file), e)
            pass

        
    train_df = pd.DataFrame(train_list)
    test_df = pd.DataFrame(test_list)

    print("For test speaker id {}, {} train, {} test".format(test_speaker, len(train_df), len(test_df)))

    folder_path = os.path.join(out_path, test_speaker)
    # print("folder_path: ",folder_path)
    # if not os.path.exists(folder_path):
    #     os.makedirs(folder_path)

    # print("For tain data")
    # print(train_df.groupby("emotion").count()[["path"]])

    # print("------------------------------------")
    # print("For test data")
    # print(test_df.groupby("emotion").count()[["path"]])

    # print("----------------------------------------------------------------------")
    train_df.to_csv(f"{folder_path}/train.csv", sep="\t", encoding="utf-8", index=False)
    test_df.to_csv(f"{folder_path}/test.csv", sep="\t", encoding="utf-8", index=False)