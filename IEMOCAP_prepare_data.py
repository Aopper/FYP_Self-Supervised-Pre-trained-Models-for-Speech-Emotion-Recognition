import pandas as pd
import os


label_fusion = {
    # 'hap': 'Happy',
    'exc': 'Excited',
    # 'fru': 'Negative',
    'ang': 'Anger',
    # 'dis': 'Negative',
    'sad': 'Sad',
    # 'fea': 'Sad/Fear',
    'neu': 'Neutral',
    # 'sur': 'Neutral/Surprise',
}


# Load the dataset
df = pd.read_csv('/home/felix/Aopp/0.0/IEMOCAP/iemocap/iemocap_full_dataset.csv')
df['emotion'] = df['emotion'].map(label_fusion)

df = df.dropna(subset=['emotion'])

base_output_dir = '/home/felix/Aopp/FINAL/data/IEMOCAP/lablefusing_video'
os.makedirs(base_output_dir, exist_ok=True)

# There are 5 sessions, so we loop from 1 to 5
for test_session in range(1, 6):
    # Create a specific directory for the current split
    output_dir = os.path.join(base_output_dir, f'test{test_session}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Splitting the DataFrame based on the session number
    test_df = df[df['session'] == test_session]
    train_df = df[df['session'] != test_session]
    
    # Select only the 'path' and 'emotion' columns
    train_df = train_df[['path', 'emotion']]
    test_df = test_df[['path', 'emotion']]
    
    # Save the train and test DataFrames to CSV files
    train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False, sep="\t", encoding="utf-8")
    test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False, sep="\t", encoding="utf-8")
