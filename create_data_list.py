import os


def get_data_list(audio_path, list_path):
    sound_sum = 0
    audios = os.listdir(audio_path)
    os.makedirs(list_path, exist_ok=True)
    f_train = open(os.path.join(list_path, 'train_list.txt'), 'w', encoding='utf-8')
    f_test = open(os.path.join(list_path, 'test_list.txt'), 'w', encoding='utf-8')
    f_label = open(os.path.join(list_path, 'label_list.txt'), 'w', encoding='utf-8')

    for i in range(len(audios)):
        f_label.write(f'{audios[i]}\n')
        sounds = os.listdir(os.path.join(audio_path, audios[i]))
        for sound in sounds:
            sound_path = os.path.join(audio_path, audios[i], sound).replace('\\', '/')
            if sound_sum % 10 == 0:
                f_test.write(f'{sound_path}\t{i}\n')
            else:
                f_train.write(f'{sound_path}\t{i}\n')
            sound_sum += 1
        print(f"Audio：{i + 1}/{len(audios)}")
    f_label.close()
    f_test.close()
    f_train.close()





if __name__ == '__main__':
     get_data_list('dataset/audio', 'dataset')
    
