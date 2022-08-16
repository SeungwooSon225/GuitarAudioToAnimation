import numpy as np
import json
import os
import matplotlib.pyplot as plt


def interpolate_missing_data():
    print('interpolating missing data...')

    dir_path = "./data"

    file_number = 0
    for (root, directories, files) in os.walk(dir_path):
        for file in files:
            if '_video.json' in file:
                file_number = file_number + 1
                file_path = os.path.join(root, file)
                file_path = file_path.replace('\\', '/')
                print('interpolate missing data: ', file_path)

                with open(file_path, 'r') as file:
                    json_data = json.load(file)

                hand_landmarks = json_data['0']

                for index in range(1, len(hand_landmarks)):

                    if sum(hand_landmarks[index]) == 0:
                        zero_data_start = index
                        zero_data_end = index

                        while sum(hand_landmarks[zero_data_end]) == 0:
                            zero_data_end = zero_data_end + 1
                            if zero_data_end == len(hand_landmarks):
                                zero_data_end = zero_data_end - 1
                                break

                        zero_data_length = zero_data_end - zero_data_start

                        if zero_data_length <= 30:
                            step_size = (np.array(hand_landmarks[zero_data_end]) - np.array(
                                hand_landmarks[zero_data_start - 1])) / (zero_data_length + 1)

                            for i in range(1, zero_data_length + 1):
                                hand_landmarks[zero_data_start + i - 1] = (
                                            np.array(hand_landmarks[zero_data_start - 1]) + step_size * i).tolist()

                json_data = {
                    "0": hand_landmarks
                }

                file_path = file_path.replace('data', 'processed data')
                with open(file_path, 'w', encoding='utf-8') as file:
                    json.dump(json_data, file)


def concatenate_data():
    print('concatenating data...')

    dir_path = "./processed data"

    concatenated_video_data = []
    concatenated_audio_data = []

    for (root, directories, files) in os.walk(dir_path):
        for file in files:
            if '_video.json' in file:
                file_path = os.path.join(root, file)
                file_path = file_path.replace('\\', '/')

                print(file_path)
                with open(file_path, 'r') as file:
                    json_data = json.load(file)

                hand_landmarks = json_data['0']

                concatenated_video_data = concatenated_video_data + hand_landmarks

                # for i in range(100):
                #     concatenated_video_data = concatenated_video_data + [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                #                                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                #                                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                #                                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                #                                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                #                                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                #                                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                #                                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                #                                                           0, 0, 0, 0]]

        for file in files:
            if '_audio.json' in file:
                file_path = os.path.join(root, file)
                file_path = file_path.replace('\\', '/')

                print(file_path)
                print(file_path)
                with open(file_path, 'r') as file:
                    json_data = json.load(file)

                audio_features = json_data['0']

                concatenated_audio_data = concatenated_audio_data + audio_features

                # for i in range(100):
                #     concatenated_audio_data = concatenated_audio_data + [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                #                                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                #                                                           0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]

    video_json_data = {
        "0": concatenated_video_data
    }

    with open('./processed data/concatenated_video_data.json', 'w', encoding='utf-8') as file:
        json.dump(video_json_data, file)

    audio_json_data = {
        "0": concatenated_audio_data
    }

    with open('./processed data/concatenated_audio_data.json', 'w', encoding='utf-8') as file:
        json.dump(audio_json_data, file)


def pick_audio_data():
    print('picking audio data...')

    dir_path = "./data"

    video_data_lengths = []

    for (root, directories, files) in os.walk(dir_path):
        for file in files:
            if '_video.json' in file:
                file_path = os.path.join(root, file)
                file_path = file_path.replace('\\', '/')

                with open(file_path, 'r') as file:
                    json_data = json.load(file)

                video_data_lengths = video_data_lengths + [len(json_data['0'])]

    print(video_data_lengths)

    for (root, directories, files) in os.walk(dir_path):
        file_number = 0

        for file in files:
            if '_audio.json' in file:
                file_number = file_number + 1
                file_path = os.path.join(root, file)
                file_path = file_path.replace('\\', '/')

                audio_file_path = file_path.replace('video', 'audio')
                audio_file_path = audio_file_path.replace('data', 'processed data')

                with open(file_path, 'r') as file:
                    json_data = json.load(file)

                print(file_number, ': ', file_path, '/ ', len(json_data['0']))

                audio_feature = []

                for i in range(video_data_lengths[file_number-1]):
                    if i * 3 >= len(json_data['0']):
                        audio_feature = audio_feature + [json_data['0'][len(json_data['0'])-1]]
                    else:
                        audio_feature = audio_feature + [json_data['0'][i * 3]]

                print(len(audio_feature))

                json_data = {
                    "0": audio_feature
                }

                with open(audio_file_path, 'w', encoding='utf-8') as file:
                    json.dump(json_data, file)


def show_data(dir_path):
    for (root, directories, files) in os.walk(dir_path):
        for file in files:
            if '_audio.json' in file:
                file_path = os.path.join(root, file)
                file_path = file_path.replace('\\', '/')

                with open(file_path, 'r') as file:
                    json_data = json.load(file)

                plt.plot(json_data['0'])
                plt.show()


def remove_trash_data():
    print('removing trash data...')

    dir_path = "./data"

    for (root, directories, files) in os.walk(dir_path):
        for file in files:
            if '_video.json' in file:
                file_path = os.path.join(root, file)
                file_path = file_path.replace('\\', '/')

                audio_file_path = file_path.replace('video', 'audio')
                audio_file_path = audio_file_path.replace('data', 'processed data')

                print(file_path)
                print(audio_file_path)

                with open(file_path, 'r') as file:
                    video_data = json.load(file)

                hand_landmarks = video_data['0']

                with open(audio_file_path, 'r') as file:
                    audio_data = json.load(file)

                audio_features = audio_data['0']

                plt.plot(hand_landmarks)
                plt.show()

                plt.plot(audio_features)
                plt.show()

                trash_data_index = []

                for index in range(0, len(hand_landmarks)):
                    if sum(hand_landmarks[index]) == 0:
                        zero_data_start = index
                        zero_data_end = index

                        while sum(hand_landmarks[zero_data_end]) == 0:
                            hand_landmarks[zero_data_end][0] = -1
                            zero_data_end = zero_data_end + 1
                            if zero_data_end == len(hand_landmarks): break

                        zero_data_length = zero_data_end - zero_data_start

                        if zero_data_length > 60:
                            # print('start: ', zero_data_start, '|  end: ', zero_data_end)
                            trash_data_index = trash_data_index + [(zero_data_start, zero_data_end)]

                print(trash_data_index)

                for start, end in trash_data_index:
                    if end >= len(audio_features):
                        end = end-1

                    print(len(audio_features), ' / ', start, end)
                    for i in range(start, end-1):
                        audio_features[i] = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                             0, 0, 0, 0, 0, 0]

                # print(np.array(audio_features).shape)
                # plt.plot(audio_features)
                # plt.show()
                # print(np.array(hand_landmarks).shape)
                # plt.plot(hand_landmarks)
                # plt.show()

                audio_json_data = {
                    "0": audio_features
                }

                with open(audio_file_path, 'w', encoding='utf-8') as file:
                    json.dump(audio_json_data, file)


# pick_audio_data()
# remove_trash_data()
# interpolate_missing_data()
# concatenate_data()

# show_data("./data")

# with open('data/data_legacy/training1_video.json', 'r') as file:
#     json_data = json.load(file)

with open('data/training17_video1.json', 'r') as file:
    json_data = json.load(file)

print(np.array(json_data['0']).shape)
#
plt.plot(json_data['0'])
plt.show()


# with open('processed data/concatenated_video_data.json', 'r') as file:
#     json_data = json.load(file)
#
# print(np.array(json_data['0']).shape)
#
# # plt.plot(json_data['0'])
# # plt.show()
#
# json_data = json_data['0']
#
# for i in range(len(json_data)):
#     json_data[i] = [json_data[i][4*2]] + [json_data[i][4*2+1]] + \
#                    [json_data[i][8*2]] + [json_data[i][8*2+1]] + \
#                    [json_data[i][12*2]] + [json_data[i][12*2+1]] + \
#                    [json_data[i][16*2]] + [json_data[i][16*2+1]] + \
#                    [json_data[i][20*2]] + [json_data[i][20*2+1]] + \
#                    [json_data[i][3 * 2]] + [json_data[i][3 * 2 + 1]] + \
#                    [json_data[i][6 * 2]] + [json_data[i][6 * 2 + 1]] + \
#                    [json_data[i][10 * 2]] + [json_data[i][10 * 2 + 1]] + \
#                    [json_data[i][14 * 2]] + [json_data[i][14 * 2 + 1]] + \
#                    [json_data[i][18 * 2]] + [json_data[i][18 * 2 + 1]] + \
#                    [json_data[i][2 * 2]] + [json_data[i][2 * 2 + 1]] + \
#                    [json_data[i][5 * 2]] + [json_data[i][5 * 2 + 1]] + \
#                    [json_data[i][9 * 2]] + [json_data[i][9 * 2 + 1]] + \
#                    [json_data[i][13 * 2]] + [json_data[i][13 * 2 + 1]] + \
#                    [json_data[i][17 * 2]] + [json_data[i][17 * 2 + 1]] + \
#                    [json_data[i][0 * 2]] + [json_data[i][0 * 2 + 1]] + \
#                    [json_data[i][42 + 4 * 2]] + [json_data[i][42 + 4 * 2 + 1]] + \
#                    [json_data[i][42 + 8 * 2]] + [json_data[i][42 + 8 * 2 + 1]] + \
#                    [json_data[i][42 + 12 * 2]] + [json_data[i][42 + 12 * 2 + 1]] + \
#                    [json_data[i][42 + 16 * 2]] + [json_data[i][42 + 16 * 2 + 1]] + \
#                    [json_data[i][42 + 20 * 2]] + [json_data[i][42 + 20 * 2 + 1]] + \
#                    [json_data[i][42 + 3 * 2]] + [json_data[i][42 + 3 * 2 + 1]] + \
#                    [json_data[i][42 + 6 * 2]] + [json_data[i][42 + 6 * 2 + 1]] + \
#                    [json_data[i][42 + 10 * 2]] + [json_data[i][42 + 10 * 2 + 1]] + \
#                    [json_data[i][42 + 14 * 2]] + [json_data[i][42 + 14 * 2 + 1]] + \
#                    [json_data[i][42 + 18 * 2]] + [json_data[i][42 + 18 * 2 + 1]] + \
#                    [json_data[i][42 + 2 * 2]] + [json_data[i][42 + 2 * 2 + 1]] + \
#                    [json_data[i][42 + 5 * 2]] + [json_data[i][42 + 5 * 2 + 1]] + \
#                    [json_data[i][42 + 9 * 2]] + [json_data[i][42 + 9 * 2 + 1]] + \
#                    [json_data[i][42 + 13 * 2]] + [json_data[i][42 + 13 * 2 + 1]] + \
#                    [json_data[i][42 + 17 * 2]] + [json_data[i][42 + 17 * 2 + 1]] + \
#                    [json_data[i][42 + 0 * 2]] + [json_data[i][42 + 0 * 2 + 1]]
# #
# print(np.array(json_data).shape)
#
# json_data = {
#     "0": json_data
# }
#
# with open('processed data/concatenated_video_data.json', 'w', encoding='utf-8') as file:
#     json.dump(json_data, file)



