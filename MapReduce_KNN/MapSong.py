import pandas as pd
import ast


def map_id_to_song(file):
    song_dict = {}
    original_file = pd.read_csv('.\data.csv')
    with open(file,encoding='utf-16') as src:
        for line in src:
            input_song, output_songs = line.split('\t')
            output_songs = output_songs.replace('\n', '')
            input_song = ast.literal_eval(input_song)
            output_songs = ast.literal_eval(output_songs)
            song_dict[input_song] = output_songs
        
    for key, value in song_dict.items():
        i_song = pd.DataFrame()
        o_song = pd.DataFrame()
        i_song = i_song.append(original_file[(original_file['id'] == key)])
        for index, song in i_song.iterrows():
            print('Ihre Vorschläge für das Lied: {} von {}: \n'.format(song['name'], song['artists']))
        for suggested_song in value:
            o_song = o_song.append(original_file[(original_file['id'] == suggested_song)])
        for index, song in o_song.iterrows():
            print(' {} von {} \n'.format(song['name'], song['artists']))

                    
    # for index, song in df_new.iterrows():
    #     print('Name des Songs: {} \n Künstler: {} \n'.format(song['name'], song['artists']))
        
    # print(df_new)

map_id_to_song('.\output.json')