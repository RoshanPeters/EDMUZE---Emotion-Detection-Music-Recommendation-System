#!/usr/bin/env python
# coding: utf-8

# In[2]:


#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys

#name = input('Please enter name: ')

import pywhatkit as pwt

import pandas as pd

import cv2
#from deepface import DeepFace

import math

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

import mysql.connector

#music_df = pd.read_csv('unseen_dataset.csv')

#Connecting to sql
mydb = mysql.connector.connect(
    host="",
    user="",
    passwd= "",
    database=""
)
mycursor = mydb.cursor()

# user login and sign up
choice = int(input("press 1 : if you are a first time user\npress 2: if you want to login\n"))
def signup():
    user_name = input('please enter your name:')
    passwd = input('please enter password:')
    params = (user_name, passwd)
    mycursor.execute("INSERT INTO users (name, pswd) VALUES (%s, %s)", params)
    mydb.commit()

    print('Hey {} your account has been created successfully'.format(user_name))

    print('login with your account to use edmuze')

logingate = []


def login():


    def add_song_db(songname):
        df2 = music_df.loc[music_df['SONG'] == songname]
        contents = []

        # getting the contents of a specific row
        for col in music_df:
            print(df2[col].values[0])
            contents.append(df2[col].values[0])

        song_name = songname
        artist = contents[2]
        emotion = contents[3]
        lang = contents[4]
        genre = contents[5]
        user_id = logingate[0]
        params = (song_name, artist, emotion, user_id, lang, genre)
        mycursor.execute(
            "INSERT INTO heard_songs (SONG, ARTIST, EMOTION, user_id, LANGUAGE, GENRE)VALUES (%s, %s, %s, %s, %s, %s)", params)
        mydb.commit()
        
    def remove_from_unheard(songname):
        try:
            sql = "DELETE from unheard_songs WHERE SONG = %s"
            param = (songname,)
            mycursor.execute(sql, param)
            mydb.commit()
        except:
            return  # the song has been already removed
 


    user_name = input('please enter your name:')
    pswd = input('enter your password:')
    params = (user_name, pswd)
    mycursor.execute(
        "SELECT * FROM users WHERE name = %s AND pswd = %s", params)
    myresult = mycursor.fetchall()
    for i in myresult:
        logingate.append(i[0])

    if len(logingate) == 1:
        print('you have successfully logged in')

        #Main code starts from here
        print('Welcome to Edmuze')

        mycursor.execute("SELECT * FROM playlist" )
        myresult = mycursor.fetchall()
        df = pd.DataFrame(myresult)
        #print(df.head())
        music_df = df.rename(columns = {0: "song_no", 1: "SONG", 2: "ARTIST", 3: "EMOTION",  4: "LANGUAGE", 5: "GENRE"})
        #print(music_df)

        music_df['features'] = music_df['ARTIST'] + ', ' + music_df['EMOTION'] + ', ' + music_df['LANGUAGE'] + ', ' + music_df['GENRE']

        # using the tfidf vectorizer
        from sklearn.feature_extraction.text import TfidfVectorizer

        tv = TfidfVectorizer(min_df=3, max_features=None, strip_accents='unicode', analyzer='word', token_pattern=r'\w{1,}',
                            ngram_range=(1, 3), stop_words='english')

        # creating the sparse matrix
        tv_sparse_matrix = tv.fit_transform(music_df['features'])  # the output here is giving the scores

        # making the scores in the range of 0 to 1
        from sklearn.metrics.pairwise import sigmoid_kernel

        sigmoid = sigmoid_kernel(tv_sparse_matrix, tv_sparse_matrix)
        indices = pd.Series(music_df.index, index=music_df['SONG']).drop_duplicates()

        song_list = []









# In[2]:


        def first_song(user_emo):
            

            #df_seen = pd.read_csv('seen_dataset.csv')
            mycursor.execute("SELECT * FROM heard_songs" )
            myresult = mycursor.fetchall()
            df = pd.DataFrame(myresult)
            df_seen = df.rename(columns = {0: "song_no", 1: "SONG", 2: "ARTIST", 3: "EMOTION", 4: "user_id", 5: "LANGUAGE", 6: "GENRE"})


            #df_unseen = pd.read_csv('unseen_dataset.csv')

            mycursor.execute("SELECT * FROM unheard_songs" )
            myresult = mycursor.fetchall()
            df = pd.DataFrame(myresult)
            df_unseen = df.rename(columns = {0: "song_no", 1: "SONG", 2: "ARTIST", 3: "EMOTION", 4: "user_id", 5: "LANGUAGE", 6: "GENRE"})

            if len(df_seen) >= 3:
                
                #print(df_seen.head())
                #print(df_unseen.head())
                #print(df_seen.columns)
                #print(df_unseen.columns)

                # df_seen.dropna(inplace=True)
                # df_unseen.dropna(inplace=True)
                
                
                details = []
                for i in range(len(df_seen)):
                    #print(i)
                    ar = df_seen["ARTIST"][i]
                    gn = df_seen["GENRE"][i]
                    lang = df_seen["LANGUAGE"][i]
                    emo = df_seen["EMOTION"][i]
                    ar_gn = str(ar).strip() + " " + str(gn).strip() + " " + str(lang).strip() + " " + str(emo).strip()
                    details.append(ar_gn)
                df_seen['Details'] = details

                details = []
                for i in range(len(df_unseen)):
                    #print(i)
                    ar = df_unseen["ARTIST"][i]
                    gn = df_unseen["GENRE"][i]
                    lang = df_unseen["LANGUAGE"][i]
                    emo = df_unseen["EMOTION"][i]
                    ar_gn = str(ar).strip() + " " + str(gn).strip() + " " + str(lang).strip() + " " + str(emo).strip()
                    details.append(ar_gn)
                df_unseen['Details'] = details
                
                
                emotions = ["Happy", "Sad", "Angry"]
                
                
                getEmotion = 0
                if user_emo == "Happy":
                    getEmotion = 1
                elif user_emo == "Sad":
                    getEmotion = 2
                elif user_emo == "Angry":
                    getEmotion = 3
                
                emotion_str = emotions[getEmotion-1]

                #print(df_unseen.head())  # for testing purposes

                df_seen_new = df_seen[df_seen['EMOTION'] == emotion_str]
                df_unseen_new = df_unseen[df_unseen['EMOTION'] == emotion_str]
                
                df_seen_new = df_seen_new.reset_index(drop=True)
                df_unseen_new = df_unseen_new.reset_index(drop=True)
                
                
                no_of_recomendations = 3
                
                df_dict = {}
                for i in range(len(df_unseen_new)):
                    df_dict[str(df_unseen_new['SONG'][i]).strip()] = [0,0]
                
                #print("df_unseen_new")
                #print(df_unseen_new)
                #print(" ")
                len_new = len(df_seen_new)
                for i in range(len_new):

                    entry_song = df_seen_new['SONG'][i]
                    entry_artist = df_seen_new['ARTIST'][i]
                    entry_emotion = df_seen_new['EMOTION'][i]
                    entry_language = df_seen_new['LANGUAGE'][i]
                    entry_genre = df_seen_new['GENRE'][i]
                    entry_details = df_seen_new['Details'][i]
                    l = len(df_unseen_new)

                    df_copy = pd.DataFrame(df_unseen_new)

                    #print("df_copy")
                    #print(df_copy)
                    #print()


                    df_copy.loc[l] = [3, entry_song, entry_artist, entry_emotion, 0, entry_language, entry_genre, entry_details]

                    #print(df_copy)

                    appended_song = str(df_copy['SONG'][l])

                    tf = TfidfVectorizer(analyzer='word', ngram_range=(1, 3), min_df=0, stop_words='english')
                    matrix = tf.fit_transform(df_copy['Details'])
                    cosine_similarities = linear_kernel(matrix,matrix)
                    song_title = df_copy['SONG']
                    indices = pd.Series(df_copy.index, index=df_copy['SONG'])
                    
                    def song_recommend(original_title):

                        idx = indices[original_title]

                        sim_scores = list(enumerate(cosine_similarities[idx]))

                        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

                        sim_scores = sim_scores[1:20]

                        song_indices = [i[0] for i in sim_scores]

                        #print(song_indices)

                        return song_title.iloc[song_indices]
                    
                    song_item = df_copy['SONG'][l]
                    df_recommended = song_recommend(song_item).head(no_of_recomendations)

                    df_n = df_recommended.to_frame()
                    presence = len(df_n[df_n['SONG'] == appended_song])

                    if presence>0:
                        index_val = df_n[df_n['SONG'] == appended_song].index.values[0]
                        df_n = df_n.drop(index_val)

                    temp = 0

                    for index, row in df_n.iterrows():
                        df_dict[row['SONG']][0] += temp+1
                        df_dict[row['SONG']][1] += 1
                        temp += 1
                
            
                
                recommendation = []
                for key, value in sorted(df_dict.items(), key = lambda e: (-e[1][1],e[1][0])):
                    recommendation.append(key)
                    #print(key, value)
                
                #print(" ")
                #print(" ")
                #print(" ")
                
                recommended_songs = recommendation[:no_of_recomendations]
                return recommended_songs
        
            else:

                user_language = input("Enter your preferred language - ")

                short_df = df_unseen[df_unseen['EMOTION'] == user_emo]
                short_df = short_df[short_df['LANGUAGE'] == user_language]
                # capturing the song based on the emotion and language
                random_rec_song = short_df['SONG'].sample(n=3)
                print(random_rec_song)
                random_song_name = input(
                    'please enter the name of the song of your choice as appeared on the screen:')
                # print('The recommended song for you to listen to is: ', random_song_name)
                pwt.playonyt(random_song_name)
                song_list.append(random_song_name)
                add_song_db(random_song_name)
                remove_from_unheard(random_song_name)

                #Recommend from unheard
                id_number = indices[random_song_name]
                sigmoid_scores = sorted(list(enumerate(sigmoid[id_number])), key=lambda x: x[1], reverse=True)
                sigmoid_scores = sigmoid_scores[1:6]
                song_indices = [s[0] for s in sigmoid_scores]
                rec = df_unseen['SONG'].iloc[song_indices]
                rec_list = list(rec)
                return rec

                

# In[3]:


        def emotion_detection():
            from deepface import DeepFace

            user_choice = int(input('press 0 if you want to manually enter your current emotion \n(or)\npress 1 if you want to '
                                    'automate it :'))

            if user_choice == 0:
                #user_lang = input(f'hey {name}, please enter your preferred lang to listen to: ')
                user_emotion = input('please enter your current state of mood: ')
                return user_emotion
                #recommended_songs = first_song(user_emotion)

            else:
                #user_lang = input(f'hey {name}, please enter your preferred lang to listen to: ')

                # drawing the rectangle around the face
                # haar cascade is a face recognition algorithm by viola jones
                faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

                # initiating the camera
                cap = cv2.VideoCapture(1)

                if not cap.isOpened():
                    cap = cv2.VideoCapture(0)
                if not cap.isOpened():
                    raise IOError("Cannot open Webcam")

                # i = 0
                emotion_in_frame = []

                while True:
                    ret, frame = cap.read()  # read one image from a video
                    result = DeepFace.analyze(frame, actions=['emotion'])

                    # creating the variable for storing the emotion in each frame
                    emotion_in_frame.append(result['dominant_emotion'])
                    # print(emotion_in_frame)
                    if len(emotion_in_frame) == 15:
                        break

                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # by default cv2 image is in bgr format
                    faces = faceCascade.detectMultiScale(gray, 1.1, 4)

                    for (x, y, w, h) in faces:
                        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # green border

                        font = cv2.FONT_HERSHEY_SIMPLEX

                        cv2.putText(frame,
                                    result['dominant_emotion'],
                                    (50, 50), font, 3,
                                    (0, 0, 255), 2, cv2.LINE_4)  # red text
                        cv2.imshow('Capture Video', frame)

                        if cv2.waitKey(2) & 0xFF == ord('q'):
                            break
                cap.release()
                cv2.destroyAllWindows()

                def most_frequent(List):  # finding the most frequent emotion in the list
                    return max(set(List), key=List.count)

                output_emotion = most_frequent(emotion_in_frame)
                print('We have found out you are:', output_emotion)

                # grouping the emotions into a total of 3 main emotions
                if output_emotion == 'neutral':
                    output_emotion = 'Happy'

                elif output_emotion == 'happy':
                    output_emotion = 'Happy'

                elif output_emotion == 'angry':
                    output_emotion = 'Angry'

                elif output_emotion == 'sad':
                    output_emotion = 'Sad'

                elif output_emotion == 'fear':
                    output_emotion = 'Sad'

                elif output_emotion == 'surprise':
                    output_emotion = 'Happy'

                else:
                    output_emotion = 'Angry'

                user_emotion = output_emotion
                return user_emotion
                #recommended_songs = first_song(user_emotion)


# In[4]:


        emotion_user = emotion_detection()

        recommended_songs = first_song(emotion_user)


# In[5]:


        def recommend(title, sigmoid=sigmoid):
            id_number = indices[title]
            sigmoid_scores = sorted(list(enumerate(sigmoid[id_number])), key=lambda x: x[1], reverse=True)
            sigmoid_scores = sigmoid_scores[1:6]
            song_indices = [s[0] for s in sigmoid_scores]
            rec = music_df['SONG'].iloc[song_indices]
            rec_list = list(rec)
            print(rec)

            # coming out of the loop
            if len(song_list) == 2:
                user_input = input("press 'A' if you want to continue with the current playlist: \npress 'B' if you want "
                                "to make changes: \npress 'X' if you want to end the program")
                if user_input == 'A':
                    song_list.clear()
                    recommend(rec_list[2])
                elif user_input == 'B':
                    song_list.clear()
                    emotion_user = emotion_detection()
                    recommended_songs = first_song(emotion_user)
                    print(" ")
                    print("The recommended songs are: ")
                    print(recommended_songs)
                    index_song = int(input('enter the index number (starting from 0) of the recommended song you would like to listen to: '))
                    pwt.playonyt(recommended_songs[index_song])
                    song_list.append(recommended_songs[index_song])
                    add_song_db(recommended_songs[index_song])
                    remove_from_unheard(recommended_songs[index_song])
                    recommend(recommended_songs[index_song])
                    
                else:
                    sys.exit()

            # open a youtube link and send it into rec function and load it into a list
            index_list = int(input('enter the index number of the recommended song you would like to listen to: '))
            

            if index_list == 0:
                pwt.playonyt(rec_list[0])
                song_list.append(rec_list[0])
                add_song_db(rec_list[0])
                remove_from_unheard(rec_list[0])
                recommend(rec_list[0])

            elif index_list == 1:
                pwt.playonyt(rec_list[1])
                song_list.append(rec_list[1])
                add_song_db(rec_list[1])
                remove_from_unheard(rec_list[1])
                recommend(rec_list[1])

            elif index_list == 2:
                pwt.playonyt(rec_list[2])
                song_list.append(rec_list[2])
                add_song_db(rec_list[2])
                remove_from_unheard(rec_list[2])
                recommend(rec_list[2])

            elif index_list == 3:
                pwt.playonyt(rec_list[3])
                song_list.append(rec_list[3])
                add_song_db(rec_list[3])
                remove_from_unheard(rec_list[3])
                recommend(rec_list[3])

            else:
                pwt.playonyt(rec_list[4])
                song_list.append(rec_list[4])
                add_song_db(rec_list[4])
                remove_from_unheard(rec_list[4])
                recommend(rec_list[4])


# In[6]:

        print(" ")
        print("The recommended songs are: ")
        print(recommended_songs)
        index_song = int(input('enter the index number (starting from 0) of the recommended song you would like to listen to: '))
        
        pwt.playonyt(recommended_songs[index_song])
        add_song_db(recommended_songs[index_song])
        remove_from_unheard(recommended_songs[index_song])
        song_list.append(recommended_songs[index_song])
        recommend(recommended_songs[index_song])


# In[ ]:

if choice == 1:
    signup()
    login()
elif choice == 2:
    login()
else:
    print('you have entered an incorrect input')


