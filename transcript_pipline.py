import math
import time
import numpy as np
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
import pandas as pd
import datetime
import os

from pydub import AudioSegment

from aai_transcripotor import convert_to_dialogue,aai_trainscript,download_audio
from transcript_functions import generate_transcript,get_id,get_title,save_and_upload_file,upload_mp3_drive,split_audio,save_locally

#gauth = GoogleAuth()
#gauth.LocalWebserverAuth()


FILE_URL = "https://github.com/iskaiska911/wav_files/raw/4ac005e1a4e5cd13c10f78a9c2f98636d445c0d3/Case%20study%20clinical%20example%20CBT%20First%20session%20with%20a%20client%20with%20symptoms%20of%20depression%20(CBT%20model).mp3"
YOUR_API_TOKEN = "4cec269d03324f32b13c1abd33e61e5b"
GITHUB_TOKEN = "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"

#A
#https://www.youtube.com/watch?v=zq3jFUq-P0Y
#https://www.youtube.com/watch?v=K6IFaXghzK0
#https://www.youtube.com/watch?v=vudaAYx2IcE
#https://www.youtube.com/watch?v=UZTyvbmW92M
#https://www.youtube.com/watch?v=1vd7Tx3E0kM
#https://www.youtube.com/watch?v=lwuouSVtWAE


#B
#https://www.youtube.com/watch?v=86KM1g6RIsA
#https://www.youtube.com/watch?v=yrFevU-EE1s
#https://www.youtube.com/watch?v=_8-dhTodlKI
#https://www.youtube.com/watch?v=7SykD4oi450
#https://www.youtube.com/watch?v=sY8F-tuW4yo
#https://www.youtube.com/watch?v=KpqAfEPHRP8


#B
#https://www.youtube.com/watch?v=sfavW8r0kM0
#https://www.youtube.com/watch?v=ZQbNJa2zUZ4
#https://www.youtube.com/watch?v=Fu1wCmwKG2A
#https://www.youtube.com/watch?v=dqV6pjX_dP0
#https://www.youtube.com/watch?v=51uE4WTvs5M
#https://www.youtube.com/watch?v=gVEqTwY-LZQ
#https://www.youtube.com/watch?v=BqP5wQ8KE5I
#https://www.youtube.com/watch?v=aXfgIYMrJ7c

#A
#https://www.youtube.com/watch?v=Mxq3CP9Tzf4
#https://www.youtube.com/watch?v=TChiE1FDXdY
#https://www.youtube.com/watch?v=BBQb1SVPboI
#https://www.youtube.com/watch?v=J16Zyknu9Mw

#jung
#A
#Carl Jung
#https://www.youtube.com/watch?v=oBYEFX2dqpM

#B
#https://www.youtube.com/watch?v=lioQ616lEzA
#https://www.youtube.com/watch?v=oSx_RxJjjMk
#https://www.youtube.com/watch?v=Em-VqtpNrgg

#A
#https://www.youtube.com/watch?v=FePVZSKrxlA

#B
#https://www.youtube.com/watch?v=E7Yri6YvFmI
#https://www.youtube.com/watch?v=6eaRPVTepNc
#https://www.youtube.com/watch?v=jsnUonUZBuY
#https://www.youtube.com/watch?v=VQZ8oPXgK8o


######B
#https://www.youtube.com/watch?v=e49wwwmLj74


#B
#https://www.youtube.com/watch?v=bTh8Cogqi3c
#https://www.youtube.com/watch?v=NnnJkciaFSI
#https://www.youtube.com/watch?v=zVqw0DsCkjo

#https://www.youtube.com/watch?v=cOHIcxm2qEQ
#https://www.youtube.com/watch?v=TCG4VWbG8ck

LinkList=[
          'https://www.youtube.com/watch?v=bTh8Cogqi3c',
    'https://www.youtube.com/watch?v=NnnJkciaFSI',
    'https://www.youtube.com/watch?v=zVqw0DsCkjo',
    'https://www.youtube.com/watch?v=q9b2DIrCeHU'
                  ]


texts=[]
segment_duration=15

for link in LinkList:
    time.sleep(np.random.choice([x / 10 for x in range(7, 22)]))
    id = get_id(link)
    try:
        download_link = save_and_upload_file(get_title(id), generate_transcript(id)[0])
        file_name = download_audio(link)
        if (AudioSegment.from_file(file_name, format='mp3').duration_seconds / 60) <= 70:
            download_url = upload_mp3_drive(file_name)
            transcript = aai_trainscript(api_token=YOUR_API_TOKEN, file_url=download_url)
            dialogue = convert_to_dialogue(transcript['words'])
            transcript_link = save_and_upload_file(file_name, dialogue)
            save_locally(file_name, dialogue)
            texts.append([get_title(id), link, download_link, download_url, transcript_link,'speaker A - coach', dialogue.count('Speaker B')])
        else:
            file_names = split_audio(file_name)
            for file_name in file_names:
                download_url = upload_mp3_drive(file_name)
                transcript = aai_trainscript(api_token=YOUR_API_TOKEN, file_url=download_url)
                dialogue = convert_to_dialogue(transcript['words'])
                transcript_link = save_and_upload_file(file_name, dialogue)
                save_locally(file_name, dialogue)
                texts.append([get_title(id), link, download_link, download_url, transcript_link,'speaker A - coach',dialogue.count('Speaker B')])

    except TypeError:
        pass

df = pd.DataFrame(texts, columns=['title', 'video', 'text', 'download_url', 'transcript_link', 'flag','count_prompts'], dtype=str)
df.to_csv(os.path.join(os.getcwd(), 'results_txt','transcrpit'+str(datetime.datetime.now().strftime("%Y_%m_%d-%I_%M_%S_%p"))+'.csv'), encoding='utf-8', sep=',')








