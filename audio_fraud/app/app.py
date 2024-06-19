import psycopg2
import random
import io
import wave
from src.predict import Predict
from decouple import config

DB_NAME = config('DB_NAME')
DB_USER = config('DB_USER')
DB_PASSWORD = config('DB_PASSWORD')
DB_HOST = config('DB_HOST')
DB_PORT = config('DB_PORT')

# prediction_label, score = Predict.run_predict('BobRoss_Fake_Real.wav')

# print('prediction_label', prediction_label)
# print('score', score)
# print('<<{"result": '+ str(prediction_label) + ', "score": '+ str(score) + '}>>')

# print(DB_NAME, DB_USER, DB_PASSWORD, DB_HOST, DB_PORT)

conn = psycopg2.connect(dbname=DB_NAME,
                        user=DB_USER,
                        password=DB_PASSWORD,
                        host=DB_HOST,
                        port=DB_PORT)
cur = conn.cursor()

select_query = """
    SELECT 
       file, operation_id 
    FROM
        public.tcc_2
    WHERE
            was_processed is false 
        AND
            ai_flow = 'audio_fraud'
"""
cur.execute(select_query)
rows = cur.fetchall()
for row in rows:
    ## DO WHAT THE IMAGE MUST DO AND UPDATE
    audio_data, operation_id = row

    ## prediction_label, score = Predict.run_predict(row.file)

    audio_buffer = io.BytesIO(audio_data)

    # Abra o buffer de bytes usando o módulo wave
    with wave.open(audio_buffer, 'rb') as audio_file:
        # Obtenha as informações do áudio
        params = audio_file.getparams()
        
        # Leia os frames do áudio
        audio_frames = audio_file.readframes(params.nframes)

    # Escreva os frames em um novo arquivo .wav
    with wave.open('src/audio_files/output.wav', 'wb') as output_file:
        output_file.setparams(params)
        output_file.writeframes(audio_frames)

    print("Arquivo .wav criado com sucesso.")

    prediction_label, score = Predict.run_predict('output.wav')

    print('prediction_label', prediction_label)
    print('score', score)
    print('<<{"result": '+ str(prediction_label) + ', "score": '+ str(score) + '}>>')
    
    update_query = f"""
    UPDATE 
        public.tcc_2 
    SET 
        was_processed = true,
        score = {score},
        is_real = {prediction_label}
    WHERE     
            was_processed is false 
        AND
            ai_flow = 'audio_fraud'
        AND 
            operation_id = {operation_id}
            ;
    """
    cur.execute(update_query)
    conn.commit()