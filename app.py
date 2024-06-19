from flask import Flask, render_template, request
import subprocess
import psycopg2
from psycopg2 import sql
from io import BytesIO
from flask_cors import CORS
from decouple import config

DB_NAME = config('DB_NAME')
DB_USER = config('DB_USER')
DB_PASSWORD = config('DB_PASSWORD')
DB_HOST = config('DB_HOST')
DB_PORT = config('DB_PORT')

app = Flask(__name__)
CORS(app) 


@app.route('/')
def index():
    return render_template('index_buttons.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return 'No file part'

    # Database configuration
    conn = psycopg2.connect(dbname=DB_NAME,
                            user=DB_USER,
                            password=DB_PASSWORD,
                            host=DB_HOST,
                            port=DB_PORT)
    cur = conn.cursor()
    print(request.files)
    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    file_data = file.read()
    # print(file_data)
    # Insert file into PostgreSQL
    # Determine which Docker image to trigger based on the button clicked
    # docker_image_name = request.form['docker_image']
    docker_image_name = 'audio_fraud'
    cur.execute("INSERT INTO tcc_2 (comment,ai_flow, file) VALUES (%s,%s,%s)", ('test_2',docker_image_name, file_data))

    conn.commit()
    cur.close()
    conn.close()

    
    # Trigger Docker image with file as an argument
    #subprocess.run(['docker', 'run', '--network', 'container:postgres-pipeline',docker_image_name])

    return render_template('index_buttons.html')

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)
