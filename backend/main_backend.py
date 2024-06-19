from flask import Flask, render_template, request, jsonify
import subprocess
import psycopg2
from psycopg2 import sql
from io import BytesIO
from flask_cors import CORS
import re
import json
from decouple import config

DB_NAME = config('DB_NAME')
DB_USER = config('DB_USER')
DB_PASSWORD = config('DB_PASSWORD')
DB_HOST = config('DB_HOST')
DB_PORT = config('DB_PORT')

app = Flask(__name__)
CORS(app) 
@app.route('/upload', methods=['POST'])
def upload_file_audio():
    if 'file' not in request.files:
        return 'No file part'

    # Database configuration

    conn = psycopg2.connect(dbname=DB_NAME,
                            user=DB_USER,
                            password=DB_PASSWORD,
                            host=DB_HOST,
                            port=DB_PORT)
    cur = conn.cursor()
    module_name = request.form.get('module_name')
    file = request.files['file']

    if file.filename == '':
        return 'No selected file'

    file_data = file.read()

    # Insert file into PostgreSQL
    cur.execute("INSERT INTO tcc_2 (comment,ai_flow, file) VALUES (%s,%s,%s)", (file.filename,module_name, file_data))

    conn.commit()
    cur.close()
    conn.close()
    
    result = subprocess.run(
    ['docker', 'run', '--rm', module_name],
    capture_output=True,
    text=True  # Ensure the output is returned as a string instead of bytes
    )
    print("Standard out:")
    print(result.stdout)
    pattern = r'<<(.*?)>>'
    # Encontrar find the result
    matches = re.findall(pattern, result.stdout)
    print(matches)
    if len(matches) == 0:
        return 'error', 400
    return jsonify(matches), 200

if __name__ == '__main__':
    app.run(debug=True,host='0.0.0.0', port=5000)
