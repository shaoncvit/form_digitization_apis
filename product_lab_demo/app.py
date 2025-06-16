from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
import uuid
from pipeline import main as pipeline_main

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'output')
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

TEMPLATES = ['hdfc_bank', 'sbi_1', 'sbi_2']

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Handle file upload and template selection
        if 'form_image' not in request.files:
            return render_template('index.html', error='No file part', templates=TEMPLATES)
        file = request.files['form_image']
        template_name = request.form.get('template_name')
        if file.filename == '' or not allowed_file(file.filename):
            return render_template('index.html', error='No selected file or invalid file type', templates=TEMPLATES)
        if template_name not in TEMPLATES:
            return render_template('index.html', error='Invalid template selected', templates=TEMPLATES)
        # Save file with unique name
        filename = f"{uuid.uuid4().hex}_{file.filename}"
        upload_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(upload_path)
        # Use the uploaded filename (without extension) as output_prefix
        output_prefix = os.path.splitext(filename)[0]
        # Directly call the pipeline main function
        pipeline_main(upload_path, template_name, output_prefix)
        # Output files
        boxed_img = f"{output_prefix}_boxed.jpg"
        ocr_json = f"{output_prefix}_ocr.json"
        boxed_img_path = os.path.join(app.config['UPLOAD_FOLDER'], boxed_img)
        ocr_json_path = os.path.join(app.config['UPLOAD_FOLDER'], ocr_json)
        # Read OCR JSON
        ocr_data = {}
        if os.path.exists(ocr_json_path):
            import json
            with open(ocr_json_path, 'r') as f:
                ocr_data = json.load(f)
        return render_template('result.html', boxed_img=boxed_img, ocr_data=ocr_data)
    return render_template('index.html', templates=TEMPLATES)

@app.route('/output/<filename>')
def output_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(host='0.0.0.0', port=5006, debug=True) 