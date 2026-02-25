import os
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from utils.predictor import predictor

app = Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        if request.headers.get('Content-Type') == 'application/json':
             return jsonify({'error': 'No file part'}), 400
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        if request.headers.get('Content-Type') == 'application/json':
             return jsonify({'error': 'No selected file'}), 400
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get full result from professional predictor
        result = predictor.predict(filepath)
        
        if "error" in result:
             return result["error"], 500

        # Pass all fields to template
        return render_template('result.html', 
                               image_path=filepath,
                               disease=result['disease_name'],
                               crop=result['crop'],
                               risk_level=result['risk_level'],
                               confidence=result['confidence'],
                               confidence_score=result['confidence_score'],
                               confidence_level=result['confidence_level'],
                               confidence_class=result['confidence_class'],
                               description=result['description'],
                               causes=result['causes'],
                               treatment=result['treatment'],
                               prevention=result['prevention'])

    return redirect(request.url)

if __name__ == '__main__':
    app.run(debug=True)
