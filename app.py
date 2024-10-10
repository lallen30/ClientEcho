import os
import re
import json
from flask import Flask, request, render_template, jsonify, redirect, url_for, escape, send_from_directory
from werkzeug.utils import secure_filename
from pydub import AudioSegment
import whisper
import openai
from dotenv import load_dotenv
import subprocess
import logging
import threading
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy.exc import SQLAlchemyError
import time

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///clientecho.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)

# Set the upload folder to the Docker volume mount point
UPLOAD_FOLDER = '/app/secure_uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MB limit

# Set the projects folder
PROJECTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'projects')
os.makedirs(PROJECTS_FOLDER, exist_ok=True)

# Allowed extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Configure OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')

# Database models
class Project(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    videos = db.relationship('Video', backref='project', lazy=True)

class Video(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(100), nullable=False)
    project_id = db.Column(db.Integer, db.ForeignKey('project.id'), nullable=False)
    processed = db.Column(db.Boolean, default=False)
    status = db.Column(db.String(50), default='Uploaded')
    issues = db.relationship('Issue', backref='video', lazy=True)

class Issue(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    summary = db.Column(db.Text, nullable=False)
    start_timestamp = db.Column(db.Float)
    end_timestamp = db.Column(db.Float)
    video_id = db.Column(db.Integer, db.ForeignKey('video.id'), nullable=False)
    image_paths = db.Column(db.Text)  # Store multiple image paths as JSON

    def get_image_paths(self):
        return json.loads(self.image_paths) if self.image_paths else []

    def set_image_paths(self, paths):
        self.image_paths = json.dumps(paths)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_audio(video_path, audio_path):
    try:
        logging.info(f"Extracting audio from {video_path}")
        audio = AudioSegment.from_file(video_path)
        audio.export(audio_path, format="wav")
        logging.info(f"Audio extracted successfully to {audio_path}")
    except Exception as e:
        logging.error(f"Error extracting audio: {e}")
        raise

def transcribe_audio(audio_path):
    try:
        logging.info(f"Transcribing audio from {audio_path}")
        model = whisper.load_model('base')
        result = model.transcribe(audio_path)
        logging.info("Audio transcription completed")
        return result
    except Exception as e:
        logging.error(f"Error transcribing audio: {e}")
        raise

def analyze_transcript(transcript):
    try:
        logging.info("Analyzing transcript")
        segments = transcript['segments']
        reviews = []
        current_review = None

        for segment in segments:
            text = segment['text'].strip().lower()
            if "start review" in text:
                if current_review:
                    reviews.append(current_review)
                current_review = {
                    'start': segment['start'],
                    'text': [],
                    'end': None
                }
            elif "end review" in text and current_review:
                current_review['end'] = segment['end']
                current_review['text'] = ' '.join(current_review['text'])
                reviews.append(current_review)
                current_review = None
            elif current_review:
                current_review['text'].append(segment['text'])

        if current_review:
            reviews.append(current_review)

        return reviews

    except Exception as e:
        logging.error(f"Error analyzing transcript: {e}")
        raise

def extract_frame(video_path, timestamp, output_path):
    try:
        logging.info(f"Extracting frame at {timestamp} seconds from {video_path}")
        cmd = [
            'ffmpeg',
            '-y',  # Force overwrite without prompting
            '-ss', str(timestamp),
            '-i', video_path,
            '-vframes', '1',
            '-q:v', '2',
            '-update', '1',
            output_path
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logging.info(f"Frame extracted successfully to {output_path}")
        logging.info(f"ffmpeg stdout: {result.stdout}")
        logging.info(f"ffmpeg stderr: {result.stderr}")
    except subprocess.CalledProcessError as e:
        logging.error(f"Error extracting frame: {e}")
        logging.error(f"ffmpeg command: {' '.join(e.cmd)}")
        logging.error(f"ffmpeg return code: {e.returncode}")
        logging.error(f"ffmpeg stdout: {e.stdout}")
        logging.error(f"ffmpeg stderr: {e.stderr}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in extract_frame: {str(e)}")
        raise

def save_issue_data(video_id, issue_number, review_data, video_path):
    try:
        logging.info(f"Saving issue data for video {video_id}, issue {issue_number}")
        issue_folder = os.path.join(PROJECTS_FOLDER, f'video_{video_id}', f'issue_{issue_number}')
        os.makedirs(issue_folder, exist_ok=True)

        # Save the issue summary
        with open(os.path.join(issue_folder, 'summary.txt'), 'w') as f:
            f.write(review_data['text'])

        # Extract and save the frame
        image_filename = f'image_{issue_number}.jpg'
        image_path = os.path.join(issue_folder, image_filename)
        extract_frame(video_path, review_data['start'], image_path)

        # Save issue to database
        relative_image_path = os.path.join(f'video_{video_id}', f'issue_{issue_number}', image_filename)
        issue = Issue(
            summary=review_data['text'],
            start_timestamp=review_data['start'],
            end_timestamp=review_data['end'],
            video_id=video_id,
            image_paths=json.dumps([relative_image_path])
        )
        db.session.add(issue)
        db.session.commit()
        logging.info(f"Issue data saved successfully for video {video_id}, issue {issue_number}")
        logging.info(f"Image saved at: {image_path}")
        logging.info(f"Relative image path: {relative_image_path}")

    except Exception as e:
        logging.error(f"Error saving issue data: {e}")
        db.session.rollback()
        raise

def process_video(video_id):
    try:
        video = Video.query.get(video_id)
        if not video:
            logging.error(f"Video with id {video_id} not found")
            return

        logging.info(f"Processing video {video.filename} for project {video.project.name}")
        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{video.filename}_audio.wav')
        
        video.status = 'Extracting Audio'
        db.session.commit()
        extract_audio(video_path, audio_path)
        
        video.status = 'Transcribing'
        db.session.commit()
        transcript = transcribe_audio(audio_path)
        
        video.status = 'Analyzing Transcript'
        db.session.commit()
        reviews = analyze_transcript(transcript)

        video.status = 'Processing Issues'
        db.session.commit()

        for i, review in enumerate(reviews, 1):
            save_issue_data(video.id, i, review, video_path)
        
        video.processed = True
        video.status = 'Completed'
        db.session.commit()
        
        logging.info(f"Video processing completed for {video.filename}")
        
        # Clean up temporary files
        os.remove(audio_path)
    except Exception as e:
        logging.error(f"Error processing video: {e}")
        video.status = f'Error: {str(e)}'
        db.session.commit()
        db.session.rollback()
        raise
    finally:
        db.session.close()

def async_process_video(video_id):
    def run_in_app_context():
        with app.app_context():
            process_video(video_id)
    
    threading.Thread(target=run_in_app_context).start()

@app.route('/')
def home():
    projects = Project.query.all()
    return render_template('home.html', projects=projects)

@app.route('/project/new', methods=['POST'])
def new_project():
    project_name = escape(request.form['project_name'])
    project = Project(name=project_name)
    try:
        db.session.add(project)
        db.session.commit()
    except SQLAlchemyError as e:
        db.session.rollback()
        logging.error(f"Error creating new project: {e}")
        return jsonify({"error": "Failed to create project"}), 500
    finally:
        db.session.close()
    return redirect(url_for('home'))

@app.route('/project/<int:project_id>')
def project_detail(project_id):
    project = Project.query.get_or_404(project_id)
    return render_template('project_detail.html', project=project)

@app.route('/project/<int:project_id>/upload', methods=['POST'])
def upload_video(project_id):
    logging.info(f"Received upload request for project ID: {project_id}")
    project = Project.query.get_or_404(project_id)
    if 'video_file' not in request.files:
        logging.error("No file part in the request")
        return jsonify({"error": "No file part"}), 400
    file = request.files['video_file']
    if file.filename == '':
        logging.error("No selected file")
        return jsonify({"error": "No selected file"}), 400
    if file and allowed_file(file.filename):
        try:
            filename = secure_filename(file.filename)
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(video_path)
            logging.info(f"File {filename} saved to {video_path}")

            video = Video(filename=filename, project_id=project.id, status='Uploaded')
            db.session.add(video)
            db.session.commit()
            logging.info(f"Video record created for {filename}")

            async_process_video(video.id)

            return jsonify({"message": f"Video uploaded and processing started for project {project.name}. You will be notified when processing is complete."}), 202
        except Exception as e:
            db.session.rollback()
            logging.error(f"Error processing upload: {str(e)}")
            return jsonify({"error": f"An error occurred while processing the video: {str(e)}"}), 500
        finally:
            db.session.close()
    else:
        logging.error("Invalid file type")
        return jsonify({"error": "Invalid file type"}), 400

@app.route('/video/<int:video_id>/status')
def video_status(video_id):
    video = Video.query.get_or_404(video_id)
    return jsonify({"status": video.status, "processed": video.processed})

@app.route('/projects/<path:filename>')
def serve_project_file(filename):
    logging.info(f"Serving project file: {filename}")
    return send_from_directory(PROJECTS_FOLDER, filename)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
