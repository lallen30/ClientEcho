import os
import re
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
    timestamp = db.Column(db.Float)
    video_id = db.Column(db.Integer, db.ForeignKey('video.id'), nullable=False)
    image_path = db.Column(db.String(200))

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
        prompt = """
Analyze the following transcript segments and extract a list of issues described by the user.
For each issue, provide:
1. A brief summary of the issue (start with "Summary:")
2. The exact start timestamp where the issue is first mentioned (start with "Start:")
3. The exact end timestamp where the discussion of the issue concludes (start with "End:")

Format each issue as follows:
Issue:
Summary: [Brief description of the issue]
Start: [Start timestamp in seconds]
End: [End timestamp in seconds]

Segments:
"""

        for segment in segments:
            prompt += f"[{segment['start']:.2f} - {segment['end']:.2f}] {segment['text']}\n"

        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are an assistant that extracts issues from transcript segments with precise timestamps."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            temperature=0,
            timeout=60  # Set a 60-second timeout
        )

        issues = response['choices'][0]['message']['content']
        logging.info("Transcript analysis completed")
        return issues
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

def save_issue_data(video_id, issue_number, issue_summary, timestamp, video_path):
    try:
        logging.info(f"Saving issue data for video {video_id}, issue {issue_number}")
        issue_folder = os.path.join('projects', f'video_{video_id}', f'issue_{issue_number}')
        os.makedirs(issue_folder, exist_ok=True)

        # Save the issue summary
        with open(os.path.join(issue_folder, 'summary.txt'), 'w') as f:
            f.write(issue_summary)

        # Extract and save the frame
        image_path = os.path.join(issue_folder, 'image.jpg')
        extract_frame(video_path, timestamp, image_path)

        # Save issue to database
        issue = Issue(summary=issue_summary, timestamp=timestamp, video_id=video_id, image_path=image_path)
        db.session.add(issue)
        db.session.commit()
        logging.info(f"Issue data saved successfully for video {video_id}, issue {issue_number}")

    except Exception as e:
        logging.error(f"Error saving issue data: {e}")
        db.session.rollback()
        raise

def parse_issues(issues_text):
    issues = []
    current_issue = {}
    for line in issues_text.split('\n'):
        line = line.strip()
        if line.startswith("Issue:"):
            if current_issue:
                issues.append(current_issue)
            current_issue = {}
        elif line.startswith("Summary:"):
            current_issue['summary'] = line[8:].strip()
        elif line.startswith("Start:"):
            try:
                current_issue['start_timestamp'] = float(line[6:].strip())
            except ValueError:
                logging.warning(f"Invalid start timestamp format: {line}")
        elif line.startswith("End:"):
            try:
                current_issue['end_timestamp'] = float(line[4:].strip())
            except ValueError:
                logging.warning(f"Invalid end timestamp format: {line}")

    if current_issue:
        issues.append(current_issue)

    logging.info(f"Parsed {len(issues)} issues")
    for i, issue in enumerate(issues):
        logging.info(f"Issue {i+1}: {issue}")

    return issues

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
        issues_text = analyze_transcript(transcript)

        video.status = 'Processing Issues'
        db.session.commit()
        issues = parse_issues(issues_text)

        for i, issue in enumerate(issues, 1):
            summary = issue.get('summary', 'No summary provided')
            if 'start_timestamp' in issue:
                timestamp = issue['start_timestamp']
                save_issue_data(video.id, i, summary, timestamp, video_path)
            else:
                logging.warning(f"Issue {i} has no start timestamp. Skipping frame extraction.")
                # Save issue summary without extracting frame
                issue = Issue(summary=summary, video_id=video.id)
                db.session.add(issue)
        
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
    return send_from_directory('projects', filename)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0')
