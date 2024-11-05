import os
# Add these lines
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'
os.environ['OAUTHLIB_RELAX_TOKEN_SCOPE'] = '1'
import re
import json
from flask import Flask, request, render_template, jsonify, redirect, url_for, escape, send_from_directory, session, flash
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
from requests_oauthlib import OAuth2Session
import requests
import pprint
from functools import wraps
import torch

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
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB limit

# Set the projects folder
PROJECTS_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'project-files')
os.makedirs(PROJECTS_FOLDER, exist_ok=True)

# Allowed extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv'}

# Configure OpenAI API
openai.api_key = os.getenv('OPENAI_API_KEY')
if not openai.api_key:
    logging.error("OpenAI API key not found in .env file. Title generation will be disabled.")
else:
    logging.info("OpenAI API key loaded from .env file.")

# Basecamp OAuth configuration
app.secret_key = os.urandom(24)
BASECAMP_ACCOUNT_ID = os.getenv('BASECAMP_ACCOUNT_ID')
CLIENT_ID = os.getenv('BASECAMP_CLIENT_ID')
CLIENT_SECRET = os.getenv('BASECAMP_CLIENT_SECRET')
REDIRECT_URI = os.getenv('BASECAMP_REDIRECT_URI')
authorization_base_url = "https://launchpad.37signals.com/authorization/new"
token_url = "https://launchpad.37signals.com/authorization/token"

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
    archived = db.Column(db.Boolean, default=False)

    def get_image_paths(self):
        return json.loads(self.image_paths) if self.image_paths else []

    def set_image_paths(self, paths):
        self.image_paths = json.dumps(paths)

# Add this function
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_audio(video_path, audio_path):
    try:
        logging.info(f"Extracting audio from {video_path}")
        
        # First get video information
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=start_time,duration,time_base,r_frame_rate',
            '-of', 'json',
            video_path
        ]
        
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        video_info = json.loads(probe_result.stdout)
        logging.info(f"Video info: {video_info}")
        
        # Extract audio with precise timing
        cmd = [
            'ffmpeg',
            '-y',  # Overwrite output file
            '-i', video_path,
            '-vn',  # Disable video
            '-acodec', 'pcm_s16le',  # Use PCM format
            '-ar', '44100',  # Set sample rate
            '-ac', '2',  # Set audio channels
            '-copyts',  # Copy timestamps
            '-start_at_zero',  # Start at zero timestamp
            '-map_metadata', '0',  # Copy metadata
            '-fflags', '+genpts',  # Generate presentation timestamps
            audio_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            logging.error(f"FFmpeg stderr: {result.stderr}")
            raise RuntimeError(f"FFmpeg audio extraction failed: {result.stderr}")
            
        logging.info(f"Audio extracted successfully to {audio_path}")
        
        # Verify audio timing
        audio_probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_entries', 'format=start_time,duration',
            '-of', 'json',
            audio_path
        ]
        
        audio_probe_result = subprocess.run(audio_probe_cmd, capture_output=True, text=True)
        audio_info = json.loads(audio_probe_result.stdout)
        logging.info(f"Extracted audio info: {audio_info}")
        
        return video_info, audio_info
        
    except Exception as e:
        logging.error(f"Error extracting audio: {str(e)}")
        raise

def transcribe_audio(audio_path):
    try:
        logging.info(f"Starting transcription of audio from {audio_path}")
        
        # Verify audio file exists and is readable
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
        # Load Whisper model with explicit device selection
        logging.info("Loading Whisper model...")
        try:
            device = "cuda" if torch.cuda.is_available() else "cpu"
            logging.info(f"Using device: {device}")
            model = whisper.load_model('base', device=device)
            logging.info("Whisper model loaded successfully")
        except Exception as e:
            logging.error(f"Error loading Whisper model: {e}")
            raise
        
        # Perform transcription
        logging.info("Starting transcription...")
        try:
            result = model.transcribe(
                audio_path,
                verbose=True,
                fp16=False
            )
            
            # Verify transcription result
            if not result or 'segments' not in result:
                raise ValueError("Transcription result is invalid or empty")
            
            # Log segments for debugging
            for segment in result['segments']:
                logging.info(f"Segment: {segment['start']:.3f}-{segment['end']:.3f}: {segment['text']}")
            
            return result
            
        except Exception as e:
            logging.error(f"Error during transcription: {e}")
            raise
            
    except Exception as e:
        logging.error(f"Error in transcribe_audio: {str(e)}")
        raise RuntimeError(f"Transcription failed: {str(e)}")

def analyze_transcript(transcript, video_info, audio_info):
    try:
        logging.info("Analyzing transcript")
        segments = transcript['segments']
        reviews = []
        current_review = None
        buffer_text = []
        
        # Log all segments for analysis
        logging.info("\nAll segments in transcript:")
        for i, segment in enumerate(segments):
            logging.info(f"Segment {i}: {segment['start']:.2f}-{segment['end']:.2f}: {segment['text']}")

        # Compile regex patterns for review markers
        start_review_pattern = re.compile(r'\bstart\s+review\b', re.IGNORECASE)
        stop_review_pattern = re.compile(r'\b(stop|end)\s+review\b', re.IGNORECASE)

        for i, segment in enumerate(segments):
            text = segment['text'].strip()
            lower_text = text.lower()
            
            # Detect "start review"
            if start_review_pattern.search(lower_text):
                start_idx = lower_text.find("start review")
                
                # Split into words and find word index of "start review"
                words = text.split()
                lower_words = lower_text.split()
                
                # Find which word is "start review"
                start_review_word_idx = -1
                for j, word_pair in enumerate(zip(words, lower_words)):
                    if "start" in word_pair[1] and j + 1 < len(lower_words) and "review" in lower_words[j + 1]:
                        start_review_word_idx = j
                        break
                
                # Calculate timing
                if start_review_word_idx >= 0:
                    segment_duration = segment['end'] - segment['start']
                    words_count = len(words)
                    
                    # Calculate time per word more accurately
                    time_per_word = segment_duration / words_count if words_count > 0 else 0
                    
                    # Calculate start time based on word position
                    base_time = segment['start'] + (start_review_word_idx * time_per_word)
                    
                    # Add offset for saying "start review" (1 second)
                    start_time = base_time + 1.0
                    
                    # Ensure we don't start before the actual phrase
                    if start_time < base_time + 0.8:  # Minimum time to say "start review"
                        start_time = base_time + 0.8
                    
                    # Add a small buffer to prevent early triggers
                    if start_time < segment['start'] + 0.5:
                        start_time = segment['start'] + 0.5
                    
                    logging.info(f"""
                    Found 'start review':
                    - Segment {i}
                    - Full text: "{text}"
                    - Word index: {start_review_word_idx} of {words_count}
                    - Segment start: {segment['start']:.2f}
                    - Base time: {base_time:.2f}
                    - Final start time: {start_time:.2f}
                    - Segment end: {segment['end']:.2f}
                    """)
                else:
                    start_time = segment['start'] + 0.8  # Default offset if word index not found
                
                # Close previous review if exists
                if current_review:
                    current_review['end'] = start_time
                    current_review['text'] = ' '.join(buffer_text)
                    if current_review['text'].strip():
                        reviews.append(current_review)
                    buffer_text = []
                
                # Start new review
                current_review = {
                    'start': start_time,
                    'text': [],
                    'end': None,
                    'segment_index': i,
                    'timing_details': {
                        'segment_start': segment['start'],
                        'word_index': start_review_word_idx,
                        'total_words': len(words),
                        'base_time': base_time
                    }
                }
                
                # Get remaining text after "start review"
                remaining_text = text[start_idx + len("start review"):].strip()
                remaining_text = re.sub(r'^[.\s]+', '', remaining_text)
                if remaining_text:
                    buffer_text.append(remaining_text)
            
            # Detect "stop review" or "end review"
            elif current_review and stop_review_pattern.search(lower_text):
                stop_idx = lower_text.find("stop review")
                if stop_idx == -1:
                    stop_idx = lower_text.find("end review")
                
                # Include any text before "stop review" or "end review" in buffer
                if stop_idx > 0:
                    text_before_stop = text[:stop_idx].strip()
                    if text_before_stop:
                        buffer_text.append(text_before_stop)
                
                # Finalize the current review entry
                current_review['end'] = segment['end']
                current_review['text'] = ' '.join(buffer_text)
                if current_review['text'].strip():
                    reviews.append(current_review)
                current_review = None
                buffer_text = []
            
            # Buffer text within an ongoing review session
            elif current_review:
                cleaned_text = text.strip()
                if cleaned_text:
                    buffer_text.append(cleaned_text)

        # Finalize any open review if transcript ended before closing
        if current_review:
            current_review['end'] = segments[-1]['end']
            current_review['text'] = ' '.join(buffer_text)
            if current_review['text'].strip():
                reviews.append(current_review)

        # Process reviews and generate titles
        processed_reviews = []
        for review in reviews:
            if not review['text'].strip():
                continue
            
            review['title'] = generate_title(review['text'])
            review['screenshot_time'] = review['start']
            
            logging.info(f"""
            Processed review:
            - Title: {review['title']}
            - Start: {review['start']:.3f} ({int(review['start']//60)}:{int(review['start']%60):02d})
            - End: {review['end']:.3f} ({int(review['end']//60)}:{int(review['end']%60):02d})
            - Text: {review['text'][:100]}...
            """)
            
            processed_reviews.append(review)

        return processed_reviews

    except Exception as e:
        logging.error(f"Error analyzing transcript: {e}")
        raise

def extract_frame(video_path, timestamp, output_path):
    try:
        logging.info(f"Extracting frame at {timestamp} seconds from {video_path}")
        
        # Get detailed video information including FPS
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-select_streams', 'v:0',
            '-show_entries', 'stream=r_frame_rate,start_time,duration,avg_frame_rate',
            '-of', 'json',
            video_path
        ]
        
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        video_info = json.loads(probe_result.stdout)
        logging.info(f"Video frame info: {video_info}")
        
        # Calculate FPS from the frame rate fraction (e.g., "30000/1001" -> 29.97)
        fps_str = video_info['streams'][0].get('r_frame_rate', '30/1')
        fps_num, fps_den = map(int, fps_str.split('/'))
        fps = fps_num / fps_den
        
        # Calculate the nearest frame boundary
        video_start = float(video_info['streams'][0].get('start_time', '0'))
        frame_duration = 1.0 / fps
        adjusted_timestamp = timestamp - video_start
        
        # Round to nearest frame boundary
        frame_number = round(adjusted_timestamp * fps)
        frame_perfect_timestamp = frame_number * frame_duration
        
        logging.info(f"""
        Frame calculation:
        - FPS: {fps}
        - Frame duration: {frame_duration}
        - Original timestamp: {timestamp}
        - Video start time: {video_start}
        - Adjusted timestamp: {adjusted_timestamp}
        - Frame number: {frame_number}
        - Frame-perfect timestamp: {frame_perfect_timestamp}
        """)
        
        cmd = [
            'ffmpeg',
            '-y',  # Force overwrite
            '-ss', str(frame_perfect_timestamp),  # Seek to frame-perfect timestamp
            '-i', video_path,
            '-vf', f'fps={fps}',  # Force input FPS
            '-vframes', '1',
            '-q:v', '2',
            '-update', '1',
            output_path
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        logging.info(f"Frame extracted successfully to {output_path}")
        logging.info(f"Command output: {result.stdout}")
        logging.info(f"Command stderr: {result.stderr}")
        
    except subprocess.CalledProcessError as e:
        logging.error(f"Error extracting frame: {e}")
        logging.error(f"Command: {' '.join(e.cmd)}")
        logging.error(f"Return code: {e.returncode}")
        logging.error(f"Output: {e.stdout}")
        logging.error(f"Error: {e.stderr}")
        raise
    except Exception as e:
        logging.error(f"Unexpected error in extract_frame: {str(e)}")
        raise

def generate_title(summary):
    try:
        logging.info(f"Generating title for summary: {summary[:100]}...")  # Log first 100 chars of summary
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text in 5 words."},
                {"role": "user", "content": f"Summarize the following in 5 words:\n\n{summary}"}
            ],
            max_tokens=20,
            n=1,
            temperature=0.7,
        )
        title = response.choices[0].message['content'].strip()
        logging.info(f"Generated title: {title}")
        if not title:
            raise ValueError("OpenAI returned an empty title")
        return title
    except Exception as e:
        logging.error(f"Error generating title with OpenAI: {e}")
        return None

def save_issue_data(video_id, issue_number, review_data, video_path):
    try:
        logging.info(f"Saving issue data for video {video_id}, issue {issue_number}")
        logging.info(f"Review data: Start={review_data['start']:.2f}, End={review_data['end']:.2f}, Title={review_data['title']}")
        
        issue_folder = os.path.join(PROJECTS_FOLDER, f'video_{video_id}', f'issue_{issue_number}')
        os.makedirs(issue_folder, exist_ok=True)

        # Use the title from review data
        title = review_data['title']
        
        # Prepare the issue summary - removed the duplicate text
        issue_summary = (
            f"{title}\n\n"
            f"Timestamp: {int(review_data['start']//60)}:{int(review_data['start']%60):02d}\n\n"
            f"{review_data['text']}"
        )

        # Save the issue summary
        with open(os.path.join(issue_folder, 'summary.txt'), 'w') as f:
            f.write(issue_summary)

        # Extract and save the frame at the screenshot time
        image_filename = f'image_{issue_number}.jpg'
        image_path = os.path.join(issue_folder, image_filename)
        
        # Use the exact start time for screenshot
        screenshot_time = review_data.get('screenshot_time', review_data['start'])
        extract_frame(video_path, screenshot_time, image_path)

        # Save issue to database
        relative_image_path = os.path.join(f'video_{video_id}', f'issue_{issue_number}', image_filename)
        issue = Issue(
            summary=issue_summary,
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

        video_path = os.path.join(app.config['UPLOAD_FOLDER'], video.filename)
        audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f'{video.filename}_audio.wav')
        
        try:
            video.status = 'Extracting Audio'
            db.session.commit()
            video_info, audio_info = extract_audio(video_path, audio_path)
            
            video.status = 'Transcribing'
            db.session.commit()
            transcript = transcribe_audio(audio_path)
            
            video.status = 'Analyzing Transcript'
            db.session.commit()
            reviews = analyze_transcript(transcript, video_info, audio_info)
            
            video.status = 'Processing Issues'
            db.session.commit()

            for i, review in enumerate(reviews, 1):
                save_issue_data(video.id, i, review, video_path)
            
            video.processed = True
            video.status = 'Completed'
            db.session.commit()
            
            logging.info(f"Video processing completed for {video.filename}")
            
        except Exception as e:
            logging.error(f"Error processing video: {e}")
            video.status = f'Error: {str(e)}'
            db.session.commit()
            raise
        finally:
            # Clean up temporary files
            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception as e:
                logging.error(f"Error cleaning up audio file: {e}")
                
    except Exception as e:
        logging.error(f"Error in process_video: {e}")
        if video:
            video.status = f'Error: {str(e)}'
            db.session.commit()
    finally:
        db.session.close()

def async_process_video(video_id):
    def run_in_app_context():
        with app.app_context():
            process_video(video_id)
    
    threading.Thread(target=run_in_app_context).start()

# ... (rest of your imports and configurations)

# ... (your existing route functions)


def get_projects(token):
    logging.debug(f"Attempting to fetch Basecamp projects for account {BASECAMP_ACCOUNT_ID}")
    try:
        oauth = OAuth2Session(CLIENT_ID, token=token)
        url = f'https://3.basecampapi.com/{BASECAMP_ACCOUNT_ID}/projects.json'
        all_projects = []
        
        while url:
            response = oauth.get(url)
            logging.debug(f"Basecamp API response status: {response.status_code}")
            
            if response.status_code == 200:
                projects = response.json()
                all_projects.extend(projects)
                
                # Check for next page
                link_header = response.headers.get('Link', '')
                if 'rel="next"' in link_header:
                    url = link_header.split(';')[0].strip('<>')
                else:
                    url = None
            else:
                logging.error(f"Failed to fetch projects: {response.text}")
                return None
        
        logging.debug(f"Fetched {len(all_projects)} projects")
        return all_projects
    except Exception as e:
        logging.exception(f"An error occurred while fetching projects: {str(e)}")
        return None

def get_todo_lists(token, project_id):
    logging.debug(f"Attempting to fetch todo lists for project {project_id}")
    logging.debug(f"Token: {token}")
    logging.debug(f"Account ID: {BASECAMP_ACCOUNT_ID}")
    logging.debug(f"Client ID: {CLIENT_ID}")
    try:
        oauth = OAuth2Session(CLIENT_ID, token=token)
        headers = {
            'User-Agent': 'YourAppName (yourname@example.com)'
        }

        # First, fetch the project details
        project_url = f'https://3.basecampapi.com/{BASECAMP_ACCOUNT_ID}/projects/{project_id}.json'
        project_response = oauth.get(project_url, headers=headers)
        
        logging.debug(f"Project API response status: {project_response.status_code}")
        logging.debug(f"Project API response content: {project_response.text[:1000]}...")  # Log first 1000 characters
        
        if project_response.status_code != 200:
            logging.error(f"Failed to fetch project details: {project_response.text}")
            return None

        project_data = project_response.json()
        
        # Find the todoset in the project details
        todoset = next((dock for dock in project_data.get('dock', []) if dock['name'] == 'todoset'), None)
        
        if not todoset:
            logging.error("Todoset not found in project details")
            return None

        # Now fetch the todoset details
        todoset_response = oauth.get(todoset['url'], headers=headers)
        
        logging.debug(f"Todo lists API response status: {todoset_response.status_code}")
        logging.debug(f"Todo lists API response headers: {todoset_response.headers}")
        logging.debug(f"Todo lists API response content: {todoset_response.text[:1000]}...")  # Log first 1000 characters
        
        if todoset_response.status_code == 200:
            todoset_data = todoset_response.json()
            # The actual todo lists are not in this response, we need to fetch them separately
            todo_lists_url = todoset_data.get('todolists_url')
            if todo_lists_url:
                todo_lists_response = oauth.get(todo_lists_url, headers=headers)
                if todo_lists_response.status_code == 200:
                    todo_lists = todo_lists_response.json()
                    return [{'id': list['id'], 'name': list['name']} for list in todo_lists]
                else:
                    logging.error(f"Failed to fetch todo lists: {todo_lists_response.text}")
                    return None
            else:
                logging.error("Todo lists URL not found in todoset data")
                return None
        else:
            logging.error(f"Failed to fetch todoset: {todoset_response.text}")
            return None
    except Exception as e:
        logging.exception(f"An error occurred while fetching todo lists: {str(e)}")
        return None

def get_todos(token, project_id, todolist_id):
    logging.debug(f"Attempting to fetch todos for project {project_id}, todolist {todolist_id}")
    try:
        oauth = OAuth2Session(CLIENT_ID, token=token)
        headers = {
            'User-Agent': 'YourAppName (yourname@example.com)'
        }
        url = f'https://3.basecampapi.com/{BASECAMP_ACCOUNT_ID}/buckets/{project_id}/todolists/{todolist_id}/todos.json'
        logging.debug(f"Requesting URL: {url}")
        response = oauth.get(url, headers=headers)
        
        logging.debug(f"Todos API response status: {response.status_code}")
        logging.debug(f"Todos API response headers: {response.headers}")
        logging.debug(f"Todos API response content: {response.text[:1000]}...")  # Log first 1000 characters
        
        if response.status_code == 200:
            todos = response.json()
            return [{'id': todo['id'], 'title': todo['content']} for todo in todos]
        else:
            logging.error(f"Failed to fetch todos: {response.text}")
            return None
    except Exception as e:
        logging.exception(f"An error occurred while fetching todos: {str(e)}")
        return None

def create_todo(token, project_id, todolist_id, title, notes):
    logging.debug(f"Attempting to create a new todo in project {project_id}, todolist {todolist_id}")
    logging.debug(f"Title: {title}")
    logging.debug(f"Notes: {notes}")
    try:
        oauth = OAuth2Session(CLIENT_ID, token=token)
        headers = {
            'User-Agent': 'YourAppName (yourname@example.com)',
            'Content-Type': 'application/json'
        }
        url = f'https://3.basecampapi.com/{BASECAMP_ACCOUNT_ID}/buckets/{project_id}/todolists/{todolist_id}/todos.json'
        data = {
            'content': title,
            'description': notes
        }
        logging.debug(f"Sending POST request to URL: {url}")
        logging.debug(f"Request data: {data}")
        logging.debug(f"Headers: {headers}")
        response = oauth.post(url, headers=headers, json=data)
        
        logging.debug(f"Create todo API response status: {response.status_code}")
        logging.debug(f"Create todo API response content: {response.text[:1000]}...")  # Log first 1000 characters
        
        response.raise_for_status()  # This will raise an HTTPError for bad responses
        
        if response.status_code == 201:
            return response.json()
        else:
            logging.error(f"Failed to create todo: {response.text}")
            return None
    except Exception as e:
        logging.exception(f"An error occurred while creating todo: {str(e)}")
        raise

def get_access_token():
    return "your_access_token_here"  # Replace with your actual access token

@app.errorhandler(404)
def not_found_error(error):
    app.logger.error(f"404 error: {str(error)}")

def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'oauth_token' not in session:
            return redirect(url_for('login', next=request.url))
        return f(*args, **kwargs)
    return decorated_function

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/home')
@login_required
def home():
    token = session.get('oauth_token')
    basecamp_projects = get_projects(token)
    local_projects = Project.query.all()
    return render_template('home.html', basecamp_projects=basecamp_projects, local_projects=local_projects)

@app.route('/login')
def login():
    oauth = OAuth2Session(CLIENT_ID, redirect_uri=REDIRECT_URI)
    authorization_url, state = oauth.authorization_url(
        authorization_base_url,
        type='web_server'
    )
    session['oauth_state'] = state
    return redirect(authorization_url)

@app.route('/oauth/callback')
def callback():
    oauth = OAuth2Session(CLIENT_ID, state=session['oauth_state'], redirect_uri=REDIRECT_URI)
    token = oauth.fetch_token(
        token_url,
        client_secret=CLIENT_SECRET,
        authorization_response=request.url,
        include_client_id=True,
        type='web_server'
    )
    session['oauth_token'] = token
    next_url = request.args.get('next') or url_for('home')
    return redirect(next_url)

@app.route('/projects')
@login_required
def projects():
    token = session.get('oauth_token')
    projects = get_projects(token)
    if projects:
        return render_template('projects.html', projects=projects)
    else:
        return "Failed to retrieve projects", 500

@app.route('/todos')
@login_required
def todos():
    token = session.get('oauth_token')
    projects = get_projects(token)
    if projects:
        return render_template('todos.html', projects=projects)
    else:
        return "Failed to retrieve projects", 500

@app.route('/todo_lists/<project_id>')
def todo_lists(project_id):
    token = session.get('oauth_token')
    if not token:
        logging.error("No OAuth token found in session")
        return jsonify({"error": "No OAuth token found. Please re-authenticate."}), 401
    
    todo_lists = get_todo_lists(token, project_id)
    if todo_lists is None:
        return jsonify({"error": "Failed to fetch todo lists. The project might not exist, you might not have access to it, or there was an API error. Please check the server logs for more details."}), 500
    return jsonify(todo_lists)

@app.route('/todos/<project_id>/<todolist_id>')
def get_todos_route(project_id, todolist_id):
    logging.debug(f"get_todos_route accessed with project_id: {project_id}, todolist_id: {todolist_id}")
    token = session.get('oauth_token')
    if not token:
        logging.error("No OAuth token found in session")
        return jsonify({"error": "No OAuth token found. Please re-authenticate."}), 401
    
    try:
        todos = get_todos(token, project_id, todolist_id)
        if todos is None:
            logging.error(f"Failed to fetch todos for project {project_id}, todolist {todolist_id}")
            return jsonify({"error": "Failed to fetch todos. Please check the server logs for more details."}), 500
        logging.debug(f"Successfully fetched {len(todos)} todos")
        return jsonify(todos)
    except Exception as e:
        logging.exception(f"An unexpected error occurred while fetching todos: {str(e)}")
        return jsonify({"error": "An unexpected error occurred. Please check the server logs for more details."}), 500

def create_todo(token, project_id, todolist_id, title, notes):
    logging.debug(f"Attempting to create a new todo in project {project_id}, todolist {todolist_id}")
    logging.debug(f"Title: {title}")
    logging.debug(f"Notes: {notes}")
    try:
        oauth = OAuth2Session(CLIENT_ID, token=token)
        headers = {
            'User-Agent': 'YourAppName (yourname@example.com)',
            'Content-Type': 'application/json'
        }
        url = f'https://3.basecampapi.com/{BASECAMP_ACCOUNT_ID}/buckets/{project_id}/todolists/{todolist_id}/todos.json'
        data = {
            'content': title,
            'description': notes
        }
        logging.debug(f"Sending POST request to URL: {url}")
        logging.debug(f"Request data: {data}")
        logging.debug(f"Headers: {headers}")
        response = oauth.post(url, headers=headers, json=data)
        
        logging.debug(f"Create todo API response status: {response.status_code}")
        logging.debug(f"Create todo API response content: {response.text[:1000]}...")  # Log first 1000 characters
        
        response.raise_for_status()  # This will raise an HTTPError for bad responses
        
        if response.status_code == 201:
            return response.json()
        else:
            logging.error(f"Failed to create todo: {response.text}")
            return None
    except Exception as e:
        logging.exception(f"An error occurred while creating todo: {str(e)}")
        raise

@app.route('/create_todolist', methods=['POST'])
@login_required
def create_todolist():
    token = session.get('oauth_token')
    if not token:
        return jsonify({"error": "No OAuth token found. Please re-authenticate."}), 401
    
    data = request.json
    project_id = data.get('projectId')
    name = data.get('name')
    description = data.get('description', '')
    
    if not project_id or not name:
        return jsonify({"error": "Missing required fields"}), 400
    
    try:
        oauth = OAuth2Session(CLIENT_ID, token=token)
        headers = {
            'User-Agent': 'YourAppName (yourname@example.com)',
            'Content-Type': 'application/json'
        }
        
        # First, get the todoset ID for the project
        project_url = f'https://3.basecampapi.com/{BASECAMP_ACCOUNT_ID}/projects/{project_id}.json'
        project_response = oauth.get(project_url, headers=headers)
        project_response.raise_for_status()
        project_data = project_response.json()
        
        todoset = next((dock for dock in project_data.get('dock', []) if dock['name'] == 'todoset'), None)
        if not todoset:
            return jsonify({"error": "Todoset not found in project"}), 404
        
        todoset_id = todoset['id']
        
        # Now create the todo list using the correct URL structure
        create_todolist_url = f'https://3.basecampapi.com/{BASECAMP_ACCOUNT_ID}/buckets/{project_id}/todosets/{todoset_id}/todolists.json'
        
        data = {
            'name': name,
            'description': description
        }
        
        logging.info(f"Sending POST request to create todolist: URL={create_todolist_url}, Data={data}")
        response = oauth.post(create_todolist_url, headers=headers, json=data)
        
        logging.info(f"Received response: Status={response.status_code}, Content={response.text[:1000]}")
        response.raise_for_status()
        
        if response.status_code == 201:
            new_todolist = response.json()
            return jsonify({
                "id": new_todolist['id'],
                "name": new_todolist['name'],
                "message": "Todo list created successfully!"
            }), 201
        else:
            return jsonify({"error": "Failed to create todo list"}), 500
    except requests.exceptions.RequestException as e:
        logging.error(f"Request error creating todo list: {str(e)}")
        return jsonify({"error": f"An error occurred while creating the todo list: {str(e)}"}), 500
    except Exception as e:
        logging.error(f"Unexpected error creating todo list: {str(e)}")
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

@app.route('/create_todo', methods=['POST'])
def handle_create_todo():
    token = session.get('oauth_token')
    if not token:
        return jsonify({"error": "No OAuth token found. Please re-authenticate."}), 401
    
    data = request.json
    project_id = data.get('projectId')
    todolist_id = data.get('todoListId')
    title = data.get('title')
    notes = data.get('notes')
    
    logging.debug(f"Received create todo request: project_id={project_id}, todolist_id={todolist_id}, title={title}")
    
    if not project_id or not todolist_id or not title:
        return jsonify({"error": "Missing required fields"}), 400
    
    try:
        result = create_todo(token, project_id, todolist_id, title, notes)
        if result:
            return jsonify({"message": "Todo created successfully", "todo": result}), 201
        else:
            return jsonify({"error": "Failed to create todo. The API returned no result."}), 500
    except Exception as e:
        logging.error(f"Error creating todo: {str(e)}")
        return jsonify({"error": f"An error occurred while creating the todo: {str(e)}"}), 500

@app.route('/upload_attachment/<project_id>', methods=['POST'])
def upload_attachment(project_id):
    token = session.get('oauth_token')
    if not token:
        return jsonify({"error": "No OAuth token found. Please re-authenticate."}), 401

    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400

    if file:
        filename = secure_filename(file.filename)
        upload_url = f'https://3.basecampapi.com/{BASECAMP_ACCOUNT_ID}/attachments.json'
        
        headers = {
            'Authorization': f'Bearer {token["access_token"]}',
            'Content-Type': file.content_type
        }

        params = {
            'name': filename
        }

        try:
            response = requests.post(upload_url, headers=headers, params=params, data=file)
            response.raise_for_status()
            attachment_data = response.json()
            return jsonify({"attachable_sgid": attachment_data['attachable_sgid']}), 200
        except requests.RequestException as e:
            logging.error(f"Error uploading attachment: {str(e)}")
            return jsonify({"error": str(e)}), 500

@app.route('/project/new', methods=['POST'])
@login_required
def new_project():
    data = request.json
    project_id = data.get('project_id')
    project_name = data.get('project_name')
    
    logging.info(f"Received request to create new project: {project_name} (ID: {project_id})")
    
    # Check if project already exists
    existing_project = Project.query.filter_by(name=project_name).first()
    if existing_project:
        return jsonify({"redirect": url_for('project_detail', project_id=existing_project.id)})
    
    project = Project(name=project_name)
    try:
        logging.info("Attempting to add project to database")
        db.session.add(project)
        logging.info("Attempting to commit changes to database")
        db.session.commit()
        logging.info(f"New project created: {project_name}")
        return jsonify({"redirect": url_for('project_detail', project_id=project.id)})
    except SQLAlchemyError as e:
        db.session.rollback()
        logging.error(f"SQLAlchemy error creating new project: {str(e)}")
        return jsonify({"error": 'Failed to create project. Please try again.'})
    except Exception as e:
        db.session.rollback()
        logging.error(f"Unexpected error creating new project: {str(e)}")
        return jsonify({"error": 'An unexpected error occurred. Please try again.'})
    finally:
        db.session.close()

@app.route('/project/<int:project_id>')
@login_required
def project_detail(project_id):
    project = Project.query.get_or_404(project_id)
    token = session.get('oauth_token')
    basecamp_projects = get_projects(token)
    
    # Find the corresponding Basecamp project
    basecamp_project = next((p for p in basecamp_projects if p['name'] == project.name), None)
    
    todo_lists = []
    basecamp_project_id = None
    if basecamp_project:
        basecamp_project_id = basecamp_project['id']
        todo_lists = get_todo_lists(token, basecamp_project_id)
    
    return render_template('project_detail.html', project=project, todo_lists=todo_lists, basecamp_project_id=basecamp_project_id)

@app.route('/project/<int:project_id>/upload', methods=['POST'])
def upload_video(project_id):
    try:
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
                
                # Save the uploaded file
                file.save(video_path)
                logging.info(f"File {filename} saved to {video_path}")
                
                # Verify the file is a valid video file
                probe_cmd = [
                    'ffmpeg',
                    '-v', 'error',
                    '-i', video_path,
                    '-f', 'null',
                    '-'
                ]
                
                try:
                    subprocess.run(probe_cmd, check=True, capture_output=True, text=True)
                except subprocess.CalledProcessError as e:
                    # Clean up the invalid file
                    os.remove(video_path)
                    logging.error(f"Invalid video file: {e.stderr}")
                    return jsonify({
                        "error": "The uploaded file appears to be corrupted or is not a valid video file. Please try uploading again."
                    }), 400

                # Verify video format before processing
                video_info = verify_video_format(video_path)
                
                # Create video record
                video = Video(
                    filename=filename,
                    project_id=project.id,
                    status='Uploaded',
                    metadata=json.dumps(video_info)
                )
                db.session.add(video)
                db.session.commit()
                
                # Start processing
                async_process_video(video.id)
                
                return jsonify({
                    "message": f"Video uploaded and processing started for project {project.name}. You will be notified when processing is complete."
                }), 202
                
            except Exception as e:
                # Clean up on error
                if os.path.exists(video_path):
                    os.remove(video_path)
                db.session.rollback()
                logging.error(f"Error processing upload: {str(e)}")
                return jsonify({
                    "error": f"An error occurred while processing the video: {str(e)}"
                }), 500
            finally:
                db.session.close()
        else:
            logging.error("Invalid file type")
            return jsonify({
                "error": f"Invalid file type. Allowed types are: {', '.join(ALLOWED_EXTENSIONS)}"
            }), 400
    except Exception as e:
        logging.error(f"Upload error: {e}")
        return jsonify({"error": str(e)}), 400


@app.route('/video/<int:video_id>/status')
def video_status(video_id):
    video = Video.query.get_or_404(video_id)
    return jsonify({"status": video.status, "processed": video.processed})

@app.route('/project-files/<path:filename>')
def serve_project_file(filename):
    logging.info(f"Serving project file: {filename}")
    return send_from_directory(PROJECTS_FOLDER, filename)
# ... (rest of your code)

def init_db():
    with app.app_context():
        db.create_all()

if __name__ == '__main__':
    init_db()
    app.run(debug=False, host='0.0.0.0')



# New routes from basecamp_projects.py

@app.route('/test')
def test():
    return jsonify({"message": "Test route working"}), 200

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def catch_all(path):
    print(f"Caught request for path: {path}")
    return jsonify({"error": f"Path not found: {path}"}), 404

@app.route('/upload_video')
def upload_video_page():
    token = session.get('oauth_token')
    if not token:
        return redirect(url_for('login'))
    projects = get_projects(token)
    return render_template('upload_video.html', projects=projects)

def init_db():
    with app.app_context():
        db.create_all()

if __name__ == '__main__':
    init_db()
    app.run(debug=False, host='0.0.0.0')

@app.route('/archive_issue/<int:issue_id>', methods=['POST'])
@login_required
def archive_issue(issue_id):
    issue = Issue.query.get_or_404(issue_id)
    try:
        # Option 1: Remove the issue from the database
        db.session.delete(issue)
        
        # Option 2: Archive the issue (if you want to keep a record)
        # issue.archived = True
        
        db.session.commit()
        return jsonify({"message": "Issue archived successfully"}), 200
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error archiving issue: {str(e)}")
        return jsonify({"error": "Failed to archive issue"}), 500

@app.route('/delete_issue/<int:issue_id>', methods=['POST'])
@login_required
def delete_issue(issue_id):
    issue = Issue.query.get_or_404(issue_id)
    try:
        db.session.delete(issue)
        db.session.commit()
        return jsonify({"message": "Issue deleted successfully"}), 200
    except Exception as e:
        db.session.rollback()
        logging.error(f"Error deleting issue: {str(e)}")
        return jsonify({"error": "Failed to delete issue"}), 500

def verify_video_format(video_path):
    try:
        probe_cmd = [
            'ffprobe',
            '-v', 'error',
            '-show_format',
            '-show_streams',
            '-of', 'json',
            video_path
        ]
        
        result = subprocess.run(probe_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise ValueError(f"Invalid video format: {result.stderr}")
            
        info = json.loads(result.stdout)
        logging.info(f"Video format info: {json.dumps(info, indent=2)}")
        
        # Check for potential timing issues
        format_info = info['format']
        if float(format_info.get('start_time', '0')) != 0:
            logging.warning(f"Video has non-zero start time: {format_info['start_time']}")
        
        return info
        
    except Exception as e:
        logging.error(f"Error verifying video format: {e}")
        raise

# Add this function to check permissions
def ensure_upload_permissions():
    try:
        if not os.path.exists(UPLOAD_FOLDER):
            os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        
        # Test write permissions with a temp file
        test_file = os.path.join(UPLOAD_FOLDER, '.test')
        try:
            with open(test_file, 'w') as f:
                f.write('test')
            os.remove(test_file)
        except Exception as e:
            logging.error(f"Upload directory is not writable: {e}")
            raise
            
        logging.info(f"Upload directory {UPLOAD_FOLDER} is writable")
    except Exception as e:
        logging.error(f"Error checking upload permissions: {e}")
        raise

# Add this to your app initialization
with app.app_context():
    ensure_upload_permissions()
