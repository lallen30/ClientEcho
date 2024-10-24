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
            elif "stop review" in text and current_review:
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

def generate_title(summary):
    try:
        logging.info(f"Attempting to generate title for summary: {summary[:100]}...")  # Log first 100 chars of summary
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes text in 5 words."},
                {"role": "user", "content": f"Summarize the following in 8 words:\n\n{summary}"}
            ],
            max_tokens=20,
            n=1,
            temperature=0.7,
        )
        title = response.choices[0].message['content'].strip()
        logging.info(f"Raw OpenAI response: {response}")
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
        issue_folder = os.path.join(PROJECTS_FOLDER, f'video_{video_id}', f'issue_{issue_number}')
        os.makedirs(issue_folder, exist_ok=True)

        # Generate title using OpenAI
        title = generate_title(review_data['text'])
        logging.info(f"Generated title for issue {issue_number}: {title}")
        
        if title is None:
            # If title generation failed, use the first 8 words of the content
            title = ' '.join(review_data['text'].split()[:8])
            logging.warning(f"Using fallback title: {title}")

        # Prepare the issue summary with the generated title
        issue_summary = f"{title}\n\n{review_data['text']}"

        # Save the issue summary
        with open(os.path.join(issue_folder, 'summary.txt'), 'w') as f:
            f.write(issue_summary)

        # Extract and save the frame
        image_filename = f'image_{issue_number}.jpg'
        image_path = os.path.join(issue_folder, image_filename)
        extract_frame(video_path, review_data['start'], image_path)

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

