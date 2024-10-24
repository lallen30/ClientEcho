# ClientEcho

ClientEcho is a web application that efficiently manages and analyzes client feedback videos. It leverages advanced AI techniques to transcribe videos, extract issues, and provide timestamps for easy reference.

## Features

- Upload and process video files
- Real-time status updates for video processing
- Transcribe audio from videos using Whisper ASR
- Analyze transcripts to extract issues using OpenAI GPT-4
- Capture screenshots at issue timestamps
- Organize projects and videos
- User-friendly web interface with detailed processing information
- Improved error handling and logging

## Technologies Used

- Python
- Flask
- SQLAlchemy
- OpenAI GPT-4
- Whisper ASR
- FFmpeg
- pydub
- Docker

## Setup and Installation

1. Clone the repository:

   ```
   git clone https://github.com/lallen30/ClientEcho.git
   cd clientecho
   ```

2. Create a `.env` file in the project root and add your OpenAI API key:

   ```
   OPENAI_API_KEY=your_api_key_here
   ```

3. Build the Docker image:

   ```
   docker build -t clientecho . --no-cache
   ```

4. Create a directory on your host machine for secure uploads:

   ```
   mkdir -p ~/app
   ```

5. Run the Docker container:

   ```
   docker run -d -p 9001:5000 --env-file .env -v ~/app:/app/secure_uploads clientecho

   docker tag clientecho lallen30/clientecho:latest
   docker push lallen30/clientecho:latest
   ```

   On the server, run:

   ```
   docker pull lallen30/clientecho:latest
   docker run -d -p 9001:5000 --env-file .env -v ~/app:/app/secure_uploads clientecho
   ```

   Note: The `-v` option creates a volume mount. The path before the colon (`~/app`) is the directory on your host machine, and the path after the colon (`/app/secure_uploads`) is where it will be mounted inside the container.

6. Access the application at `http://localhost:9001`

## Usage

1. Create a new project on the home page.
2. Upload a video file for the project.
3. Monitor the real-time status updates as the video is processed (audio extraction, transcription, and analysis).
4. View the extracted issues and screenshots on the project detail page once processing is complete.

## Development

To set up the development environment:

1. Create a virtual environment:

   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. Install the required packages:

   ```
   pip install -r requirements.txt
   ```

3. Set up the database:

   ```
   python init_db.py
   ```

4. Run the Flask development server:
   ```
   flask run
   ```

## Error Handling and Logging

ClientEcho now features improved error handling and logging throughout the video processing pipeline. Errors are caught, logged, and displayed to the user, ensuring a smoother experience and easier troubleshooting.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License.
