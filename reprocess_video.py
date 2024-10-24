from app import app, db, Video, process_video
import logging

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

with app.app_context():
    video = Video.query.filter_by(filename='test_video.mp4').first()
    if video:
        print(f"Reprocessing video: {video.filename}")
        video.status = 'Uploaded'  # Reset the status
        db.session.commit()
        process_video(video.id)
    else:
        print("Video not found")