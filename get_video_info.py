from app import app, db, Video

with app.app_context():
    video = Video.query.filter_by(filename='test_video.mp4').first()
    if video:
        print(f"Video ID: {video.id}")
        print(f"Project ID: {video.project_id}")
        print(f"Filename: {video.filename}")
        print(f"Status: {video.status}")
    else:
        print("Video not found")