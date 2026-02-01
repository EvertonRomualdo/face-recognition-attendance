from app import application

id_camera = 0
cap = application.get_video_capture(id_camera)
application.execute_recognization(cap)