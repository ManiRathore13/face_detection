import cv2
import numpy as np

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

def detect_faces_in_image(image_path):
    """Detect faces in a static image"""
    # Read the image
    img = cv2.imread(image_path)
    
    # Convert to grayscale
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_classifier.detectMultiScale(
        gray_img, 
        scaleFactor=1.1, 
        minNeighbors=5,
        minSize=(40, 40)
    )
    
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
    
    print(f'Faces found: {len(faces)}')
    
    # Display the result
    cv2.imshow('Face Detection', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return img

def detect_bounding_box(vid):
    """Function to detect faces and draw bounding boxes"""
    gray_image = cv2.cvtColor(vid, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray_image, 1.1, 5, minSize=(40, 40))
    
    for (x, y, w, h) in faces:
        cv2.rectangle(vid, (x, y), (x + w, y + h), (0, 255, 0), 4)
    
    return faces

def real_time_face_detection():
    """Real-time face detection using webcam"""
    # Access the webcam
    video_capture = cv2.VideoCapture(0)
    
    if not video_capture.isOpened():
        print("Error: Could not open video capture")
        return
    
    print("Press 'q' to quit the face detection")
    
    while True:
        # Read frame from video capture
        result, video_frame = video_capture.read()
        
        if result is False:
            break
        
        # Apply face detection
        faces = detect_bounding_box(video_frame)
        
        # Display the number of faces detected
        cv2.putText(
            video_frame, 
            f'Faces: {len(faces)}', 
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            1, 
            (255, 255, 255), 
            2
        )
        
        # Display the processed frame
        cv2.imshow("Face Detection Project", video_frame)
        
        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    # Clean up
    video_capture.release()
    cv2.destroyAllWindows()

def detect_faces_with_landmarks(image_path=None):
    """Enhanced face detection with additional features"""
    if image_path:
        # For static image
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = face_classifier.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30)
        )
        
        for (x, y, w, h) in faces:
            # Draw rectangle around face
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Add face detection confidence text
            cv2.putText(
                img, 
                'Face Detected', 
                (x, y - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )
        
        cv2.imshow('Face Detection with Landmarks', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        
    else:
        # For real-time detection
        real_time_face_detection()

# Main execution
if __name__ == "__main__":
    print("Face Detection Project")
    print("1. Press '1' for image face detection")
    print("2. Press '2' for real-time face detection")
    
    choice = input("Enter your choice (1 or 2): ")
    
    if choice == '1':
        # For static image detection
        image_path = input("Enter image path (or press Enter to use webcam): ")
        if image_path.strip():
            detect_faces_in_image(image_path)
        else:
            print("No image path provided. Starting webcam...")
            real_time_face_detection()
    
    elif choice == '2':
        # For real-time detection
        real_time_face_detection()
    
    else:
        print("Invalid choice. Starting real-time detection...")
        real_time_face_detection()