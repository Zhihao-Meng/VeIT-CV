import cv2


# Function to find the checkerboard pattern
def find_checkerboard(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Define the checkerboard size
    pattern_size = (6, 9)  # Change this according to your checkerboard

    # Find the checkerboard corners
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        # Draw the checkerboard corners
        cv2.drawChessboardCorners(image, pattern_size, corners, ret)
        return True, image
    else:
        return False, None


# Open video capture
cap = cv2.VideoCapture(2)  # Replace 'your_video_file.mp4' with your video file path

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Find checkerboard in the current frame
    found, annotated_frame = find_checkerboard(frame)

    if found:
        cv2.imshow('Checkerboard Detection', annotated_frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release video capture and close windows
cap.release()
cv2.destroyAllWindows()
