"""
Advanced Facial Recognition System using InsightFace
Author: AI Assistant
Description: Autonomous person identification using facial embeddings and real-time webcam recognition
"""

import cv2
import numpy as np
import os
import glob
from typing import List, Tuple, Optional
# import insightface
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
# import matplotlib.pyplot as plt
import requests # for sending data to the server

class FaceRecognitionSystem:
    """
    A comprehensive facial recognition system using InsightFace buffalo_l model.
    Supports master embedding creation, single image recognition, and real-time webcam recognition.
    """
    
    def __init__(self, model_name: str = 'buffalo_l', threshold: float = 0.400):
        """
        Initialize the face recognition system.
        
        Args:
            model_name (str): InsightFace model name (default: 'buffalo_l')
            threshold (float): Similarity threshold for person recognition (default: 0.5)
        """
        self.threshold = threshold
        self.master_embedding = None
        self.face_analyzer = None
        
        # Initialize InsightFace
        self._initialize_insightface(model_name)
        
    def _initialize_insightface(self, model_name: str):
        """Initialize InsightFace with the specified model."""
        try:
            self.face_analyzer = FaceAnalysis(name=model_name)
            self.face_analyzer.prepare(ctx_id=0, det_size=(640, 640))
            print(f"✅ InsightFace initialized successfully with {model_name} model")
        except Exception as e:
            print(f"❌ Error initializing InsightFace: {e}")
            raise
    
    def _load_and_process_images(self, folder_path: str) -> List[np.ndarray]:
        """
        Load all images from a folder and return them as numpy arrays.
        
        Args:
            folder_path (str): Path to folder containing images
            
        Returns:
            List[np.ndarray]: List of image arrays
        """
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.webp']
        image_paths = []
        
        for ext in image_extensions:
            image_paths.extend(glob.glob(os.path.join(folder_path, ext)))
            image_paths.extend(glob.glob(os.path.join(folder_path, ext.upper())))
        
        images = []
        for img_path in image_paths:
            try:
                img = cv2.imread(img_path)
                if img is not None:
                    images.append(img)
                    print(f"📸 Loaded: {os.path.basename(img_path)}")
                else:
                    print(f"⚠️  Could not load: {img_path}")
            except Exception as e:
                print(f"❌ Error loading {img_path}: {e}")
        
        print(f"📊 Total images loaded: {len(images)}")
        return images
    
    def _extract_face_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """
        Extract face embedding from a single image.
        
        Args:
            image (np.ndarray): Input image
            
        Returns:
            Optional[np.ndarray]: Face embedding or None if no face detected
        """
        try:
            faces = self.face_analyzer.get(image)
            if len(faces) > 0:
                # Return embedding of the largest face (by area)
                largest_face = max(faces, key=lambda x: (x.bbox[2] - x.bbox[0]) * (x.bbox[3] - x.bbox[1]))
                return largest_face.embedding
            return None
        except Exception as e:
            print(f"❌ Error extracting embedding: {e}")
            return None
    
    def _compute_cosine_similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1 (np.ndarray): First embedding
            embedding2 (np.ndarray): Second embedding
            
        Returns:
            float: Cosine similarity score (-1 to 1)
        """
        # Normalize embeddings
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # Compute cosine similarity
        similarity = np.dot(embedding1, embedding2) / (norm1 * norm2)
        return float(similarity)
    
    def build_master_embedding(self, folder_path: str) -> bool:
        """
        Build master embedding from all images in a folder.
        
        Args:
            folder_path (str): Path to folder containing photos of the person
            
        Returns:
            bool: True if successful, False otherwise
        """
        print(f"🔍 Building master embedding from folder: {folder_path}")
        
        if not os.path.exists(folder_path):
            print(f"❌ Folder does not exist: {folder_path}")
            return False
        
        # Load all images
        images = self._load_and_process_images(folder_path)
        
        if len(images) == 0:
            print("❌ No images found in the folder")
            return False
        
        # Extract embeddings from all images
        embeddings = []
        successful_extractions = 0
        
        for i, img in enumerate(images):
            print(f"🔄 Processing image {i+1}/{len(images)}...")
            embedding = self._extract_face_embedding(img)
            
            if embedding is not None:
                embeddings.append(embedding)
                successful_extractions += 1
                print(f"✅ Face detected and embedding extracted")
            else:
                print(f"⚠️  No face detected in image {i+1}")
        
        if len(embeddings) == 0:
            print("❌ No faces detected in any images")
            return False
        
        # Compute average embedding
        self.master_embedding = np.mean(embeddings, axis=0)
        
        print(f"✅ Master embedding created successfully!")
        print(f"📊 Statistics:")
        print(f"   - Total images processed: {len(images)}")
        print(f"   - Successful face extractions: {successful_extractions}")
        print(f"   - Success rate: {successful_extractions/len(images)*100:.1f}%")
        
        return True
    
    def recognize_person(self, image_path: str) -> Tuple[bool, float, str]:
        """
        Recognize if a person in an image matches the master embedding.
        
        Args:
            image_path (str): Path to image to analyze
            
        Returns:
            Tuple[bool, float, str]: (is_person, similarity_score, message)
        """
        if self.master_embedding is None:
            return False, 0.0, "❌ Master embedding not built yet. Call build_master_embedding() first."
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            return False, 0.0, f"❌ Could not load image: {image_path}"
        
        # Extract face embedding
        embedding = self._extract_face_embedding(image)
        if embedding is None:
            return False, 0.0, "❌ No face detected in the image"
        
        # Compute similarity
        similarity = self._compute_cosine_similarity(embedding, self.master_embedding)
        
        # Determine if it's the person
        is_person = similarity >= self.threshold
        
        if is_person:
            message = f"This is the person ✅ (Similarity: {similarity:.3f})"
        else:
            message = f"Not the person ❌ (Similarity: {similarity:.3f})"
        
        return is_person, similarity, message
    
    def webcam_recognition(self):
        """
        Real-time face recognition using webcam.
        Shows bounding boxes, similarity scores, and labels.
        """
        if self.master_embedding is None:
            print("❌ Master embedding not built yet. Call build_master_embedding() first.")
            return
        
        # Initialize webcam
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("❌ Could not open webcam")
            return
        
        print("🎥 Starting webcam recognition...")
        print("Press 'q' to quit, 's' to save current frame")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Could not read from webcam")
                break
            
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Detect faces
            faces = self.face_analyzer.get(frame)

            # Compute frame center
            frame_height, frame_width = frame.shape[:2]
            frame_center_x = frame_width // 2

            # Keep only the highest-similarity face that passes threshold
            best_person = None  # (similarity, bbox)
            for face in faces:
                bbox = face.bbox.astype(int)
                x1, y1, x2, y2 = bbox
                embedding = face.embedding
                similarity = self._compute_cosine_similarity(embedding, self.master_embedding)
                if similarity >= self.threshold:
                    if best_person is None or similarity > best_person[0]:
                        best_person = (similarity, bbox)

            # Only draw a green box for the best matching Vijay Patil (if any)
            if best_person is not None:
                similarity, bbox = best_person
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0)
                label = "Vijay Patil ✅"
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                similarity_text = f"{label} ({similarity:.3f})"
                (text_width, text_height), _ = cv2.getTextSize(
                    similarity_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2
                )
                cv2.rectangle(
                    frame,
                    (x1, y1 - text_height - 10),
                    (x1 + text_width, y1),
                    color,
                    -1
                )
                cv2.putText(
                    frame,
                    similarity_text,
                    (x1, y1 - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

            if best_person is not None:
                bbox = best_person[1]
                x1, y1, x2, y2 = bbox
                face_center_x = (x1 + x2) // 2
                norm_offset = abs((face_center_x - frame_center_x) / (frame_width / 2))
                norm_offset = float(max(0.0, min(1.0, norm_offset)))
                center_threshold = 0.10
                if norm_offset <= center_threshold:
                    direction = "Center"
                else:
                    direction = "Left" if face_center_x < frame_center_x else "Right"
                cv2.putText(
                    frame,
                    f"{direction} {norm_offset:.2f}",
                    (x1, min(y2 + 25, frame_height - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )
            
            # Add instructions
            cv2.putText(
                frame,
                "Press 'q' to quit, 's' to save frame",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2
            )
            
            # Display frame
            cv2.imshow('Face Recognition', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save current frame
                filename = f"captured_frame_{len(glob.glob('captured_frame_*.jpg')) + 1}.jpg"
                cv2.imwrite(filename, frame)
                print(f"📸 Frame saved as {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("🎥 Webcam recognition stopped")


def main():
    """Main function demonstrating usage of the FaceRecognitionSystem."""
    
    # Initialize the system
    face_system = FaceRecognitionSystem(threshold=0.400)
    
    # Example usage
    print("=" * 60)
    print("🤖 Advanced Face Recognition System")
    print("=" * 60)
    
    # Build master embedding from person photos
    person_photos_folder = "person_photos"
    
    if os.path.exists(person_photos_folder):
        print(f"\n📁 Found photos folder: {person_photos_folder}")
        success = face_system.build_master_embedding(person_photos_folder)
        
        if success:
            print("\n🎯 Master embedding built successfully!")
            
            # Example: Recognize a single image
            test_image = "test_image.jpg"  # Replace with actual test image path
            if os.path.exists(test_image):
                print(f"\n🔍 Testing recognition on: {test_image}")
                is_person, similarity, message = face_system.recognize_person(test_image)
                print(f"Result: {message}")
            
            # Start webcam recognition
            print(f"\n🎥 Starting webcam recognition...")
            print(f"Threshold: {face_system.threshold}")
            face_system.webcam_recognition()
            
        else:
            print("❌ Failed to build master embedding")
    else:
        print(f"❌ Photos folder not found: {person_photos_folder}")
        print("Please create a folder named 'person_photos' and add photos of the person")
        print("Example structure:")
        print("person_photos/")
        print("├── photo1.jpg")
        print("├── photo2.jpg")
        print("└── ... (66 photos)")


if __name__ == "__main__":
    main()
