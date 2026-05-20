# main.py
from detector import BaggageDetectionSystem

if __name__ == "__main__":
    system = BaggageDetectionSystem()
    
    input_video = "sample2.mp4"
    output_video = "output_final-15-05-26.mp4"
    
    print(f"Starting baggage detection on {input_video}...")
    system.process_video(input_video, output_video)