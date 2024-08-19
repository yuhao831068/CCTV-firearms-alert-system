import os
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from modelsFunctions import run_multi_modal_model
from sendEmail import send_email


class ImageHandler(FileSystemEventHandler):
    def __init__(self, output_folder):
        super().__init__()
        self.image_counter = 0  # Set a counter
        self.output_folder = output_folder

    def on_created(self, event):
        if event.is_directory:  # if it is a directory that is created, do not execute it.
            return

        image_path = event.src_path
        print(f"Detected new image imported: {image_path}")

        # Use multimodal model to generate the predicted class, to save the image and to generate a description.
        predicted_class, processed_image, description = run_multi_modal_model(image_path)
        if predicted_class == 0:
            print("Firearm is detected!")
            file_name = f"Processed_image_{self.image_counter}.jpg"
            save_path = os.path.join(self.output_folder, file_name)

            processed_image.save(save_path)
            self.image_counter += 1
            send_email(save_path, description)
            print(f"Email sending: {description}")
        else:
            print("No firearm is detected.")


# function for start monitoring
def start_monitoring(folder_to_watch):
    output_folder = "Firearms Detected Images"
    event_handler = ImageHandler(output_folder)
    observer = Observer()
    observer.schedule(event_handler, path=folder_to_watch, recursive=False)
    observer.start()

    print(f"Folder is being monitored: {folder_to_watch}")
    try:
        while True:
            time.sleep(1)  # continue monitoring every 1 second.
    except KeyboardInterrupt:
        observer.stop()
    observer.join()