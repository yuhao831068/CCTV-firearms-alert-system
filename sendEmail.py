import time
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
from dotenv import load_dotenv
import os

load_dotenv()


# send email
def send_email(image_path, description):
    sender_email = os.getenv("SENDER_EMAIL")
    receiver_email = os.getenv("RECEIVER_EMAIL")
    subject = "Warning! Firearms detected in your area!"
    smtp_server = "smtp.gmail.com"
    smtp_port = 587
    password = os.getenv("EMAIL_PASSWORD")

    msg = MIMEMultipart()
    msg['From'] = sender_email
    msg['To'] = receiver_email
    msg['Subject'] = subject

    # Add Email Content
    warning_message: str = "<p><strong>Please stay calm and contact authorities immediately if this is not a false alarm.</strong></p>"
    enlarged_text = f"<p style='font-size:20px'>{warning_message}</p>"  # HTML with enlarged text
    body = f"<p>Description: {description}</p><p>Time: {time.ctime()}</p><p>{warning_message}</p>"
    msg.attach(MIMEText(body, 'html'))

    # Add image
    with open(image_path, 'rb') as f:
        img = MIMEImage(f.read())
        msg.attach(img)

    # Send email
    with smtplib.SMTP(smtp_server, smtp_port) as server:
        server.starttls()
        server.login(sender_email, password)
        server.send_message(msg)
