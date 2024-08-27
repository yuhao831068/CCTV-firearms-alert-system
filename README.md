# CCTV-firearms-alert-system
This is an alert system that monitors an assigned directory and feeds any image newly saved in it into a multimodal model to detect if there is any firearms existing in it.
Once the model detects a firearm in the image, it will automatically send an email to the user.
This project uses SRCNN, BLIP-2-OPT-2.7b, VGG19 model for processing the image, which are also in this repository.
Also, there are comparisons of accuracy among using the self-defined multimodal model, VGG19 and a DNN model.
