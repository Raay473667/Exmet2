from PIL import Image
import os

folder_path = "hornak"
folder_output_path = 'imgpdf'

# loop through all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".tif"):
        # open the tif file using Pillow
        image = Image.open(os.path.join(folder_path, file_name))
        # convert to RGB mode
        if image.mode == 'I;16':
            image = image.convert('I')
        image = image.convert('RGB')
        # save the image as a pdf
        pdf_file_name = os.path.splitext(file_name)[0] + ".pdf"
        image.save(os.path.join(folder_output_path, pdf_file_name), "PDF", resolution=200.0)
