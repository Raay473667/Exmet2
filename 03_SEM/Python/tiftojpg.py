from PIL import Image
import os

folder_path = "hornak"
folder_output_path = 'imgpdf'

# loop through all files in the folder
for file_name in os.listdir(folder_path):
    if file_name.endswith(".tif"):
        # open the tif file using Pillow
        image = Image.open(os.path.join(folder_path, file_name))

        # convert to JPEG format
        image = image.convert("RGB")
        jpg_file_name = os.path.splitext(file_name)[0] + ".jpg"
        image.save(os.path.join(folder_output_path, jpg_file_name), "JPEG", quality=100)

        # convert JPEG to PDF
        #pdf_file_name = os.path.splitext(file_name)[0] + ".pdf"
        #with open(os.path.join(folder_path, pdf_file_name), "wb") as f:
        #    f.write(img2pdf.convert(os.path.join(folder_path, jpg_file_name)))
