import glob
import os
import random
from pathlib import Path

from pdf2image import convert_from_path
from tqdm import tqdm

random.seed(42)


def convert_pdf_to_images(pdf_file: str, save_folder: str):
    """
    Convert each page of a pdf to a jpg image and save them in a folder.

    Args:
    - pdf_file (str): path to the pdf file
    - save_folder (str): path to the folder where the images will be saved

    """
    images = convert_from_path(pdf_file)

    for i, image in enumerate(images):
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
        image.save(os.path.join(save_folder, f"page_{i+1}.jpg"), "JPEG")


def convert_all_pdfs_to_images(path_to_folder: str, n_samples: int = 0):
    """
    Convert all pdfs in a folder and its subfolder to images and save them in a folder.
    It will sample n_samples pdf files in each subfolder, allowing to have granularity on the number of pdf files to convert.


    Args:
    - path_to_folder (str): path to the folder containing the pdf files
    - n_samples (int): number of pdf files to sample in each subfolder

    directory structure:
    - path_to_folder
        - subfolder1
            - pdf1
            - pdf2
            - ...
        - subfolder2
            - pdf1
            - pdf2
            - ...
        - ...

    """
    # take n_samples pdf files in each subfolder : I want to take 10 pdf files from each subfolder
    sub_dirs = [d for d in os.listdir(path_to_folder) if os.path.isdir(os.path.join(path_to_folder, d))]

    sampled_files = []

    for sub_dir in sub_dirs:
        pdf_files = glob.glob(os.path.join(path_to_folder, sub_dir, "*.pdf"))

        if (n_samples == 0) or (len(pdf_files) <= n_samples):
            print(f"Taking all pdf files in {sub_dir}")
            sampled_files.extend(pdf_files)

        else:
            print(f"Taking {n_samples} pdf files in {sub_dir}")
            sampled_files.extend(random.sample(pdf_files, n_samples))

    pdf_files = [str(file) for file in sampled_files]

    # Create an empty text file that will contain the file paths of the corrupted pdf files
    dirpath_corrupted = Path(path_to_folder) / "corrupted_pdf_files.txt"
    dirpath_corrupted.parent.mkdir(parents=True, exist_ok=True)

    with dirpath_corrupted.open("w") as f:
        with tqdm(total=len(pdf_files)) as pbar:
            for pdf_file in pdf_files:
                pbar.set_description(f"Processing {pdf_file}")
                save_folder = os.path.join("pages_extracted", *Path(pdf_file).parts[-2:])
                if not os.path.exists(os.path.join(path_to_folder, save_folder)):
                    try:
                        convert_pdf_to_images(pdf_file, os.path.join(path_to_folder, save_folder))
                    except Exception as e:
                        print(f"Error converting {pdf_file}: {e}")
                        f.write(pdf_file)
                        f.write("\n")
                pbar.update(1)
    return
