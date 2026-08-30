from pathlib import Path

ALLOWED_EXTENSIONS = {'.pdf', '.docx'}


def uploading_document(file_path):
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"File Not Found: {file_path}")

    if path.suffix.lower() not in ALLOWED_EXTENSIONS:
        raise ValueError(f"Only pdf or docx files are supported!")

    return {
        "filename" : path.name,
        "file_path" : str(path),
        "file_type" : path.suffix.lower()
    }

print("Document is uploaded successfully!")