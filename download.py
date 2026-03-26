import kagglehub
def download():
    # Download latest version
    path = kagglehub.dataset_download("ishanpradhan95/industrial-pump-physics-grounded-digital-twin")
    print("Path to dataset files:", path)
    return path