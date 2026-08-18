import os
import zipfile
import shutil

def extract_dataset(zip_path: str = r"d:\Downloads\archive.zip", output_dir: str = r"d:\Cognizant\data\dataset"):
    if not os.path.exists(zip_path):
        print(f"[Error] Zip archive not found at: {zip_path}")
        return

    os.makedirs(output_dir, exist_ok=True)
    meta_dir = os.path.join(output_dir, "movie_metadata")
    annot_dir = os.path.join(output_dir, "rule_based_annotations")
    os.makedirs(meta_dir, exist_ok=True)
    os.makedirs(annot_dir, exist_ok=True)

    print(f"[Extractor] Opening archive: {zip_path}")
    with zipfile.ZipFile(zip_path, 'r') as zf:
        namelist = zf.namelist()
        print(f"[Extractor] Total files in archive: {len(namelist)}")

        extracted_count = 0

        for f in namelist:
            # 1. Movie metadata CSV
            if f.endswith("movie_meta_data.csv"):
                target = os.path.join(meta_dir, "movie_meta_data.csv")
                with zf.open(f) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                print(f"  [+] Extracted metadata CSV -> {target}")
                extracted_count += 1

            # 2. Rule based annotations JSON
            elif "rule_based_annotations" in f and f.endswith(".json"):
                filename = os.path.basename(f)
                target = os.path.join(annot_dir, filename)
                with zf.open(f) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted_count += 1
                if extracted_count % 500 == 0:
                    print(f"  [+] Extracted {extracted_count} annotation files...")

            # 3. Character genders pickle
            elif f.endswith("character_genders.pickle"):
                target = os.path.join(output_dir, "character_genders.pickle")
                with zf.open(f) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                print(f"  [+] Extracted character genders pickle -> {target}")
                extracted_count += 1

    print(f"\n[Extractor] Extraction complete! Total {extracted_count} files extracted to: {output_dir}")

if __name__ == "__main__":
    extract_dataset()
