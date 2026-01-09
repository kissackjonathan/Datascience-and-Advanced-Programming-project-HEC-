"""
Script pour convertir tous les fichiers .numbers en .csv
À exécuter sur Mac uniquement
"""

import sys
from pathlib import Path

try:
    from numbers_parser import Document
except ImportError:
    print("Installation de numbers-parser...")
    import subprocess

    subprocess.run([sys.executable, "-m", "pip", "install", "numbers-parser"])
    from numbers_parser import Document


def convert_numbers_to_csv(numbers_file: Path) -> Path:
    """Convertit un fichier .numbers en .csv"""
    try:
        # Lire le fichier .numbers
        doc = Document(numbers_file)

        # Prendre la première feuille
        sheets = doc.sheets
        if not sheets:
            print(f"  ⚠️  Pas de feuille trouvée dans {numbers_file.name}")
            return None

        table = sheets[0].tables[0]

        # Créer le fichier CSV
        csv_file = numbers_file.with_suffix(".csv")

        # Sauvegarder en CSV
        data = []
        for row in table.rows():
            data.append([cell.value for cell in row])

        import csv

        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerows(data)

        print(f"  ✓ Converti: {numbers_file.name} → {csv_file.name}")
        return csv_file

    except Exception as e:
        print(f"  ✗ Erreur avec {numbers_file.name}: {e}")
        return None


def main():
    # Chemin vers le dossier data/raw
    project_root = Path(__file__).parent
    raw_data_path = project_root / "data" / "raw"

    if not raw_data_path.exists():
        print(f"❌ Le dossier {raw_data_path} n'existe pas!")
        return

    # Trouver tous les fichiers .numbers
    numbers_files = list(raw_data_path.glob("*.numbers"))

    if not numbers_files:
        print("❌ Aucun fichier .numbers trouvé dans data/raw/")
        return

    print(f"\n{'='*70}")
    print(f"CONVERSION DE {len(numbers_files)} FICHIERS .NUMBERS EN .CSV")
    print(f"{'='*70}\n")

    converted = 0
    for numbers_file in numbers_files:
        csv_file = convert_numbers_to_csv(numbers_file)
        if csv_file:
            converted += 1

    print(f"\n{'='*70}")
    print(f"✓ CONVERSION TERMINÉE: {converted}/{len(numbers_files)} fichiers convertis")
    print(f"{'='*70}\n")

    if converted > 0:
        print("Vous pouvez maintenant:")
        print("1. Committer les fichiers CSV:")
        print("   git add data/raw/*.csv")
        print("   git commit -m 'Add CSV versions of data files'")
        print("   git push hec main")
        print("\n2. Ou les copier manuellement sur Windows\n")


if __name__ == "__main__":
    main()
