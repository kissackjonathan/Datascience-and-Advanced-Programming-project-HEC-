"""
Script to convert all .numbers files to .csv format.

This script scans the data/raw directory for Apple Numbers files and converts
them to CSV format for cross-platform compatibility. Run this script on Mac
before committing data to ensure Windows users can access the data.

Usage:
    python scripts/convert_data.py

Note:
    This script must be run on Mac as numbers-parser only works on macOS.
"""

import sys
from pathlib import Path

# Add parent directory to path to import from src
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data_loader import convert_numbers_to_csv  # noqa: E402


def main():
    """Convert all .numbers files in data/raw/ to CSV format."""
    # Path to data/raw directory
    raw_data_path = project_root / "data" / "raw"

    if not raw_data_path.exists():
        print(f"❌ Le dossier {raw_data_path} n'existe pas!")
        return

    # Find all .numbers files
    numbers_files = list(raw_data_path.glob("*.numbers"))

    if not numbers_files:
        print("❌ Aucun fichier .numbers trouvé dans data/raw/")
        return

    print(f"\n{'=' * 70}")
    print(f"CONVERSION DE {len(numbers_files)} FICHIERS .NUMBERS EN .CSV")
    print(f"{'=' * 70}\n")

    converted = 0
    for numbers_file in numbers_files:
        csv_file = convert_numbers_to_csv(numbers_file)
        if csv_file:
            print(f"  ✓ Converti: {numbers_file.name} → {csv_file.name}")
            converted += 1
        else:
            print(f"  ✗ Erreur avec {numbers_file.name}")

    print(f"\n{'=' * 70}")
    print(f"✓ CONVERSION TERMINÉE: {converted}/{len(numbers_files)} fichiers convertis")
    print(f"{'=' * 70}\n")

    if converted > 0:
        print("Vous pouvez maintenant:")
        print("1. Committer les fichiers CSV:")
        print("   git add data/raw/*.csv")
        print("   git commit -m 'Add CSV versions of data files'")
        print("   git push")
        print("\n2. Ou les copier manuellement sur Windows\n")


if __name__ == "__main__":
    main()
