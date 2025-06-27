#!/usr/bin/env python3
"""
Fix dataset labels that contain invalid class indices.
For a 2-class dataset (nc=2), valid class indices are 0 and 1.
This script finds and fixes any labels with class index >= 2.
"""

import os
import glob
from pathlib import Path

def fix_dataset_labels(dataset_path, max_class_index=1, fix=False):
    """
    Scan and optionally fix invalid class labels in dataset.
    
    Args:
        dataset_path: Path to dataset folder
        max_class_index: Maximum valid class index (for 2 classes, this is 1)
        fix: If True, fix the invalid labels; if False, just report them
    """
    
    invalid_files = []
    total_fixed = 0
    
    # Find all label files
    label_patterns = [
        f"{dataset_path}/**/labels/*.txt",
        f"{dataset_path}/**/*.txt"
    ]
    
    label_files = []
    for pattern in label_patterns:
        label_files.extend(glob.glob(pattern, recursive=True))
    
    # Filter out non-label files
    label_files = [f for f in label_files if 'labels' in f or (
        os.path.basename(f).replace('.txt', '') + '.jpg' in str(f).replace('labels', 'images').replace('.txt', '.jpg')
    )]
    
    print(f"Scanning {len(label_files)} label files...")
    
    for label_file in label_files:
        try:
            with open(label_file, 'r') as f:
                lines = f.readlines()
            
            fixed_lines = []
            file_has_invalid = False
            
            for line_num, line in enumerate(lines, 1):
                line = line.strip()
                if not line:
                    fixed_lines.append(line + '\n')
                    continue
                
                parts = line.split()
                if len(parts) < 5:  # Invalid format
                    print(f"  Warning: Invalid format in {label_file}:{line_num}: {line}")
                    fixed_lines.append(line + '\n')
                    continue
                
                try:
                    class_id = int(float(parts[0]))  # Convert to int (handles "2.0" -> 2)
                    
                    if class_id > max_class_index:
                        print(f"  Found invalid class {class_id} in {label_file}:{line_num}")
                        file_has_invalid = True
                        
                        if fix:
                            # Fix strategy: map class 2 to class 1 (smoke -> smoke)
                            # You can change this logic based on your dataset
                            new_class_id = min(class_id, max_class_index)
                            parts[0] = str(new_class_id)
                            fixed_line = ' '.join(parts) + '\n'
                            fixed_lines.append(fixed_line)
                            total_fixed += 1
                            print(f"    Fixed: class {class_id} -> {new_class_id}")
                        else:
                            fixed_lines.append(line + '\n')
                    else:
                        fixed_lines.append(line + '\n')
                        
                except ValueError:
                    print(f"  Warning: Non-numeric class in {label_file}:{line_num}: {line}")
                    fixed_lines.append(line + '\n')
            
            if file_has_invalid:
                invalid_files.append(label_file)
                
                if fix:
                    # Backup original file
                    backup_file = label_file + '.backup'
                    if not os.path.exists(backup_file):
                        os.rename(label_file, backup_file)
                        print(f"  Backed up original to {backup_file}")
                    
                    # Write fixed file
                    with open(label_file, 'w') as f:
                        f.writelines(fixed_lines)
                    print(f"  Fixed {label_file}")
                    
        except Exception as e:
            print(f"Error processing {label_file}: {e}")
    
    print(f"\nSummary:")
    print(f"  Total label files scanned: {len(label_files)}")
    print(f"  Files with invalid labels: {len(invalid_files)}")
    if fix:
        print(f"  Total labels fixed: {total_fixed}")
        print(f"  Original files backed up with .backup extension")
    else:
        print(f"  Run with fix=True to fix the invalid labels")
    
    if invalid_files:
        print(f"\nFiles with invalid labels:")
        for f in invalid_files[:10]:  # Show first 10
            print(f"  {f}")
        if len(invalid_files) > 10:
            print(f"  ... and {len(invalid_files) - 10} more")
    
    return invalid_files, total_fixed

if __name__ == "__main__":
    dataset_path = "datasets_smokefire"
    
    print("=" * 60)
    print("SCANNING FOR INVALID LABELS (nc=2, valid classes: 0, 1)")
    print("=" * 60)
    
    # First scan without fixing
    invalid_files, _ = fix_dataset_labels(dataset_path, max_class_index=1, fix=False)
    
    if invalid_files:
        print("\n" + "=" * 60)
        print("FIXING INVALID LABELS")
        print("=" * 60)
        
        # Fix the invalid labels
        fix_dataset_labels(dataset_path, max_class_index=1, fix=True)
        
        print("\n" + "=" * 60)
        print("VERIFICATION SCAN")
        print("=" * 60)
        
        # Verify fix
        invalid_files_after, _ = fix_dataset_labels(dataset_path, max_class_index=1, fix=False)
        
        if not invalid_files_after:
            print("✅ All labels are now valid!")
        else:
            print(f"⚠️  Still {len(invalid_files_after)} files with invalid labels")
    else:
        print("✅ No invalid labels found!")
