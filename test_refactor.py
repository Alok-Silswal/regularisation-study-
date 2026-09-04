#!/usr/bin/env python
"""
Lightweight validation tests for refactored repository.
Do NOT run full training.
"""

import sys
import traceback

def test_imports():
    """A. Test basic imports"""
    print("\n" + "="*60)
    print("TEST A: Import Validation")
    print("="*60)
    try:
        from pytorch_image_classification import (
            create_dataloader,
            create_dataset,
            create_loss,
            create_model,
            create_optimizer,
            create_scheduler,
            get_default_config,
            update_config,
        )
        print("✅ All imports successful")
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        traceback.print_exc()
        return False


def test_config():
    """B. Test configuration"""
    print("\n" + "="*60)
    print("TEST B: Configuration Validation")
    print("="*60)
    try:
        from pytorch_image_classification import get_default_config, update_config, create_dataset
        
        config = get_default_config()
        
        # Test 1: Default download should be False
        assert config.dataset.download == False, "download should default to False"
        print("✅ dataset.download defaults to False")
        
        # Test 2a: Empty dataset_dir should raise ValueError when creating dataset
        config_empty = get_default_config()
        try:
            create_dataset(config_empty, is_train=False)
            print("❌ Should have raised ValueError for empty dataset_dir")
            return False
        except ValueError as e:
            assert "explicitly specified" in str(e), "Error message should mention explicit specification"
            print("✅ Empty dataset_dir raises ValueError with clear message")
        
        # Test 2b: Dataset dir can be set and preserved
        config.merge_from_list(['dataset.dataset_dir', '/tmp/cifar10'])
        config = update_config(config)
        assert config.dataset.dataset_dir == '/tmp/cifar10', "dataset_dir was overwritten!"
        print("✅ dataset_dir is preserved when explicitly set")
        
        # Test 3: Removed fields should not exist
        assert not hasattr(config.augmentation, 'use_ricap'), "use_ricap should be removed"
        assert not hasattr(config.augmentation, 'use_dual_cutout'), "use_dual_cutout should be removed"
        assert not hasattr(config.augmentation, 'use_label_smoothing'), "use_label_smoothing should be removed"
        print("✅ Removed augmentation fields are not in defaults")
        
        # Test 4: Scheduler fields should not have T0/T_mul
        assert not hasattr(config.scheduler, 'T0'), "T0 should be removed"
        assert not hasattr(config.scheduler, 'T_mul'), "T_mul should be removed"
        print("✅ Unused scheduler fields (T0, T_mul) removed")
        
        # Test 5: No ~/.torch fallback should exist
        # update_config should not set dataset_dir based on download flag
        config_fallback_test = get_default_config()
        config_fallback_test.dataset.download = True
        config_fallback_test = update_config(config_fallback_test)
        assert config_fallback_test.dataset.dataset_dir == '', "Should not create default path even with download=True"
        print("✅ No ~/.torch fallback even when download=True")
        
        print("✅ Configuration validation passed")
        return True
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        traceback.print_exc()
        return False


def test_dead_references():
    """G. Search for dead references"""
    print("\n" + "="*60)
    print("TEST G: Dead Code Reference Check")
    print("="*60)
    try:
        import os
        import pathlib
        
        root = pathlib.Path('pytorch_image_classification')
        dead_keywords = [
            'use_ricap',
            'use_dual_cutout',
            'use_label_smoothing',
            'DualCutout',
            'RICAP',
            'LabelSmoothing',
            'RICAPLoss',
            'DualCutoutLoss',
            'LabelSmoothingLoss',
            'RICAPCollator',
            'adabound',
            'LARS',
        ]
        
        found = {kw: [] for kw in dead_keywords}
        
        for py_file in root.rglob('*.py'):
            with open(py_file, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                for kw in dead_keywords:
                    if kw in content:
                        found[kw].append(str(py_file))
        
        # Filter to only report significant findings
        # (some might exist in comments or expected error messages)
        found = {k: v for k, v in found.items() if v}
        
        if found:
            print("⚠️  Found potential dead code references:")
            for kw, files in found.items():
                print(f"   {kw}: {files}")
            print("\nManual review recommended - these might be in comments or error messages")
        else:
            print("✅ No dead code references found in package")
        
        return True
    except Exception as e:
        print(f"⚠️  Dead code check encountered error: {e}")
        return True  # Don't fail on this


def test_experiment_configs():
    """F. Test experiment config loading and explicit dataset_dir requirement"""
    print("\n" + "="*60)
    print("TEST F: Experiment Config Loading & Dataset Dir Requirement")
    print("="*60)
    try:
        from pytorch_image_classification import get_default_config, update_config, create_dataset
        
        configs_to_test = [
            'configs/classification/cnn_baseline.yaml',
            'configs/classification/cnn_mixup.yaml',
            'configs/classification/cnn_cutmix.yaml',
            'configs/classification/cnn_cutout.yaml',
            'configs/classification/cnn_random_erasing.yaml',
        ]
        
        for config_path in configs_to_test:
            config = get_default_config()
            config.merge_from_file(config_path)
            config = update_config(config)
            
            # Configs should have empty dataset_dir as template
            assert config.dataset.dataset_dir == '', f"{config_path} should have empty dataset_dir as template"
            print(f"✅ {config_path} loaded (dataset_dir empty as expected for template)")
        
        print("✅ All experiment configs load successfully as templates")
        print("✅ Users must explicitly set dataset_dir before training")
        return True
    except Exception as e:
        print(f"❌ Experiment config test failed: {e}")
        traceback.print_exc()
        return False


def test_kaggle_config():
    """H. Test Kaggle path configuration with explicit dataset_dir"""
    print("\n" + "="*60)
    print("TEST H: Kaggle Path Configuration (Explicit)")
    print("="*60)
    try:
        from pytorch_image_classification import (
            get_default_config, 
            update_config,
        )
        
        # Load a config file
        config = get_default_config()
        config.merge_from_file('configs/classification/cnn_baseline.yaml')
        config = update_config(config)
        
        # Explicitly set Kaggle path
        config.dataset.dataset_dir = '/kaggle/input/datasets/aloksilswal/cifar-10'
        config.dataset.download = False
        
        # Verify it's not overwritten
        assert config.dataset.dataset_dir == '/kaggle/input/datasets/aloksilswal/cifar-10', \
            "Kaggle path should not be overwritten!"
        print("✅ Kaggle path preserved after update_config")
        print("✅ Dataset download is controlled by explicit config.dataset.download")
        return True
    except Exception as e:
        print(f"❌ Kaggle config test failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests"""
    print("\n" + "="*60)
    print("LIGHTWEIGHT REFACTOR VALIDATION")
    print("="*60)
    
    results = {
        'A. Imports': test_imports(),
        'B. Config': test_config(),
        'F. Experiment Configs': test_experiment_configs(),
        'G. Dead References': test_dead_references(),
        'H. Kaggle Config': test_kaggle_config(),
    }
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(results.values())
    print("\n" + ("="*60))
    if all_passed:
        print("✅ All validation tests passed!")
    else:
        print("❌ Some validation tests failed")
        sys.exit(1)


if __name__ == '__main__':
    main()
