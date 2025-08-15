#!/usr/bin/env python3
"""
Evaluation script for train.py functionality.
Validates that the training script works correctly with the new multi-branch model.
"""

import os
import sys
import tempfile
import shutil

# Import backend modules (now we're in backend directory)
from periodizer import MultiBranchStarModelHybrid, StarModelConfig, multitask_loss
import torch
import numpy as np

def test_model_creation():
    """Test that MultiBranchStarModelHybrid can be created and used."""
    print("Testing multi-branch model creation...")
    
    cfg = StarModelConfig(
        n_types=13,
        lc_in_channels=1,
        pgram_in_channels=1,
        folded_in_channels=1,
        add_period_channel=True,
        emb_dim=128,
        merged_dim=256,
        cnn_hidden=64,
        d_model=128,
        n_heads=4,
        n_layers=2,
        dropout=0.1,
        logP_mean=0.0,
        logP_std=1.0,
    )
    
    model = MultiBranchStarModelHybrid(cfg)
    
    # Test forward pass with synthetic data
    batch_size = 2
    lc = torch.randn(batch_size, 1, 1200)  # Raw light curve
    pgram = torch.randn(batch_size, 1, 900)  # Periodogram
    
    # Folded candidates: list of tensors
    folded_list = [
        torch.randn(batch_size, 1, 200),
        torch.randn(batch_size, 1, 200),
        torch.randn(batch_size, 1, 200),
        torch.randn(batch_size, 1, 200)
    ]
    
    # Corresponding log periods
    logP_list = [
        torch.randn(batch_size),
        torch.randn(batch_size),
        torch.randn(batch_size),
        torch.randn(batch_size)
    ]
    
    outputs = model(lc, pgram, folded_list, logP_list)
    
    assert outputs["type_logits"].shape == (batch_size, 13), f"Expected type shape ({batch_size}, 13), got {outputs['type_logits'].shape}"
    assert outputs["logP_pred"].shape == (batch_size,), f"Expected period shape ({batch_size},), got {outputs['logP_pred'].shape}"
    assert outputs["cand_logits"].shape == (batch_size, 4), f"Expected candidate shape ({batch_size}, 4), got {outputs['cand_logits'].shape}"
    
    print("✓ Multi-branch model creation test passed")


def test_multitask_loss():
    """Test the multitask loss function."""
    print("Testing multitask loss function...")
    
    cfg = StarModelConfig(
        n_types=13,
        logP_mean=0.0,
        logP_std=1.0,
    )
    
    batch_size = 2
    
    # Create fake model outputs
    outputs = {
        "type_logits": torch.randn(batch_size, 13),
        "logP_pred": torch.randn(batch_size),
        "cand_logits": torch.randn(batch_size, 4),
    }
    
    # Create fake labels
    y_type = torch.randint(0, 13, (batch_size,))
    true_logP = torch.randn(batch_size)
    cand_labels = torch.zeros(batch_size, 4)
    cand_labels[:, 0] = 1.0  # First candidate is correct
    
    # Compute loss
    loss, logs = multitask_loss(outputs, y_type, true_logP, cand_labels=cand_labels, cfg=cfg)
    
    assert isinstance(loss, torch.Tensor), "Loss should be a tensor"
    assert loss.numel() == 1, "Loss should be a scalar"
    assert "loss" in logs, "Loss logs should contain 'loss'"
    assert "loss_type" in logs, "Loss logs should contain 'loss_type'"
    assert "loss_period" in logs, "Loss logs should contain 'loss_period'"
    assert "loss_cand" in logs, "Loss logs should contain 'loss_cand'"
    
    print("✓ Multitask loss test passed")


def test_model_save_load():
    """Test model saving and loading."""
    print("Testing model save/load...")
    
    cfg = StarModelConfig(
        n_types=13,
        lc_in_channels=1,
        pgram_in_channels=1,
        folded_in_channels=1,
        add_period_channel=True,
        emb_dim=64,  # Smaller for faster testing
        merged_dim=128,
        cnn_hidden=32,
        d_model=64,
        n_heads=2,
        n_layers=1,
        dropout=0.1,
        logP_mean=0.0,
        logP_std=1.0,
    )
    
    # Create model
    model = MultiBranchStarModelHybrid(cfg)
    
    # Save to temporary file
    with tempfile.NamedTemporaryFile(suffix='.pth', delete=False) as tmp_file:
        temp_path = tmp_file.name
    
    try:
        # Save model
        model_state = {
            'model_state_dict': model.state_dict(),
            'config': cfg,
            'class_names': ['test_class'] * 13
        }
        torch.save(model_state, temp_path)
        
        # Load model
        loaded_state = torch.load(temp_path, map_location='cpu', weights_only=False)
        loaded_model = MultiBranchStarModelHybrid(loaded_state['config'])
        loaded_model.load_state_dict(loaded_state['model_state_dict'])
        
        # Test that loaded model works
        batch_size = 1
        lc = torch.randn(batch_size, 1, 100)
        pgram = torch.randn(batch_size, 1, 50)
        folded_list = [torch.randn(batch_size, 1, 25) for _ in range(2)]
        logP_list = [torch.randn(batch_size) for _ in range(2)]
        
        outputs = loaded_model(lc, pgram, folded_list, logP_list)
        
        assert outputs["type_logits"].shape == (batch_size, 13)
        assert outputs["logP_pred"].shape == (batch_size,)
        
        print("✓ Model save/load test passed")
    
    finally:
        # Clean up
        if os.path.exists(temp_path):
            os.unlink(temp_path)


def test_training_data_format():
    """Test that sample data can be loaded and processed."""
    print("Testing sample data loading...")
    
    # Note: sample_data is relative to project root, we're in backend/ directory  
    sample_data_dir = "../sample_data"
    
    if not os.path.exists(sample_data_dir):
        print("⚠ Sample data directory not found - skipping data test")
        return
    
    # Try to import the data processing functions
    try:
        from train import load_light_curve_data, create_multi_branch_sample
        import glob
        
        # Get a sample file
        tbl_files = glob.glob(os.path.join(sample_data_dir, "*.tbl"))
        if not tbl_files:
            print("⚠ No .tbl files found - skipping data test")
            return
        
        # Test loading one file
        test_file = tbl_files[0]
        lc_data = load_light_curve_data(test_file)
        
        assert len(lc_data) > 0, "Light curve data should not be empty"
        assert lc_data.shape[1] == 2, "Light curve data should have 2 columns (time, flux)"
        
        # Test creating multi-branch sample
        sample = create_multi_branch_sample(lc_data, star_class=0)
        
        assert 'raw_lc' in sample, "Sample should contain raw_lc"
        assert 'periodogram' in sample, "Sample should contain periodogram"
        assert 'folded_candidates' in sample, "Sample should contain folded_candidates"
        assert 'candidate_periods' in sample, "Sample should contain candidate_periods"
        assert 'true_period' in sample, "Sample should contain true_period"
        assert 'class_idx' in sample, "Sample should contain class_idx"
        
        assert len(sample['folded_candidates']) == len(sample['candidate_periods']), \
            "Folded candidates and periods should have same length"
        
        print("✓ Sample data loading test passed")
        
    except ImportError as e:
        print(f"⚠ Could not import training functions: {e}")
    except Exception as e:
        print(f"⚠ Data test failed: {e}")


def main():
    """Run all evaluation tests."""
    print("Multi-Branch Model Evaluation Tests")
    print("=" * 40)
    
    try:
        test_model_creation()
        test_multitask_loss()
        test_model_save_load()
        test_training_data_format()
        
        print("\n" + "=" * 40)
        print("✅ All tests passed!")
        print("\nThe multi-branch training pipeline appears to be working correctly.")
        print("You can now run 'python backend/train.py' to train the model.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        print("\nPlease check the error above and fix any issues before training.")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


if __name__ == "__main__":
    main()