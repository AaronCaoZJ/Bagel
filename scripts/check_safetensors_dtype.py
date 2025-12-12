"""
Script to check and report data types of all tensors in a safetensors file
"""
import torch
from safetensors.torch import load_file
import os
import sys

def check_safetensors_dtype(safetensors_path, output_report="safetensors_report.txt"):
    """
    Load a safetensors file and generate a detailed dtype report
    
    Args:
        safetensors_path: Path to the .safetensors file
        output_report: Path to save the report (default: safetensors_report.txt)
    """
    if not os.path.exists(safetensors_path):
        print(f"❌ Error: File not found: {safetensors_path}")
        return
    
    print(f"[CHECK] Loading safetensors from: {safetensors_path}")
    file_size_gb = os.path.getsize(safetensors_path) / 1024**3
    print(f"[CHECK] File size: {file_size_gb:.2f} GB")
    
    try:
        # Load the safetensors file
        state_dict = load_file(safetensors_path)
        print(f"[CHECK] Successfully loaded {len(state_dict)} tensors")
        
        # Helper function to convert dtype to readable string
        def get_dtype_str(dtype):
            if dtype == torch.float8_e4m3fn:
                return "F8_E4M3"
            elif dtype == torch.float8_e5m2:
                return "F8_E5M2"
            elif dtype == torch.bfloat16:
                return "BF16"
            elif dtype == torch.float16:
                return "FP16"
            elif dtype == torch.float32:
                return "FP32"
            elif dtype == torch.int8:
                return "INT8"
            elif dtype == torch.uint8:
                return "UINT8"
            else:
                return str(dtype).replace("torch.", "")
        
        # Generate report
        print(f"[CHECK] Generating report to {output_report}...")
        
        with open(output_report, "w", encoding="utf-8") as f:
            # Write header
            f.write(f"{'Layer Name':<70}\t{'Shape':<25}\t{'Dtype':<15}\t{'Size (MB)'}\n")
            f.write("=" * 130 + "\n")
            
            # Statistics counters
            dtype_counts = {}
            total_params = 0
            total_size_mb = 0
            
            # Process each tensor
            for name, tensor in state_dict.items():
                shape_str = str(list(tensor.shape))
                dtype_str = get_dtype_str(tensor.dtype)
                
                # Calculate size in MB
                num_elements = tensor.numel()
                bytes_per_element = tensor.element_size()
                size_mb = (num_elements * bytes_per_element) / (1024**2)
                
                # Write to report
                f.write(f"{name:<70}\t{shape_str:<25}\t{dtype_str:<15}\t{size_mb:.2f}\n")
                
                # Update statistics
                dtype_counts[dtype_str] = dtype_counts.get(dtype_str, 0) + 1
                total_params += num_elements
                total_size_mb += size_mb
            
            # Write summary
            f.write("\n" + "=" * 130 + "\n")
            f.write("SUMMARY\n")
            f.write("=" * 130 + "\n")
            f.write(f"Total tensors: {len(state_dict)}\n")
            f.write(f"Total parameters: {total_params:,}\n")
            f.write(f"Total size: {total_size_mb:.2f} MB ({total_size_mb/1024:.2f} GB)\n")
            f.write(f"File size on disk: {file_size_gb:.2f} GB\n\n")
            
            f.write("Dtype Distribution:\n")
            for dtype_name, count in sorted(dtype_counts.items()):
                percentage = (count / len(state_dict)) * 100
                f.write(f"  {dtype_name}: {count} tensors ({percentage:.1f}%)\n")
        
        print(f"[CHECK] ✅ Report saved successfully!")
        print(f"\nSummary:")
        print(f"  Total tensors: {len(state_dict)}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Total size: {total_size_mb/1024:.2f} GB")
        print(f"\nDtype Distribution:")
        for dtype_name, count in sorted(dtype_counts.items()):
            percentage = (count / len(state_dict)) * 100
            print(f"  {dtype_name}: {count} tensors ({percentage:.1f}%)")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"[CHECK] ❌ Error reading safetensors: {e}")
        print(traceback.format_exc())
        return False


if __name__ == "__main__":
    # Default path
    default_path = "/home/zhijun/Code/Bagel/models/BAGEL-7B-MoT/model_fp8.safetensors"
    
    # Check if user provided a path via command line
    if len(sys.argv) > 1:
        safetensors_path = sys.argv[1]
    else:
        safetensors_path = default_path
    
    # Optional: custom output report path
    if len(sys.argv) > 2:
        output_report = sys.argv[2]
    else:
        output_report = "safetensors_report.txt"
    
    # Run the check
    check_safetensors_dtype(safetensors_path, output_report)