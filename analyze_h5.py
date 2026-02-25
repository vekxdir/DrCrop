import h5py
import numpy as np

def analyze_h5(path):
    print(f"--- Analyzing {path} ---")
    with h5py.File(path, 'r') as f:
        # Check if it has 'model_weights' (standard Keras H5)
        if 'model_weights' in f:
            weights = f['model_weights']
            for layer_name in weights:
                print(f"Layer: {layer_name}")
                group = weights[layer_name]
                for weight_name in group:
                    data = group[weight_name]
                    if isinstance(data, h5py.Dataset):
                        print(f"  - Dataset: {weight_name} Shape: {data.shape}")
                    elif isinstance(data, h5py.Group):
                        # Some versions nest one level deeper
                        for sub_name in data:
                            sub_data = data[sub_name]
                            if isinstance(sub_data, h5py.Dataset):
                                print(f"    - Sub-Dataset: {sub_name} Shape: {sub_data.shape}")
        else:
            print("No 'model_weights' group found. Listing internal structure recursively:")
            def visitor(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"Dataset: {name} | Shape: {obj.shape}")
            f.visititems(visitor)

if __name__ == "__main__":
    analyze_h5('model/model.h5')
