import h5py

def list_layers(path):
    print(f"--- Layers in {path} ---")
    with h5py.File(path, 'r') as f:
        if 'model_weights' in f:
            for layer_name in f['model_weights']:
                print(f"Layer: {layer_name}")
                for weight_name in f['model_weights'][layer_name]:
                    print(f"  - Weight: {weight_name}")
        else:
            # Recursive listing
            def print_item(name, obj):
                if isinstance(obj, h5py.Dataset):
                    print(f"Dataset: {name} (Shape: {obj.shape})")
            f.visititems(print_item)

if __name__ == "__main__":
    list_layers('model/model.h5')
