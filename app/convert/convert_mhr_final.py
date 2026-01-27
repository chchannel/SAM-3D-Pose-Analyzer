import torch
import os
import io
import pickle
import zipfile
import sys
import types
import patch_torch

# 1. Mock Type Storage for _rebuild_tensor_v2 compatibility
# PyTorch 2.1 UntypedStorage doesn't have dtype, but legacy loader expects TypedStorage behavior
class MockTypedStorage(torch.UntypedStorage):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._dtype = torch.float32 # Default

    @property
    def dtype(self):
        return self._dtype
    
    @property
    def _untyped_storage(self):
        return self

# Using patch_torch definitions instead of local ones to avoid pickle issues

# 2. Mock __torch__ module
if '__torch__' not in sys.modules:
    sys.modules['__torch__'] = types.ModuleType('__torch__')

class MockClass:
    def __init__(self, *args, **kwargs):
        pass
    def __call__(self, *args, **kwargs):
        return self

class ConverterUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        # 1. Handle torch storage
        if module == 'torch' and 'Storage' in name:
            if hasattr(torch, name):
                return getattr(torch, name)
            return getattr(torch, 'UntypedStorage')

        # 2. Handle __torch__ JIT classes
        # These are usually ScriptModules. We just want to load them as dummy objects 
        # that hold state_dict keys if possible, or just bypass structure.
        if module == '__main__':
            # Ensure we modify the actual main module seen by sys
            main_mod = sys.modules['__main__']
            if not hasattr(main_mod, name):
                cls = type(name, (object,), {
                    '__module__': '__main__',
                    '__init__': lambda self, *args, **kwargs: None,
                    '__setstate__': lambda self, state: setattr(self, '__state__', state)
                })
                setattr(main_mod, name, cls)
            return getattr(main_mod, name)

        # 2. Handle __torch__ JIT classes
        # These are usually ScriptModules. We just want to load them as dummy objects 
        # that hold state_dict keys if possible, or just bypass structure.
        if module.startswith('__torch__'):
            # Create submodules dynamically
            parts = module.split('.')
            curr = sys.modules.get(parts[0])
            for p in parts[1:]:
                if not hasattr(curr, p):
                    setattr(curr, p, types.ModuleType(p))
                curr = getattr(curr, p)
            
            if not hasattr(curr, name):
                # Create a Mock class that can be instantiated
                setattr(curr, name, type(name, (object,), {
                    '__init__': lambda self: None,
                    '__setstate__': lambda self, state: setattr(self, '__state__', state)
                }))
            return getattr(curr, name)

        return super().find_class(module, name)

    def persistent_load(self, saved_id):
        desc, cls, key, loc, numel = saved_id
        if desc != 'storage': return None
        
        try:
            candidates = [
                self.root_prefix + key,
                self.root_prefix + 'data/' + key,
                key,
                'archive/data/' + key
            ]
            
            valid_path = None
            for c in candidates:
                if c in self.zf.namelist():
                    valid_path = c
                    break
            
            if not valid_path:
                print(f"Warning: Storage key {key} NOT FOUND in candidates: {candidates[:2]}...")
                untyped = torch.UntypedStorage(numel)
            else:
                with self.zf.open(valid_path) as f:
                    data_bytes = f.read()
                untyped = torch.UntypedStorage.from_buffer(data_bytes, byte_order='native', dtype=torch.uint8)
            
            dtype_map = {
                'UInt32Storage': torch.int32,
                'Int64Storage': torch.int64,
                'FloatStorage': torch.float32,
                'DoubleStorage': torch.float64,
                'ByteStorage': torch.uint8,
                'BoolStorage': torch.bool,
                'HalfStorage': torch.float16,
                'BFloat16Storage': torch.bfloat16,
            }
            target_dtype = dtype_map.get(cls.__name__, torch.uint8)
            
            # Use module-level Proxy defined in patch_torch
            proxy = patch_torch.StorageProxy(untyped, target_dtype)
            return proxy

        except Exception as e:
            print(f"Error loading key {key}: {e}")
            return torch.UntypedStorage(numel)

def check_and_convert(path):
    print(f"Converting: {path}")
    try:
        if not zipfile.is_zipfile(path):
            print("Not a zip file!")
            return

        with zipfile.ZipFile(path, 'r') as zf:
            names = zf.namelist()
            data_file = None
            root_prefix = ""
            for n in names:
                if n.endswith('data.pkl'):
                    data_file = n
                    root_prefix = n.rsplit('data.pkl', 1)[0]
                    break
            
            if not data_file:
                print("No data.pkl found")
                return

            print(f"Found data.pkl at {data_file}")
            
            with zf.open(data_file) as df:
                unpickler = ConverterUnpickler(df)
                unpickler.zf = zf
                unpickler.root_prefix = root_prefix
                data = unpickler.load()
        
        print("Loaded data structure.")
        # Inspect data
        # If it's a ScriptModule, 'data' is the object graph.
        # We want to extract tensors (state_dict).
        # We can try to traverse 'data' to find tensors?
        print("Loaded data structure.")
        
        # EXTRACT STATE DICT
        # We traverse the Mock objects to find tensors.
        state_dict = {}
        
        def traverse(obj, prefix):
            # If it's a Tensor, save it
            if isinstance(obj, torch.Tensor):
                state_dict[prefix.rstrip('.')] = obj
                return

            # If it's our Mock object
            if hasattr(obj, '__state__'):
                s = obj.__state__
                if isinstance(s, dict):
                    for k, v in s.items():
                        traverse(v, prefix + k + ".")
                return

            # If it's a dict (e.g. state of mock)
            if isinstance(obj, dict):
                for k, v in obj.items():
                    # Check if keys are string
                    if isinstance(k, str):
                        traverse(v, prefix + k + ".")
                return

            # If it's top level MHRDemo mock
            # It might have submodules?
            # Standard JIT structure usually has '_modules', '_parameters', '_buffers' keys in state?
            # If so, we need to be careful with paths.
            pass

        # Try to extract
        traverse(data, "")
        
        print(f"Extracted {len(state_dict)} tensors.")
        if len(state_dict) == 0:
            print("Warning: No tensors found! Inspecting data...")
            print(dir(data))
            if hasattr(data, '__state__'):
                print(data.__state__.keys())
            
            # Maybe it IS a dict already?
            if isinstance(data, dict):
                # Recursive search in dict
                pass
        
        if len(state_dict) > 0:
            torch.save(state_dict, path + ".converted")
            os.replace(path + ".converted", path)
            print("Saved converted model (state_dict only).")
        else:
            print("Failed to extract state dict.")

    except Exception as e:
        print(f"Conversion failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    p = 'sam-3d-body/checkpoints/sam-3d-body-dinov3/assets/mhr_model.pt'
    if os.path.exists(p):
        check_and_convert(p)
