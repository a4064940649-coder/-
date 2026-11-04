"""不可信 pickle 的白名單載入器：只允許 pandas／numpy／datetime 的重建函式，其他一律拒絕。"""
import pickle, io
OK_MOD = ('pandas', 'numpy', 'datetime', 'collections', 'copyreg', '_codecs', 'builtins', 'pytz', 'dateutil')
BAD = {('builtins', 'eval'), ('builtins', 'exec'), ('builtins', 'open'), ('builtins', '__import__'), ('builtins', 'getattr'), ('builtins', 'compile'), ('builtins', 'globals'), ('builtins', 'setattr'), ('builtins', 'delattr'), ('builtins', 'input'), ('builtins', 'breakpoint')}
class SafeUnpickler(pickle.Unpickler):
    seen = set()
    def find_class(self, module, name):
        if not module.split('.')[0] in OK_MOD or (module, name) in BAD:
            raise pickle.UnpicklingError(f'blocked {module}.{name}')
        if module == 'builtins' and name not in ('slice', 'range', 'complex', 'set', 'frozenset', 'list', 'dict', 'tuple', 'object', 'int', 'float', 'str', 'bytes', 'bytearray', 'bool'):
            raise pickle.UnpicklingError(f'blocked {module}.{name}')
        SafeUnpickler.seen.add((module, name))
        return super().find_class(module, name)
def load(path):
    with open(path, 'rb') as f: return SafeUnpickler(f).load()
