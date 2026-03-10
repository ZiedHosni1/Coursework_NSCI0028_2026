from decompyle3.main import decompile_file
from pathlib import Path
src = Path('matpub/__pycache__/analysis.cpython-38.pyc')
out = Path('matpub/analysis_from38.py')
with out.open('w', encoding='utf-8') as fh:
    decompile_file(str(src), outstream=fh)
print('ok', out)
