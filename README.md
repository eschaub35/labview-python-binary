# LabVIEW Python Binary

Python utilities for serializing Python data and decoding binary data exchanged with LabVIEW.

## Features

- `serialize_python_data(value)`: serializes Python data into a binary representation compatible with LabVIEW.
- `decode_labview_binary(json_string, reponse_binaire)`: decodes binary data received from LabVIEW.

## Usage

Download `Labview_python_serialize_deserialize.py` and place it in your Python project.

```python
from python import serialize_python_data, deserialize_labview_binary

# Serialize Python data
value = [1, 2, 3]
binary_data = serialize_python_data(value)

# Decode binary data received from LabVIEW
# decoded_data = deserialize_labview_binary(json_string, reponse_binaire)
```

## Requirements

- Python 3.9 or later
- NumPy

Install the required package with:

```bash
pip install numpy
```
## License

See the LICENSE file.
