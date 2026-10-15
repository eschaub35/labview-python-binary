# LabVIEW Python Binary

Python utilities for serializing Python data and decoding binary data exchanged with LabVIEW.

## Features

- `serialize_python_data(value)`: serializes Python data into a binary representation compatible with LabVIEW.
- `decode_labview_binary(json_string, reponse_binaire)`: decodes binary data received from LabVIEW.

## Usage

Download `python.py` and place it in your Python project.

```python
from python import serialize_python_data, decode_labview_binary

# Serialize Python data
value = [1, 2, 3]
binary_data = serialize_python_data(value)

# Decode binary data received from LabVIEW
# decoded_data = decode_labview_binary(json_string, reponse_binaire)
```

## Requirements

Specify the required Python packages here.

## License

See the LICENSE file.
