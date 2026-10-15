#%%
import json
import struct
import numpy as np


class LabVIEWDecoder:

    def __init__(self, data):
        # data peut être bytes, bytearray ou tableau numpy de uint8
        self.data = bytes(data)
        self.pos = 0

    def read_bytes(self, n):
        """Lit n octets et avance dans le buffer."""
        if self.pos + n > len(self.data):
            raise ValueError(
                f"Données insuffisantes : "
                f"{n} octets demandés à la position {self.pos}, "
                f"{len(self.data) - self.pos} disponibles"
            )

        result = self.data[self.pos:self.pos + n]
        self.pos += n
        return result

    def read_I32(self):
        """Lit un I32 big-endian."""
        return struct.unpack(">i", self.read_bytes(4))[0]

    # def read_U32(self):
    #     """Lit un U32 big-endian."""
    #     return struct.unpack(">I", self.read_bytes(4))[0]

    # def read_DBL(self):
    #     """Lit un DBL big-endian."""
    #     return struct.unpack(">d", self.read_bytes(8))[0]

    def read_Boolean(self):
        """Lit un Boolean LabVIEW."""
        return self.read_bytes(1)[0] != 0


    def read_numeric(self, type_):

        dtype_map = {
            "I8": ">i1",
            "U8": ">u1",
            "I16": ">i2",
            "U16": ">u2",
            "I32": ">i4",
            "U32": ">u4",
            "I64": ">i8",
            "U64": ">u8",
            "Single Float": ">f4",
            "Double Float": ">f8",
        }

        # if type_ in ("Complex Single Float", "Complex Double Float"):
        #     component_dtype = (
        #         ">f4" if type_ == "Complex Single Float" else ">f8"
        #     )

        #     components = np.frombuffer(
        #         self.read_bytes(2 * np.dtype(component_dtype).itemsize),
        #         dtype=np.dtype(component_dtype),
        #         count=2
        #     )

        #     return complex(float(components[0]), float(components[1]))


        if type_ in ("Single Complex", "Double Complex"):
            component_dtype = (
                ">f4" if type_ == "Single Complex" else ">f8"
            )

            components = np.frombuffer(
                self.read_bytes(2 * np.dtype(component_dtype).itemsize),
                dtype=np.dtype(component_dtype),
                count=2
            )

            return complex(float(components[0]), float(components[1]))


        if type_ not in dtype_map:
            raise ValueError(f"Type numérique LabVIEW inconnu : {type_}")

        dtype = np.dtype(dtype_map[type_])

        return np.frombuffer(
            self.read_bytes(dtype.itemsize),
            dtype=dtype,
            count=1
        )[0].item()

    
    def read_String(self):
        """Lit une String LabVIEW."""
        length = self.read_I32()

        if length < 0:
            raise ValueError(
                f"Longueur de String invalide : {length}"
            )

        raw = self.read_bytes(length)

        return raw.decode("cp1252")

    def decode(self, description):

        type_ = description["type"]

        if type_ == "Cluster":
            return self.decode_cluster(description)

        elif type_ == "Array":
            return self.decode_array(description)

        elif type_ == "Boolean":
            return self.read_Boolean()


        elif type_ == "String":
            return self.read_String()

        else:
            return self.read_numeric(type_)

    def decode_cluster(self, description):
        """
        Décode un cluster.
        """
        result = {}

        for name, field_description in description["fields"].items():

            result[name] = self.decode(field_description)

        return result

    def decode_array(self, description):
        """
        Décode un tableau LabVIEW dont les dimensions
        sont préfixées dans le binaire.
        """

        dimensions = description["dimensions"]

        # Lecture des tailles des dimensions
        shape = tuple(
            self.read_I32()
            for _ in range(dimensions)
        )

        # Vérification
        if any(size < 0 for size in shape):
            raise ValueError(f"Taille de tableau invalide : {shape}")

        element_description = description["element"]

        # Tableau numérique / booléen
        if self.is_numpy_array_type(element_description):

            return self.decode_numpy_array(
                shape,
                element_description
            )

        # Tableau de clusters
        else:

            return self.decode_object_array(
                shape,
                element_description
            )

    def is_numpy_array_type(self, description):
        if "type" in description:
            return description["type"] in {
                "I8", "U8",
                "I16", "U16",
                "I32", "U32",
                "I64", "U64",
                "Single Float",
                "Double Float",
                "Single Complex",
                "Double Complex",
                "Boolean"
            }

        return False


    def decode_numpy_array(self, shape, description):

        type_ = description["type"]

        dtype_map = {
            "I8": ">i1",
            "U8": ">u1",
            "I16": ">i2",
            "U16": ">u2",
            "I32": ">i4",
            "U32": ">u4",
            "I64": ">i8",
            "U64": ">u8",
            "Single Float": ">f4",
            "Double Float": ">f8",
            "Boolean": "u1",
        }

        # Types complexes : composantes réelle et imaginaire successives
        complex_types = {
            "Single Complex": (">f4", np.complex64),
            "Double Complex": (">f8", np.complex128),
        }

        count = int(np.prod(shape))

        if type_ in complex_types:
            component_dtype, complex_dtype = complex_types[type_]
            component_dtype = np.dtype(component_dtype)

            raw = self.read_bytes(
                count * 2 * component_dtype.itemsize
            )

            components = np.frombuffer(
                raw,
                dtype=component_dtype,
                count=2 * count
            )

            # Les composantes sont entrelacées :
            # réel 0, imaginaire 0, réel 1, imaginaire 1, ...
            # array = (
            #     components[0::2].astype(np.float64)
            #     + 1j * components[1::2].astype(np.float64)
            # ).astype(complex_dtype)
            array = np.empty(count, dtype=complex_dtype)
            array.real = components[0::2]
            array.imag = components[1::2]

            return array.reshape(shape, order="F")

        if type_ not in dtype_map:
            raise ValueError(f"Type numérique inconnu : {type_}")

        dtype = np.dtype(dtype_map[type_])

        raw = self.read_bytes(count * dtype.itemsize)

        array = np.frombuffer(
            raw,
            dtype=dtype,
            count=count
        )

        if type_ == "Boolean":
            array = array.astype(bool)

        return array.reshape(shape, order="F")


    def decode_object_array(self, shape, description):

        count = int(np.prod(shape))

        values = []

        for _ in range(count):
            values.append(
                self.decode_array_element(description)
            )

        # Tableau numpy d'objets
        array = np.empty(count, dtype=object)

        for i, value in enumerate(values):
            array[i] = value

        return array.reshape(shape, order="F")


    def decode_array_element(self, description):
        return self.decode(description)

def deserialize_labview_binary(json_string, reponse_binaire):
    cluster_type_dic = json.loads(json_string)
    decoder = LabVIEWDecoder(reponse_binaire)

    # Décodage direct du cluster racine
    result = decoder.decode(cluster_type_dic)

    # Vérification : tous les octets doivent avoir été consommés
    if decoder.pos != len(decoder.data):
        remaining = len(decoder.data) - decoder.pos
        raise ValueError(
            f"{remaining} octets restent après décodage "
            f"(position {decoder.pos}/{len(decoder.data)})"
        )

    return result



import struct
import numpy as np


# ------------------------------------------------------------
# Types numériques LabVIEW -> représentation NumPy big-endian
# ------------------------------------------------------------

NUMPY_TO_LABVIEW = {
    np.dtype("int8"):    ("I8", ">i1"),
    np.dtype("int16"):   ("I16", ">i2"),
    np.dtype("int32"):   ("I32", ">i4"),
    np.dtype("int64"):   ("I64", ">i8"),

    np.dtype("uint8"):   ("U8", ">u1"),
    np.dtype("uint16"):  ("U16", ">u2"),
    np.dtype("uint32"):  ("U32", ">u4"),
    np.dtype("uint64"):  ("U64", ">u8"),

    np.dtype("float32"): ("Single Float", ">f4"),
    np.dtype("float64"): ("Double Float", ">f8"),

    np.dtype("complex64"):  ("Single Complex", ">c8"),
    np.dtype("complex128"): ("Double Complex", ">c16"),
}


# ------------------------------------------------------------
# Entiers signés 32 bits : dimensions et chaînes
# ------------------------------------------------------------

def pack_i32(value):
    """Encode un entier signé I32 en big-endian."""
    if not -(2**31) <= value < 2**31:
        raise OverflowError(f"{value} ne tient pas dans un I32")
    return struct.pack(">i", value)


# ------------------------------------------------------------
# Sérialisation récursive
# ------------------------------------------------------------

def serialize_python_data(value):
    """
    Sérialise une valeur Python en octets selon les conventions
    définies pour la conversion vers un cluster LabVIEW.

    Conventions :
      bool          -> Boolean (1 octet, convention à valider)
      int           -> I32
      float         -> Double Float
      str           -> String CP1252, longueur I32
      dict          -> Cluster, champs dans l'ordre d'insertion
      ndarray       -> Array, dimensions puis données
      list / tuple  -> Array 1D homogène
      np.generic    -> type déduit du dtype NumPy

    Les types non pris en charge provoquent une exception.
    """

    # 1. Booléen : impérativement avant int
    if isinstance(value, (bool, np.bool_)):
        return b"\x01" if value else b"\x00"

    # 2. Chaîne : CP1252
    if isinstance(value, str):
        encoded = value.encode("cp1252", errors="strict")
        return pack_i32(len(encoded)) + encoded

    # 3. Données binaires : tableau 1D de U8
    if isinstance(value, (bytes, bytearray)):
        return pack_i32(len(value)) + bytes(value)

    # 4. Dictionnaire : Cluster
    if isinstance(value, dict):
        parts = []

        for key, field_value in value.items():
            if not isinstance(key, str):
                raise TypeError(
                    f"Les clés d'un cluster doivent être des chaînes : {key!r}"
                )

            parts.append(serialize_python_data(field_value))

        return b"".join(parts)

    # 5. Tableau NumPy
    if isinstance(value, np.ndarray):

        # Tableau object : clusters ou nombres complexes
        if value.dtype == object:
            elements = value.ravel(order="C")

            # Tableau de clusters (dictionnaires)
            if elements.size > 0 and all( isinstance(item, dict) for item in elements ):
                data = b"".join(serialize_python_data(item) for item in elements)

            # Tableau de nombres complexes Python/NumPy
            elif elements.size > 0 and all( isinstance(item, (complex, np.complexfloating)) for item in elements ):
                data = b"".join(serialize_python_data(item) for item in elements)

            else:
                raise TypeError(
                    "Tableau object non pris en charge : "
                    "les éléments doivent être des dictionnaires "
                    "ou des nombres complexes."
                )

        else:
            # Refuser les tableaux structurés
            if value.dtype.fields is not None:
                raise TypeError(
                    "Les tableaux NumPy structurés ne sont pas pris "
                    "en charge directement."
                )

            # Tableau de booléens
            if value.dtype.kind == "b":
                data = value.astype(np.uint8).tobytes(order="C")

            # Types numériques pris en charge
            elif value.dtype.newbyteorder("=") in NUMPY_TO_LABVIEW:
                native_dtype = value.dtype.newbyteorder("=")
                _, target_dtype = NUMPY_TO_LABVIEW[native_dtype]
                data = value.astype(target_dtype).tobytes(order="C")

            else:
                raise TypeError(
                    f"dtype NumPy non pris en charge : {value.dtype}"
                )

        # Préfixe contenant la taille de chaque dimension
        dimensions = b"".join(
            pack_i32(int(size)) for size in value.shape
        )

        return dimensions + data


    # 6. Scalaire NumPy (np.int32, np.float32, etc.)
    if isinstance(value, np.generic):
        dtype = value.dtype

        if dtype.kind == "b":
            return b"\x01" if bool(value) else b"\x00"

        if dtype in NUMPY_TO_LABVIEW:
            _, target_dtype = NUMPY_TO_LABVIEW[dtype]
            return np.asarray(value).astype(target_dtype).tobytes(order="C")

        raise TypeError(f"Type scalaire NumPy non pris en charge : {dtype}")

    # 7. Entier Python natif -> I32
    if isinstance(value, int):
        return pack_i32(value)

    # 8. Flottant Python natif -> Double Float
    if isinstance(value, float):
        return struct.pack(">d", value)

    # 9. Nombre complexe Python natif -> Double Complex
    if isinstance(value, complex):
        return struct.pack(">dd", value.real, value.imag)    

    # 10. Liste ou tuple -> tableau 1D
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(
                "Impossible de déduire le type d'un tableau vide."
            )

        # Liste/tuple de nombres complexes
        if all(
            isinstance(v, (complex, np.complexfloating))
            for v in value
        ):
            array = np.asarray(value, dtype=np.complex128)
            return serialize_python_data(array)

        # Les tableaux numériques homogènes sont convertis en ndarray.
        # On refuse les conversions implicites de types hétérogènes.
        if all(
            isinstance(v, (bool, int, float, np.generic))
            and not isinstance(v, (str, bytes))
            for v in value
        ):
            # Empêcher notamment la promotion silencieuse bool -> entier.
            if all(isinstance(v, (bool, np.bool_)) for v in value):
                array = np.asarray(value, dtype=np.bool_)
            elif all(
                isinstance(v, (int, np.integer))
                and not isinstance(v, (bool, np.bool_))
                for v in value
            ):
                # Les entiers Python natifs doivent rester I32.
                # On ne convertit pas automatiquement une liste d'entiers
                # en int64.
                if all(
                    isinstance(v, (int, np.integer))
                    and not isinstance(v, (bool, np.bool_))
                    for v in value
                ):
                    if any(
                        isinstance(v, (int, np.integer))
                        and not isinstance(v, np.generic)
                        for v in value
                    ):
                        array = np.asarray(value, dtype=np.int32)
                    else:
                        array = np.asarray(value)
            elif all(
                isinstance(v, (float, np.floating))
                and not isinstance(v, (bool, np.bool_))
                for v in value
            ):
                if any(type(v) is float for v in value):
                    array = np.asarray(value, dtype=np.float64)
                else:
                    array = np.asarray(value)
            else:
                raise TypeError(
                    "Liste numérique hétérogène : types mélangés."
                )

            return serialize_python_data(array)

        # Tableau de chaînes : longueur puis contenu pour chaque élément.
        if all(isinstance(v, str) for v in value):
            return (
                pack_i32(len(value))
                + b"".join(serialize_python_data(v) for v in value)
            )

        # Tableau de clusters : tous les éléments doivent être des dict
        # de structure identique.
        if all(isinstance(v, dict) for v in value):
            expected_keys = list(value[0].keys())

            for item in value[1:]:
                if list(item.keys()) != expected_keys:
                    raise ValueError(
                        "Les dictionnaires d'un tableau de clusters "
                        "doivent avoir les mêmes clés dans le même ordre."
                    )

            return pack_i32(len(value)) + b"".join(
                serialize_python_data(item) for item in value
            )

        raise TypeError(
            "Liste/tuple non pris en charge ou hétérogène."
        )

    raise TypeError(
        f"Type Python non pris en charge : {type(value).__name__}"
    )

