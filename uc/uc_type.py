class uCType:
    """
    Class that represents a type in the uC language.  Basic
    Types are declared as singleton instances of this type.
    """

    def __init__(
        self, name, binary_ops=set(), unary_ops=set(), rel_ops=set(), assign_ops=set()
    ):
        """
        You must implement yourself and figure out what to store.
        """
        self.typename = name
        self.unary_ops = unary_ops
        self.binary_ops = binary_ops
        self.rel_ops = rel_ops
        self.assign_ops = assign_ops

    def __str__(self):
        return f"type({self.typename})"


# Create specific instances of basic types. You will need to add
# appropriate arguments depending on your definition of uCType
IntType = uCType(
    "int",
    unary_ops={"-", "+"},
    binary_ops={"+", "-", "*", "/", "%"},
    rel_ops={"==", "!=", "<", ">", "<=", ">="},
    assign_ops={"="},
)

CharType = uCType(
    "char",
    unary_ops={"-", "+"},
    binary_ops={},
    rel_ops={"==", "!=", "<", ">", "<=", ">="},
    assign_ops={"="},
)

BoolType = uCType(
    "bool",
    unary_ops={"!"},
    binary_ops={"&&", "||"},
    rel_ops={"==", "!=", "<", ">", "<=", ">="},
    assign_ops={"="},
)

VoidType = uCType(
    "void",
    unary_ops={},
    binary_ops={},
    rel_ops={},
    assign_ops={},
)

StringType = uCType(
    "string",
    unary_ops={},
    binary_ops={},
    rel_ops={},
    assign_ops={},
)

BasicVariableTypes = [IntType, CharType, BoolType, StringType]


# TODO: Check if ArrayType and FunctionType are correct
class ArrayType(uCType):
    def __init__(self, element_type, size=None):
        """
        type: Any of the uCTypes can be used as the array's type. This
              means that there's support for nested types, like matrices.
        size: Integer with the length of the array.
        """
        self.type = element_type
        self.size = size
        super().__init__(
            f"Array<{element_type.typename}>", rel_ops={"==", "!="})

    def __eq__(self, other):
        if isinstance(other, ArrayType):
            return self.type == other.type and self.size == other.size
        return False


class FunctionType(uCType):
    def __init__(self, return_type, parameters_type, size=None):
        self.return_type = return_type
        self.parameters_type = parameters_type
        return_type_str = return_type.typename
        parameters_type_str = ", ".join([p.typename for p in parameters_type])
        super().__init__(
            f"{return_type_str} Function({parameters_type_str})")
