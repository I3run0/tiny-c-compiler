import argparse
import pathlib
import sys
from typing import Dict, List, Tuple, Union

from uc.uc_ast import (
    Assignment,
    Compound,
    Constant,
    Decl,
    DeclList,
    ExprList,
    FuncDef,
    Node,
    ParamList,
    Print,
    VarDecl,
    Assert,
    GlobalDecl,
    FuncDecl,
    FuncCall,
    If,
    For,
    While,
    Break,
    Read,
    Return,
    ID,
    BinaryOp,
    UnaryOp,
    InitList,
    ArrayRef,
    ArrayDecl,
)
from uc.uc_block import (
    CFG,
    BasicBlock,
    Block,
    ConditionBlock,
    EmitBlocks,
    format_instruction,
)
from uc.uc_interpreter import Interpreter
from uc.uc_parser import UCParser
from uc.uc_sema import NodeVisitor, Visitor
from uc.uc_type import ArrayType


class CodeGenerator(NodeVisitor):
    """
    Node visitor class that creates 3-address encoded instruction sequences
    with Basic Blocks & Control Flow Graph.
    """

    def __init__(self, viewcfg: bool, debug_print: bool = False):
        # To debug with debug_print
        self._enable_stdout_debug = debug_print
        self.viewcfg: bool = viewcfg
        # TODO: Remove this BasicBlock initialization and handle blocks correctly
        self.current_block: Block = BasicBlock("temp")

        # version dictionary for temporaries. We use the name as a Key
        self.fname: str = "_glob_"
        self.return_temp: Union[str, None] = None
        self.parameters_temp: Union[List[str], None] = None
        self.versions: Dict[str, int] = {self.fname: 0}
        # version dictionary for labels to avoid collisions
        self.label_versions: Dict[str, int] = {"if": 1, "while": 1, "for": 1}

        self.scope_stack: List[Dict[str, str]] = [{}]

        # The generated code (list of tuples)
        # At the end of visit_program, we call each function definition to emit
        # the instructions inside basic blocks. The global instructions that
        # are stored in self.text are appended at beginning of the code
        self.code: List[Tuple[str]] = []

        self.text: List[Tuple[str]] = (
            []
        )  # Used for global declarations & constants (list, strings)

        # TODO: Complete if needed.
        self.binary_ops = {
            "+": "add",
            "-": "sub",
            "*": "mul",
            "/": "div",
            "%": "mod",
            ">": 'gt',
            "<": "lt",
            "==": "eq",
            "!": "not",
            "&&": "and",
            "||": "or",
            ">=": "ge",
            "<=": "le",
            "!=": "ne",
        }

    def debug_print(self, msg):
        if self._enable_stdout_debug:
            print(msg, file=sys.stdout)

    def debug_print_instructions(self, instructions: List[Tuple[str]]):
        if self._enable_stdout_debug:
            for inst in instructions:
                print(inst, file=sys.stdout)

    def show(self, buf=sys.stdout):
        _str = ""
        for _code in self.code:
            _str += format_instruction(_code) + "\n"
        buf.write(_str)

    def new_temp(self) -> str:
        """
        Create a new temporary variable of a given scope (function name).
        """
        if self.fname not in self.versions:
            self.versions[self.fname] = 1
        name = "%" + "%d" % (self.versions[self.fname])
        self.versions[self.fname] += 1
        return name

    def new_reg(self, name: str) -> str:
        """
        Create a new temporary variable of a given scope (function name).
        """
        reg_name = "%" + "%s" % (name)
        self.scope_stack[-1][name] = reg_name
        if name not in self.versions:
            self.versions[name] = 2
            return reg_name
        reg_name = reg_name + ".%d" % (self.versions[name])
        self.versions[name] += 1
        return reg_name

    def new_global_reg(self, name: str) -> str:
        """
        Create a new temporary variable of global scope.
        """
        reg_name = "@" + "%s" % (name)
        self.scope_stack[0][name] = reg_name

        return reg_name

    def current_temp(self) -> str:
        """
        Return the current temporary variable name.
        """
        return "%" + "%d" % (self.versions[self.fname] - 1)

    def new_temp_label(self, type: str) -> str:
        """
        Create a new label an if.then, if.end, if.else, for.cond, etc.
        """
        if type not in self.label_versions:
            self.label_versions[type] = 1
        name = f"{type}" + ".%d" % (self.label_versions[type])
        self.label_versions[type] += 1
        return name

    def new_text(self, typename: str) -> str:
        """
        Create a new literal constant on global section (text).
        """
        name = "@." + typename + "." + "%d" % (self.versions["_glob_"])
        self.versions["_glob_"] += 1
        return name

    def is_global(self, varname: str) -> bool:
        '''
        Check if the variable is a global one 
        '''
        for i in range(len(self.scope_stack)-1, -1, -1):
            scope = self.scope_stack[i]
            if varname in scope:
                return i == 0

        return False

    def push_scope(self):
        '''
        Push a new scope to the stack
        '''
        self.scope_stack.append({})

    def pop_scope(self):
        '''
        Pop the last scope from the stack
        '''
        self.scope_stack.pop()

    def parse_literal_values(self, value, vtype: str):
        '''
        Parse a literral value from string to bool, int . etc
        '''
        if vtype == "int":
            return int(value)
        elif vtype == "bool":
            return bool(value)

    def visit_BinaryOp(self, node: BinaryOp):
        # Visit the left and right expressions
        if not isinstance(node.rvalue, BinaryOp) and not isinstance(
            node.lvalue, BinaryOp
        ):
            self.visit(node.rvalue)
            self.visit(node.lvalue)
        else:
            self.visit(node.rvalue)
            self.visit(node.lvalue)

        # TODO:
        # - Load the location containing the left expression
        # - Load the location containing the right expression

        # Make a new temporary for storing the result
        target = self.new_temp()

        # Create the opcode and append to list
        opcode = self.binary_ops[node.op] + "_" + node.lvalue.uc_type.typename
        inst = (opcode, node.lvalue.gen_location,
                node.rvalue.gen_location, target)
        self.current_block.append(inst)

        # Store location of the result on the node
        node.gen_location = target

    def visit_Print(self, node: Print):
        # Visit the expression
        if isinstance(node.expr, ExprList):
            exprs: ExprList = node.expr
            for expr in exprs.exprs:
                self.visit(expr)
                inst = ("print_" + expr.uc_type.typename, expr.gen_location)
                self.current_block.append(inst)

        elif node.expr is not None:
            self.visit(node.expr)

            # Create the opcode and append to list
            inst = ("print_" + node.expr.uc_type.typename,
                    node.expr.gen_location)
            self.current_block.append(inst)
        else:
            self.current_block.append(("print_void",))

    def array_dims_from_uc_type_array(self, array_type):
        dims = []
        current_array_type = array_type
        while isinstance(current_array_type, ArrayType):
            dims.append(current_array_type.size)
            current_array_type = current_array_type.type

        return dims

    def element_type_from_uc_type_array(self, array_type):
        current_array_type = array_type
        while isinstance(current_array_type, ArrayType):
            current_array_type = current_array_type.type

        return current_array_type

    def array_type_to_str(self, array_type):
        dims = self.array_dims_from_uc_type_array(array_type)
        element_type = self.element_type_from_uc_type_array(
            array_type).typename

        dims_str = "_".join([str(i) for i in dims])
        return f'{element_type}_{dims_str}'

    def init_list_to_values(self, init_list):
        if isinstance(init_list, InitList):
            return list(map(self.init_list_to_values,
                            init_list.exprs))
        else:
            return self.parse_literal_values(
                init_list.value,
                init_list.uc_type.typename)

    def visit_VarDecl(self, node: VarDecl):
        """
        Allocate the variable (global or local) with the correct initial value (if there is any).
        """
        root_decl = node.parent
        while not isinstance(root_decl, Decl):
            root_decl = root_decl.parent

        if isinstance(node.parent.parent, GlobalDecl) or \
           isinstance(root_decl.parent, GlobalDecl):
            # Global declarations handled at GlobalDecl
            return

        # Store optional init val
        if isinstance(node.parent, ArrayDecl):
            root_array_decl = node.parent
            while isinstance(root_array_decl.parent, ArrayDecl):
                root_array_decl = root_array_decl.parent

            init_list = root_array_decl.parent.init
            array_type = root_array_decl.uc_type
            element_type = self.element_type_from_uc_type_array(array_type)
            dims = self.array_dims_from_uc_type_array(array_type)
            id = root_array_decl.parent.name
            # id.scope.gen_location = _varname

            # Allocate on stack memory
            _varname = self.new_reg(node.declname.name)
            inst = (f"alloc_{node.type.name}_" +
                    '_'.join([str(d) for d in dims]), _varname)

            self.current_block.append(inst)
            node.gen_location = _varname
            node.declname.scope.gen_location = _varname

            if init_list is not None:
                init_list_global_var = None
                if isinstance(init_list, InitList):
                    init_list_global_var = self.new_text(f'const_{id.name}')
                    inst = (
                        f'global_{self.array_type_to_str(array_type)}',
                        init_list_global_var,
                        self.init_list_to_values(init_list))
                    self.text.append(inst)
                else:
                    init_list_global_var = self.new_text("str")
                    inst = ("global_string",
                            init_list_global_var, init_list.value)
                    self.text.append(inst)

                self.current_block.append(
                    (f"store_{element_type.typename}_" +
                     '_'.join([str(d) for d in dims]),
                     init_list_global_var, _varname))
        else:
            # Allocate on stack memory
            _varname = self.new_reg(node.declname.name)
            inst = ("alloc_" + node.type.name, _varname)

            self.current_block.append(inst)
            node.gen_location = _varname
            node.declname.scope.gen_location = _varname

            _init = node.parent.init
            if _init is not None:
                self.visit(_init)
                inst = (
                    "store_" + node.type.name,
                    _init.gen_location,
                    node.gen_location,
                )
                self.current_block.append(inst)

    def visit_Program(self, node: Node):
        """
        Start by visiting all global declarations. Then, visit all the function definitions and emit the code stored inside basic blocks.
        """
        # Visit all of the global declarations
        for _decl in node.gdecls:
            self.visit(_decl)
        # At the end of codegen, first init the self.code with
        # the list of global instructions allocated in self.text
        self.code = self.text.copy()
        # Also, copy the global instructions into the Program node
        node.text = self.text.copy()
        # After, visit all the function definitions and emit the
        # code stored inside basic blocks.

        # TODO: Correctly handle function definitions blocks
        bb = EmitBlocks()
        bb.visit(self.current_block)
        self.code += bb.code
        self.debug_print_instructions(self.code)

        # for _decl in node.gdecls:
        #     if isinstance(_decl, FuncDef):
        #         # _decl.cfg contains the Control Flow Graph for the function
        #         # cfg points to start basic block
        #         bb = EmitBlocks()
        #         bb.visit(_decl.cfg)
        #         for _code in bb.code:
        #             self.code.append(_code)

        # if self.viewcfg:  # evaluate to True if -cfg flag is present in command line
        #     for _decl in node.gdecls:
        #         if isinstance(_decl, FuncDef):
        #             dot = CFG(_decl.decl.name.name)
        #             # _decl.cfg contains the CFG for the function
        #             dot.view(_decl.cfg)

    def visit_FuncDef(self, node: FuncDef):
        """
        Initialize the necessary blocks to construct the CFG of the function. Visit the function declaration. Visit all the declarations within the function. After allocating all declarations, visit the arguments initialization. Visit the body of the function to generate its code. Finally, setup the return block correctly and generate the return statement (even for void function).
        """
        self.push_scope()
        # # Create the function block
        # # Use the function name as label
        # node.cfg = BasicBlock(node.decl.name.name)
        # self.current_block = node.cfg
        # self.current_temp = 0  # Set the temporary instruction index

        self.fname = node.decl.name.name
        self.return_temp = None

        # Visit the function declaration
        self.visit(node.decl)

        # Visit the function body
        self.visit(node.body)

        self.current_block.append(("exit:",))

        # For the void function is only necessary the exit to the
        # interpreter work
        if node.type.name != 'void':
            return_var = self.new_temp()
            self.current_block.append(
                (f"load_{node.type.uc_type.typename}",
                 self.return_temp, return_var)
            )
            self.current_block.append(
                (f"return_{node.type.uc_type.typename}", return_var))
        else:
            self.current_block.append(("return_void",))
        self.pop_scope()

    def visit_ParamList(self, node: ParamList):
        """
        Just visit all arguments.
        """
        for param in node.params:
            self.visit(param)

        for i, param in enumerate(node.params):
            inst = (
                f"store_{param.type.uc_type.typename}",
                self.parameters_temp[i],
                param.type.gen_location,
            )
            self.current_block.append(inst)

    def visit_GlobalDecl(self, node: GlobalDecl):
        """
        Visit each global declaration that are not function declarations. Indeed, it is usually simpler to visit the function declaration when visiting the definition of the function to generate all code at once.
        """
        for _decl in node.decls:
            if not isinstance(_decl, FuncDecl) and not isinstance(_decl.type, FuncDecl):
                self.visit(_decl)

                gen_location = self.new_global_reg(_decl.name.name)
                _id: ID = _decl.name
                _id.scope.gen_location = gen_location
                if not isinstance(_decl.type.uc_type, ArrayType):
                    var_type = _decl.type.uc_type.typename
                    inst = (f"global_{var_type}", gen_location)

                    if hasattr(_decl, "init"):
                        parsed_value = self.parse_literal_values(
                            _decl.init.value, var_type)
                        inst += (parsed_value,)

                    self.text.append(inst)
                else:
                    array_type = _decl.type.uc_type
                    init_list = _decl.init
                    inst = (
                        f'global_{self.array_type_to_str(array_type)}',
                        gen_location,
                        self.init_list_to_values(init_list))
                    self.text.append(inst)

    def visit_Decl(self, node: Decl):
        """
        Visit the type of the node (i.e., VarDecl, FuncDecl, etc.).
        """
        self.visit(node.type)

    def visit_ArrayDecl(self, node: Node):
        """
        Visit the node type.
        """
        self.visit(node.type)

    def visit_FuncDecl(self, node: Node):
        """
        Generate the function definition (including function name, return type and arguments types). This is also a good time to generate the entry point for function, allocate a temporary for the return statement (if not a void function), and visit the arguments.
        """
        # Generate Function Definition
        _func_sig: VarDecl = node.type
        _func_param_types = node.uc_type.parameters_type
        self.parameters_temp = [self.new_temp()
                                for _ in range(len(_func_param_types))]
        func_definition = (
            f"define_{_func_sig.type.name}",
            f"@{node.declname.name}",
            [
                (_func_param_types[i].typename, self.parameters_temp[i])
                for i in range(len(_func_param_types))
            ],
        )
        self.current_block.append(func_definition)

        # Generate the Entry point
        entry = ("entry:",)
        self.current_block.append(entry)

        # Generate the temp return
        if _func_sig.type.name != "void":
            self.return_temp = self.new_temp()
            self.current_block.append(
                (f"alloc_{_func_sig.type.name}", self.return_temp)
            )

        # Visit function arguments
        if node.params != None:
            self.visit(node.params)

    def visit_DeclList(self, node: DeclList):
        """
        Visit all of the declarations that appear inside for statement.
        """

        for decl in node.decls:
            self.visit(decl)

    def visit_Type(self, node: Node):
        """
        Do nothing: just pass.
        """
        pass

    def visit_If(self, node: If):
        """
        First, generate the evaluation of the condition (visit it). Create the required blocks and the branch for the condition. Move to the first block and generate the statement related to the then, create the branch to exit. In case, there is an else block, generate it in a similar way.
        """
        self.push_scope()

        # Visit to create the gen_location
        self.visit(node.cond)

        # Create the necessary labels
        then_label = self.new_temp_label("if.then")
        else_label = self.new_temp_label("if.else")
        end_label = self.new_temp_label("if.end")

        inst = ("cbranch", node.cond.gen_location,
                '%' + then_label, '%' + else_label)
        self.current_block.append(inst)

        # Create the if true instruction
        self.current_block.append((f'{then_label}:',))
        self.visit(node.iftrue)
        self.current_block.append(('jump', end_label))

        # Create the if False instructions
        self.current_block.append((f'{else_label}:',))
        if node.iffalse != None:
            self.visit(node.iffalse)

        self.current_block.append((f'{end_label}:',))

        # then_block = BasicBlock(self.new_temp_label(then_label))
        # # self.visit(node.iftrue)

        # then_block.

        # self.current_block.append(ConditionBlock(then_label))
        # self.current_block.append(ConditionBlock(end_label))
        self.pop_scope()

    def visit_For(self, node: For):
        """
        First, generate the initialization of the For and creates all the blocks required. Then, generate the jump to the condition block and generate the condition and the correct conditional branch. Generate the body of the For followed by the jump to the increment block. Generate the increment and the correct jump.
        """
        self.push_scope()

        # First visit the init function to gen the needed code
        self.visit(node.init)

        # Necessary labels to perform the for loop
        cond_label = self.new_temp_label("for.cond")
        body_label = self.new_temp_label("for.body")
        inc_label = self.new_temp_label("for.inc")
        end_label = self.new_temp_label("for.end")

        # Construct the for codition
        self.current_block.append((f'{cond_label}:',))

        self.visit(node.cond)
        cond_inst = ("cbranch", node.cond.gen_location,
                     '%' + body_label, '%' + end_label)
        self.current_block.append(cond_inst)

        # Construct the for body
        self.current_block.append((f'{body_label}:',))
        self.visit(node.body)

        # Construct the for increment
        self.current_block.append((f'{inc_label}:',))
        self.visit(node.next)
        self.current_block.append(('jump', cond_label))

        # Construct the for end
        self.current_block.append((f'{end_label}:',))
        self.pop_scope()

    def visit_While(self, node: While):
        """
        The generation of While is similar to For except that it does not require the part related to initialization and increment.
        """
        self.push_scope()

        # Necessary labels to perform the for loop
        cond_label = self.new_temp_label("while.cond")
        body_label = self.new_temp_label("while.body")
        end_label = self.new_temp_label("while.end")

        # Construct the for codition
        self.current_block.append((f'{cond_label}:',))

        self.visit(node.cond)
        cond_inst = ("cbranch", node.cond.gen_location,
                     '%' + body_label, '%' + end_label)
        self.current_block.append(cond_inst)

        # Construct the for body
        self.current_block.append((f'{body_label}:',))
        self.visit(node.body)
        self.current_block.append(('jump', cond_label))

        # Construct the for end
        self.current_block.append((f'{end_label}:',))
        self.pop_scope()

    def visit_Compound(self, node: Compound):
        """
        Visit the list of block items (declarations or statements).
        """
        self.push_scope()
        for statment in node.staments:
            self.visit(statment)
        self.pop_scope()

    def visit_Assignment(self, node: Assignment):
        """
        First, visit right side and load the value according to its type. Then, visit the left side and generate the code according to the assignment operator and the type of the expression (ID or ArrayRef).
        """

        # Visit the assignmented value
        self.visit(node.rvalue)

        # Is not needed to visit the left side we already
        self.visit(node.lvalue)

        # The following code work if node.lvalue is ID
        # TODO: check if have a better approach to the following code

        rgen = node.rvalue.gen_location
        lgen = node.lvalue.gen_location
        atype = node.lvalue.uc_type.typename

        self.current_block.append(
            (f'store_{atype}', rgen, lgen)
        )

    def visit_Break(self, node: Node):
        """
        Generate a jump instruction to the current exit label.
        """
        pass

    def visit_ArrayRef(self, node: ArrayRef):
        """
        Start by visiting the subscript. Load the values of the index in a new temporary. If the array has multiple dimensions: you need to generate arithmetic instructions to compute the index of the element in the array.
        """
        array_id = node.name
        while not isinstance(array_id, ID):
            array_id = array_id.name

        dims = self.array_dims_from_uc_type_array(array_id.uc_type)

        computed_index_var = None
        if len(dims) > 1:
            internal_array_ref = node
            array_refs = []
            while not isinstance(internal_array_ref, ID):
                array_refs.append(internal_array_ref)
                internal_array_ref = internal_array_ref.name

            current_accumulator_var = None
            for i, array_ref in enumerate(reversed(array_refs)):
                if i < len(dims) - 1:
                    dim_multiplier_var = self.new_temp()
                    self.current_block.append(
                        ('literal_int', dims[i+1], dim_multiplier_var))

                    self.visit(array_ref.subscript)
                    mul_result_var = self.new_temp()
                    self.current_block.append(
                        ('mul_int', array_ref.subscript.gen_location,
                         dim_multiplier_var, mul_result_var))

                    if i > 0:
                        new_accumulator_var = self.new_temp()
                        self.current_block.append(
                            ('add_int', current_accumulator_var,
                             mul_result_var, new_accumulator_var))
                        current_accumulator_var = new_accumulator_var
                    else:
                        current_accumulator_var = mul_result_var
                else:
                    self.visit(array_ref.subscript)

                    new_accumulator_var = self.new_temp()
                    self.current_block.append(
                        ('add_int', current_accumulator_var,
                            array_ref.subscript.gen_location,
                            new_accumulator_var))
                    current_accumulator_var = new_accumulator_var

            computed_index_var = current_accumulator_var
        else:
            self.visit(node.subscript)
            computed_index_var = node.subscript.gen_location

        address_of_element = self.new_temp()

        array_gen_location = array_id.name
        if self.is_global(array_id.name):
            array_gen_location = f'@{array_id.name}'
        else:
            array_gen_location = f'%{array_id.name}'

        self.current_block.append(
            (f'elem_{node.uc_type.typename}',
             array_gen_location,  # array address variable
             computed_index_var,  # index temp variable
             address_of_element)  # address of element at index
        )

        node.gen_location = self.new_temp()
        self.current_block.append(
            (f'load_{node.uc_type.typename}_*',
             address_of_element, node.gen_location)
        )

    def visit_FuncCall(self, node: FuncCall):
        """
        Start by generating the code for the arguments: for each one of them, visit the expression and generate a param_type instruction with its value. Then, allocate a temporary for the return value and generate the code to call the function.
        """
        if isinstance(node.args, ExprList):
            for expr in node.args.exprs:
                self.visit(expr)

            for expr in node.args.exprs:
                self.current_block.append(
                    (f"param_{expr.uc_type.typename}", expr.gen_location)
                )
        else:
            # This is the node ID case
            node_id: ID = node.args
            self.visit(node_id)
            self.current_block.append(
                (f"param_{node_id.uc_type.typename}", node_id.gen_location)
            )

        node.gen_location = self.new_temp()
        self.current_block.append(
            (f"call_{node.uc_type.typename}",
             f'@{node.name.name}', node.gen_location)
        )

    def visit_Assert(self, node: Assert):
        """
        The assert is a conditional statement which generate code quite similar to the If Block. If the expression is false, the program should issue an error message (assertfail) and terminate. If the expression is true, the program proceeds to the next sentence.

        Visit the assert condition. Create the blocks for the condition and adust their predecessors. Generate the branch instruction and adjust the blocks to jump according to the condition. Generate the code for unsuccessful assert, generate the print instruction and the jump instruction to the return block, and successful assert.
        """
        self.visit(node.expr)

        # Get the expression gen_location
        egen = node.expr.gen_location

        # Next label, for while is only exit
        next_label = 'exit'

        # Create the assert labels
        assert_true = self.new_temp_label("assert.true")
        assert_false = self.new_temp_label("assert.false")

        cbranch_inst = ("cbranch", egen, "%" + assert_true, "%" + assert_false)
        self.current_block.append(cbranch_inst)

        # Create the assert False
        self.current_block.append((f'{assert_false}:',))
        str_to_print = self.new_text("str")
        coord = str(node.expr.coord).split(" ")[1]
        fail_msg = (f'global_string', str_to_print,
                    f'assertion_fail on {coord}')
        self.text.append(fail_msg)
        self.current_block.append(("print_string", str_to_print))
        # Todo adjust to the correct block
        self.current_block.append(('jump', next_label))

        # Create the assert True
        self.current_block.append((f'{assert_true}:',))
        temp = self.new_temp()
        self.current_block.append(('literal_int', 0, temp))
        self.current_block.append(('store_int', temp, "%1"))
        # self.current_block.append(('jump', next_label))

    def visit_EmptyStatement(self, node: Node):
        pass

    def visit_Read(self, node: Node):
        pass

    def visit_Return(self, node: Return):
        '''
        If there is an expression, you need to visit it, load it if necessary and store its value to the return location. Then generate a jump to the return block if needed. Do not forget to update the predecessor of the return block.
        '''
        if node.expr is not None:
            self.visit(node.expr)

            self.current_block.append(
                (
                    f"store_{node.expr.uc_type.typename}",
                    node.expr.gen_location,
                    self.return_temp,
                )
            )

        self.current_block.append(("jump", "exit"))

    def visit_Constant(self, node: Constant):
        '''
        If the constant is of type string, create a new global that will contain the value. Otherwise just create a new temporary initialized with the value.
        '''

        # In the case for string is only need to create
        # a global variable in the text region
        if node.uc_type.typename == "string":
            _target = self.new_text("str")
            inst = ("global_string", _target, node.value)
            self.text.append(inst)

        else:
            _target = self.new_temp()
            parsed_value = self.parse_literal_values(
                node.value, node.uc_type.typename)

            self.current_block.append(
                (f"literal_{node.uc_type.typename}", parsed_value, _target)
            )

        node.gen_location = _target

    def visit_ID(self, node: ID):
        if hasattr(node, "parent") and \
                isinstance(node.parent, Decl) or isinstance(node.parent, Assignment):
            # Handle code generation for declarions on Decl node
            node.gen_location = node.scope.gen_location
        else:
            # BRUNO TODO: I think that is not the best function to load the
            # the function description in the notebook has no metion about this
            node.gen_location = self.new_temp()
            # If a global or constant var use @
            _var_name = f'{node.scope.gen_location}'
            self.current_block.append(
                (f"load_{node.uc_type.typename}", _var_name, node.gen_location)
            )

    def visit_UnaryOp(self, node: UnaryOp):

        # First visit expr
        self.visit(node.expr)

        opcode = self.binary_ops[node.op] + '_' + node.uc_type.typename

        # TODO Check if the following is the best approach
        # to threat with negative integers

        inst: Tuple = None
        if node.op == '-':
            # This istantiate a new temp %k as zero
            # to perform node.gen = %k - node.expr

            # Instantiate the literal zero
            zero_gen_location = self.new_temp()
            inst_zero = (
                f'literal_{node.uc_type.typename}', 0, zero_gen_location)
            self.current_block.append(inst_zero)

            node.gen_location = self.new_temp()
            inst = (opcode, zero_gen_location,
                    node.expr.gen_location, node.gen_location)

        elif node.op == "!":
            node.gen_location = self.new_temp()
            inst = (opcode, node.expr.gen_location, node.gen_location)

        self.current_block.append(inst)

    def visit_ExprList(self, node: Node):
        pass

    def visit_InitList(self, node: Node):
        pass


def main():
    # create argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Path to file to be used to generate uCIR. By default, this script only runs the interpreter on the uCIR. \
              Use the other options for printing the uCIR, generating the CFG or for the debug mode.",
        type=str,
    )
    parser.add_argument(
        "--ir",
        help="Print uCIR generated from input_file.",
        action="store_true",
    )
    parser.add_argument(
        "--cfg", help="Show the cfg of the input_file.", action="store_true"
    )
    parser.add_argument(
        "--debug", help="Run interpreter in debug mode.", action="store_true"
    )
    parser.add_argument("--debug_print", action="store_true")

    args = parser.parse_args()

    print_ir = args.ir
    create_cfg = args.cfg
    interpreter_debug = args.debug
    enable_stdout_debug_print = args.debug_print

    # get input path
    input_file = args.input_file
    input_path = pathlib.Path(input_file)

    # check if file exists
    if not input_path.exists():
        print("Input", input_path, "not found", file=sys.stderr)
        sys.exit(1)

    # set error function
    p = UCParser()
    # open file and parse it
    with open(input_path) as f:
        ast = p.parse(f.read())

    sema = Visitor()
    sema.visit(ast)

    gen = CodeGenerator(create_cfg, enable_stdout_debug_print)
    gen.visit(ast)
    gencode = gen.code

    if print_ir:
        print("Generated uCIR: --------")
        gen.show()
        print("------------------------\n")

    vm = Interpreter(interpreter_debug)
    vm.run(gencode)


if __name__ == "__main__":
    main()
