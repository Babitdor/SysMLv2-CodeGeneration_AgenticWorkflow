"""Semantic validator for SysML v2 code.

Walks the ANTLR parse tree (new grammar: OMG 2026-03 + patches) to build a
symbol table, then checks:
- Reference resolution: all referenced names must resolve to defined symbols
- Type checking: `satisfy` must reference a requirement, not a part
- Structural rules: `subject` in requirements must reference a valid element
- Duplicate definitions: no two definitions with the same name in the same scope
- Feature chain resolution: `part.feature` chains resolve through child scopes
- Perform action references: `perform <action>` resolves to an action definition
- Flow/connector references: flow ends and connector ends resolve to valid features
- Built-in symbols: SI units and base types are pre-populated in the root scope
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from antlr4 import ParseTreeWalker

from .NewGrammar.SysMLv2ParserListener import SysMLv2ParserListener
from .NewGrammar.SysMLv2Parser import SysMLv2Parser
from .SemanticError import SemanticError

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Symbol table types
# ---------------------------------------------------------------------------


class SymbolKind(Enum):
    """What kind of SysML element a symbol represents."""

    PACKAGE = "package"
    PART_DEF = "part_def"
    PART_USAGE = "part_usage"
    PORT_DEF = "port_def"
    PORT_USAGE = "port_usage"
    REQUIREMENT_DEF = "requirement_def"
    REQUIREMENT_USAGE = "requirement_usage"
    ATTRIBUTE_DEF = "attribute_def"
    ATTRIBUTE_USAGE = "attribute_usage"
    ITEM_DEF = "item_def"
    ITEM_USAGE = "item_usage"
    CONNECTION_DEF = "connection_def"
    CONNECTION_USAGE = "connection_usage"
    ACTION_DEF = "action_def"
    ACTION_USAGE = "action_usage"
    STATE_DEF = "state_def"
    CONSTRAINT_DEF = "constraint_def"
    CONSTRAINT_USAGE = "constraint_usage"
    ENUMERATION_DEF = "enumeration_def"
    INTERFACE_DEF = "interface_def"
    ALLOCATION_DEF = "allocation_def"
    CALCULATION_DEF = "calculation_def"
    CONCERN_DEF = "concern_def"
    CASE_DEF = "case_def"
    IMPORT = "import"
    BUILTIN = "builtin"
    UNKNOWN = "unknown"


# Definitions that can be referenced by `satisfy`
_REQUIREMENT_KINDS = {SymbolKind.REQUIREMENT_DEF, SymbolKind.REQUIREMENT_USAGE}

# Definitions that can be referenced by `subject`
_SUBJECT_KINDS = {
    SymbolKind.PART_DEF,
    SymbolKind.PART_USAGE,
    SymbolKind.ITEM_DEF,
    SymbolKind.ITEM_USAGE,
    SymbolKind.UNKNOWN,
}

# Definitions that can be referenced by `perform`
_ACTION_KINDS = {
    SymbolKind.ACTION_DEF,
    SymbolKind.ACTION_USAGE,
}

# Kinds that define a scope (have a body with child features)
_SCOPE_DEFINING_KINDS = {
    SymbolKind.PART_DEF,
    SymbolKind.PART_USAGE,
    SymbolKind.PORT_DEF,
    SymbolKind.ACTION_DEF,
    SymbolKind.ACTION_USAGE,
    SymbolKind.ITEM_DEF,
    SymbolKind.ITEM_USAGE,
    SymbolKind.CONNECTION_DEF,
    SymbolKind.INTERFACE_DEF,
    SymbolKind.STATE_DEF,
    SymbolKind.REQUIREMENT_DEF,
    SymbolKind.CONSTRAINT_DEF,
    SymbolKind.ATTRIBUTE_DEF,
}

# All definition kinds (as opposed to usage kinds)
_DEFINITION_KINDS = {
    SymbolKind.PART_DEF,
    SymbolKind.PORT_DEF,
    SymbolKind.REQUIREMENT_DEF,
    SymbolKind.ATTRIBUTE_DEF,
    SymbolKind.ITEM_DEF,
    SymbolKind.CONNECTION_DEF,
    SymbolKind.ACTION_DEF,
    SymbolKind.STATE_DEF,
    SymbolKind.CONSTRAINT_DEF,
    SymbolKind.ENUMERATION_DEF,
    SymbolKind.INTERFACE_DEF,
    SymbolKind.ALLOCATION_DEF,
    SymbolKind.CALCULATION_DEF,
    SymbolKind.CONCERN_DEF,
    SymbolKind.CASE_DEF,
    SymbolKind.PACKAGE,
}

# ---------------------------------------------------------------------------
# Built-in SI units and base types
# ---------------------------------------------------------------------------

# SI base units
_SI_BASE_UNITS = [
    "m",
    "s",
    "kg",
    "K",
    "A",
    "mol",
    "cd",
]

# SI derived units commonly used in SysML models
_SI_DERIVED_UNITS = [
    "Pa",
    "N",
    "J",
    "W",
    "V",
    "F",
    "Ohm",
    "C",
    "Hz",
    "lbf",
    "lb",
    "ft",
    "in",
    "mi",
    "km",
    "mm",
    "cm",
    "ms",
    "us",
    "ns",
    "min",
    "h",
    "L",
    "mL",
    "gal",
    "deg",
    "rad",
    "sr",
    "GPa",
    "MPa",
    "kPa",
    "kN",
    "MN",
    "kJ",
    "MJ",
    "kWh",
    "kW",
    "MW",
    "mV",
    "kV",
    "rpm",
    "g",
    "N_m",
    "Nm",
    "kNm",
    "m_s",
    "m_s2",
    "kg_m3",
]

# Common SysML base types from the standard library
_BUILTIN_TYPES = [
    "Anything",
    "ScalarValue",
    "StructuralValue",
    "Real",
    "Integer",
    "Boolean",
    "String",
    "Number",
    "Complex",
    "Positive",
    "NonNegative",
    "Negative",
    "NonPositive",
    "DateTime",
    "Duration",
    "Mass",
    "Length",
    "Time",
    "Temperature",
    "Current",
    "AmountOfSubstance",
    "LuminousIntensity",
    "Area",
    "Volume",
    "Speed",
    "Acceleration",
    "Force",
    "Pressure",
    "Energy",
    "Power",
    "Frequency",
    "Angle",
    "SolidAngle",
    "ElectricCharge",
    "Voltage",
    "Capacitance",
    "Resistance",
    "Conductance",
    "MagneticFlux",
    "MagneticFluxDensity",
    "Inductance",
    "CatalyticActivity",
    "AbsorbedDose",
    "EquivalentDose",
    "Radioactivity",
    "LuminousFlux",
]

# Action control flow pseudo-names (used in `then done;` and `first start;`)
_ACTION_CONTROL_FLOW_NAMES = [
    "done",
    "start",
]


@dataclass
class Symbol:
    """A named symbol in the symbol table."""

    name: str
    kind: SymbolKind
    line: int = 0
    column: int = 0
    scope: Optional["Scope"] = None


@dataclass
class Scope:
    """A naming scope (package, definition body, etc.)."""

    name: str
    parent: Optional["Scope"] = None
    symbols: dict[str, Symbol] = field(default_factory=dict)
    children: list["Scope"] = field(default_factory=list)

    def define(self, symbol: Symbol) -> Optional[SemanticError]:
        """Add a symbol to this scope. Returns an error if duplicate."""
        if symbol.name in self.symbols:
            existing = self.symbols[symbol.name]
            return SemanticError(
                message=(
                    f"Duplicate definition '{symbol.name}': "
                    f"already defined as {existing.kind.value} "
                    f"at line {existing.line}"
                ),
                line=symbol.line,
                column=symbol.column,
                severity="error",
                error_type="structural",
            )
        self.symbols[symbol.name] = symbol
        symbol.scope = self
        return None

    def lookup(self, name: str) -> Optional[Symbol]:
        """Look up a name in this scope and all parent scopes."""
        if "::" in name:
            return self._lookup_qualified(name)

        scope: Optional[Scope] = self
        while scope is not None:
            if name in scope.symbols:
                return scope.symbols[name]
            scope = scope.parent
        return None

    def lookup_global(self, name: str) -> Optional[Symbol]:
        """Look up a name across ALL scopes (breadth-first from root)."""
        if "::" in name:
            return self._lookup_qualified(name)

        from collections import deque

        queue: deque[Scope] = deque([self._find_root()])
        visited: set[int] = set()

        while queue:
            scope = queue.popleft()
            if id(scope) in visited:
                continue
            visited.add(id(scope))

            if name in scope.symbols:
                return scope.symbols[name]

            queue.extend(scope.children)

        return None

    def lookup_feature_chain(self, chain: str) -> Optional[Symbol]:
        """Look up a feature chain like 'partName.featureName.subFeature'."""
        segments = chain.split(".")
        if not segments:
            return None

        first = segments[0]
        symbol = self.lookup_global(first)
        if symbol is None:
            return None

        for segment in segments[1:]:
            child_scope = self._get_child_scope_for_symbol(symbol)
            if child_scope is None:
                return None
            if segment not in child_scope.symbols:
                return None
            symbol = child_scope.symbols[segment]

        return symbol

    def resolve_feature_chain_segments(
        self, chain: str
    ) -> tuple[Optional[Symbol], Optional[str]]:
        """Resolve a feature chain, returning (resolved_symbol, failed_segment)."""
        segments = chain.split(".")
        if not segments:
            return None, None

        first = segments[0]
        symbol = self.lookup_global(first)
        if symbol is None:
            return None, first

        for segment in segments[1:]:
            child_scope = self._get_child_scope_for_symbol(symbol)
            if child_scope is None:
                return symbol, segment
            if segment not in child_scope.symbols:
                return symbol, segment
            symbol = child_scope.symbols[segment]

        return symbol, None

    def _get_child_scope_for_symbol(self, symbol: Symbol) -> Optional["Scope"]:
        """Find the child scope that was created when a symbol was defined."""
        if symbol.scope is None:
            return None

        for child in symbol.scope.children:
            if child.name == symbol.name:
                return child

        return None

    def _lookup_qualified(self, qualified_name: str) -> Optional[Symbol]:
        """Look up a qualified name like 'Pkg::SubPkg::Name'."""
        parts = qualified_name.split("::")
        scope: Optional[Scope] = self._find_root()
        for part in parts[:-1]:
            if scope is None or part not in scope.symbols:
                return None
            sym = scope.symbols[part]
            if sym.scope is not None and sym.name in sym.scope.symbols:
                child = sym.scope
                scope = None
                for c in child.children:
                    if c.name == part:
                        scope = c
                        break
                if scope is None:
                    scope = child
            else:
                return None
        if scope is not None and parts[-1] in scope.symbols:
            return scope.symbols[parts[-1]]
        return None

    def _find_root(self) -> "Scope":
        """Walk up to the root scope."""
        scope = self
        while scope.parent is not None:
            scope = scope.parent
        return scope


# ---------------------------------------------------------------------------
# Reference tracking
# ---------------------------------------------------------------------------


@dataclass
class Reference:
    """A reference to a symbol that needs resolution."""

    name: str
    kind: str  # "satisfy", "subject", "type_ref", "specialization", "subsetting",
    # "perform_action", "flow_end", "connector_end", "feature_chain",
    # "feature_ref"
    line: int = 0
    column: int = 0
    context: Optional[SymbolKind] = None  # What kind of element contains this ref
    parent_ref_name: str = (
        ""  # For feature chains: the parent part name (first segment)
    )


# ---------------------------------------------------------------------------
# Semantic Validator (tree walker)
# ---------------------------------------------------------------------------


class SemanticValidator(SysMLv2ParserListener):
    """Walks the ANTLR parse tree (new grammar) to build a symbol table and
    collect references, then checks semantic constraints after the walk."""

    def __init__(self) -> None:
        self.root_scope = Scope(name="<root>")
        self.current_scope = self.root_scope
        self.references: list[Reference] = []
        self.errors: list[SemanticError] = []
        self._scope_stack: list[Scope] = [self.root_scope]

        # Track feature chain expressions: when an ownedExpression contains
        # DOT tokens chaining qualifiedNames, we suppress individual refs
        # and create a single feature_chain reference on exit.
        self._in_feature_chain_expr: bool = False

        # Pre-populate root scope with built-in symbols
        self._populate_builtin_symbols()

    def _populate_builtin_symbols(self) -> None:
        """Add built-in SI units and base types to the root scope."""
        for unit in _SI_BASE_UNITS + _SI_DERIVED_UNITS:
            self.root_scope.define(
                Symbol(name=unit, kind=SymbolKind.BUILTIN, line=0, column=0)
            )
        for type_name in _BUILTIN_TYPES:
            self.root_scope.define(
                Symbol(name=type_name, kind=SymbolKind.BUILTIN, line=0, column=0)
            )
        for name in _ACTION_CONTROL_FLOW_NAMES:
            self.root_scope.define(
                Symbol(name=name, kind=SymbolKind.BUILTIN, line=0, column=0)
            )

    # -- Scope management ---------------------------------------------------

    def _push_scope(self, name: str) -> Scope:
        """Enter a new nested scope."""
        scope = Scope(name=name, parent=self.current_scope)
        self.current_scope.children.append(scope)
        self._scope_stack.append(scope)
        self.current_scope = scope
        return scope

    def _pop_scope(self) -> None:
        """Leave the current scope, returning to the parent."""
        if len(self._scope_stack) > 1:
            self._scope_stack.pop()
            self.current_scope = self._scope_stack[-1]

    # -- Helper: extract name/text from context ------------------------------

    @staticmethod
    def _get_text(ctx) -> str:
        """Get the text from a parse tree context."""
        return ctx.getText() if ctx else ""

    @staticmethod
    def _get_line(ctx) -> int:
        """Get the line number from a parse tree context."""
        try:
            return ctx.start.line if ctx and hasattr(ctx, "start") else 0
        except AttributeError:
            return 0

    @staticmethod
    def _get_column(ctx) -> int:
        """Get the column number from a parse tree context."""
        try:
            return ctx.start.column if ctx and hasattr(ctx, "start") else 0
        except AttributeError:
            return 0

    def _extract_name(self, ctx) -> str:
        """Extract a name from an identification or name context.

        New grammar: identification is <Name> or <visibility Name>.
        The name() method returns the NameContext.
        """
        if ctx is None:
            return ""
        # Try the name() method (identification has a name() child)
        try:
            name_ctx = ctx.name()
            if name_ctx:
                return name_ctx.getText()
        except (AttributeError, TypeError):
            pass
        # Fallback: get text and strip angle brackets
        text = ctx.getText()
        if text.startswith("<") and ">" in text:
            parts = text.strip("<>").split()
            return parts[-1] if parts else text
        return text

    def _extract_qualified_name(self, ctx) -> str:
        """Extract a qualified name from a qualifiedName context."""
        if ctx is None:
            return ""
        return ctx.getText()

    # -- Helper: extract name from definition contexts -----------------------

    def _extract_definition_name(self, ctx) -> str:
        """Extract the name from a definition context.

        New grammar structure varies by definition type:
        - partDefinition -> definition -> definitionDeclaration -> identification -> name
        - actionDefinition -> definitionDeclaration -> identification -> name
        - Some definitions have definitionDeclaration directly
        """
        # Try definition() -> definitionDeclaration() -> identification()
        try:
            defn = ctx.definition()
            if defn:
                # definition() may return a list in some contexts
                if isinstance(defn, list):
                    for d in defn:
                        decl = (
                            d.definitionDeclaration()
                            if hasattr(d, "definitionDeclaration")
                            else None
                        )
                        if decl:
                            ident = decl.identification()
                            if ident:
                                return self._extract_name(ident)
                else:
                    decl = defn.definitionDeclaration()
                    if decl:
                        ident = decl.identification()
                        if ident:
                            return self._extract_name(ident)
        except AttributeError:
            pass

        # Try definitionDeclaration() directly
        try:
            decl = ctx.definitionDeclaration()
            if decl:
                ident = decl.identification()
                if ident:
                    return self._extract_name(ident)
        except AttributeError:
            pass

        return ""

    def _extract_usage_name(self, ctx) -> str:
        """Extract the name from a usage context.

        New grammar structure varies by usage type:
          - Most usages: usage() -> usageDeclaration() -> identification() -> name()
          - ActionUsage: actionUsageDeclaration() -> usageDeclaration() -> identification() -> name()
          - ConnectionUsage: usageDeclaration() -> identification() -> name()
          - ConstraintUsage: constraintUsageDeclaration() -> usageDeclaration() -> identification() -> name()
          - RequirementUsage: constraintUsageDeclaration() -> usageDeclaration() -> identification() -> name()
        """
        # Try usage() -> usageDeclaration() (most usage types)
        try:
            usage = ctx.usage()
            if usage:
                decl = usage.usageDeclaration()
                if decl:
                    ident = decl.identification()
                    if ident:
                        return self._extract_name(ident)
        except AttributeError:
            pass

        # Try actionUsageDeclaration() -> usageDeclaration() (ActionUsage)
        try:
            au_decl = ctx.actionUsageDeclaration()
            if au_decl:
                decl = au_decl.usageDeclaration()
                if decl:
                    ident = decl.identification()
                    if ident:
                        return self._extract_name(ident)
        except AttributeError:
            pass

        # Try constraintUsageDeclaration() -> usageDeclaration() (ConstraintUsage, RequirementUsage)
        try:
            cu_decl = ctx.constraintUsageDeclaration()
            if cu_decl:
                decl = cu_decl.usageDeclaration()
                if decl:
                    ident = decl.identification()
                    if ident:
                        return self._extract_name(ident)
        except AttributeError:
            pass

        # Try usageDeclaration() directly (ConnectionUsage, SatisfyRequirementUsage)
        try:
            decl = ctx.usageDeclaration()
            if decl:
                ident = decl.identification()
                if ident:
                    return self._extract_name(ident)
        except AttributeError:
            pass

        return ""

    # -- Symbol definition listeners: Packages -------------------------------

    def enterPackage(self, ctx: SysMLv2Parser.PackageContext):
        """Enter a package definition — push a new scope."""
        name = self._extract_package_name(ctx)
        if name:
            symbol = Symbol(
                name=name,
                kind=SymbolKind.PACKAGE,
                line=self._get_line(ctx),
                column=self._get_column(ctx),
            )
            err = self.current_scope.define(symbol)
            if err:
                self.errors.append(err)
            self._push_scope(name)

    def exitPackage(self, ctx: SysMLv2Parser.PackageContext):
        """Exit a package definition — pop scope."""
        self._pop_scope()

    def _extract_package_name(self, ctx) -> str:
        """Extract the package name from a package context."""
        try:
            decl = ctx.packageDeclaration()
            if decl:
                ident = decl.identification()
                if ident:
                    return self._extract_name(ident)
        except AttributeError:
            pass
        return ""

    # -- Definition listeners ------------------------------------------------

    def _enter_definition(self, ctx, kind: SymbolKind) -> None:
        """Generic handler for entering a definition (part def, port def, etc.)."""
        name = self._extract_definition_name(ctx)
        if name:
            symbol = Symbol(
                name=name,
                kind=kind,
                line=self._get_line(ctx),
                column=self._get_column(ctx),
            )
            err = self.current_scope.define(symbol)
            if err:
                self.errors.append(err)
            self._push_scope(name)

    def _exit_definition(self, ctx) -> None:
        """Generic handler for exiting a definition."""
        self._pop_scope()

    # Part definitions
    def enterPartDefinition(self, ctx: SysMLv2Parser.PartDefinitionContext):
        self._enter_definition(ctx, SymbolKind.PART_DEF)

    def exitPartDefinition(self, ctx: SysMLv2Parser.PartDefinitionContext):
        self._pop_scope()

    # Port definitions
    def enterPortDefinition(self, ctx: SysMLv2Parser.PortDefinitionContext):
        self._enter_definition(ctx, SymbolKind.PORT_DEF)

    def exitPortDefinition(self, ctx: SysMLv2Parser.PortDefinitionContext):
        self._pop_scope()

    # Requirement definitions
    def enterRequirementDefinition(
        self, ctx: SysMLv2Parser.RequirementDefinitionContext
    ):
        self._enter_definition(ctx, SymbolKind.REQUIREMENT_DEF)

    def exitRequirementDefinition(
        self, ctx: SysMLv2Parser.RequirementDefinitionContext
    ):
        self._pop_scope()

    # Attribute definitions
    def enterAttributeDefinition(self, ctx: SysMLv2Parser.AttributeDefinitionContext):
        self._enter_definition(ctx, SymbolKind.ATTRIBUTE_DEF)

    def exitAttributeDefinition(self, ctx: SysMLv2Parser.AttributeDefinitionContext):
        self._pop_scope()

    # Item definitions
    def enterItemDefinition(self, ctx: SysMLv2Parser.ItemDefinitionContext):
        self._enter_definition(ctx, SymbolKind.ITEM_DEF)

    def exitItemDefinition(self, ctx: SysMLv2Parser.ItemDefinitionContext):
        self._pop_scope()

    # Connection definitions
    def enterConnectionDefinition(self, ctx: SysMLv2Parser.ConnectionDefinitionContext):
        self._enter_definition(ctx, SymbolKind.CONNECTION_DEF)

    def exitConnectionDefinition(self, ctx: SysMLv2Parser.ConnectionDefinitionContext):
        self._pop_scope()

    # Action definitions
    def enterActionDefinition(self, ctx: SysMLv2Parser.ActionDefinitionContext):
        self._enter_definition(ctx, SymbolKind.ACTION_DEF)

    def exitActionDefinition(self, ctx: SysMLv2Parser.ActionDefinitionContext):
        self._pop_scope()

    # State definitions
    def enterStateDefinition(self, ctx: SysMLv2Parser.StateDefinitionContext):
        self._enter_definition(ctx, SymbolKind.STATE_DEF)

    def exitStateDefinition(self, ctx: SysMLv2Parser.StateDefinitionContext):
        self._pop_scope()

    # Constraint definitions
    def enterConstraintDefinition(self, ctx: SysMLv2Parser.ConstraintDefinitionContext):
        self._enter_definition(ctx, SymbolKind.CONSTRAINT_DEF)

    def exitConstraintDefinition(self, ctx: SysMLv2Parser.ConstraintDefinitionContext):
        self._pop_scope()

    # Enumeration definitions
    def enterEnumerationDefinition(
        self, ctx: SysMLv2Parser.EnumerationDefinitionContext
    ):
        self._enter_definition(ctx, SymbolKind.ENUMERATION_DEF)

    def exitEnumerationDefinition(
        self, ctx: SysMLv2Parser.EnumerationDefinitionContext
    ):
        self._pop_scope()

    # Interface definitions
    def enterInterfaceDefinition(self, ctx: SysMLv2Parser.InterfaceDefinitionContext):
        self._enter_definition(ctx, SymbolKind.INTERFACE_DEF)

    def exitInterfaceDefinition(self, ctx: SysMLv2Parser.InterfaceDefinitionContext):
        self._pop_scope()

    # Allocation definitions
    def enterAllocationDefinition(self, ctx: SysMLv2Parser.AllocationDefinitionContext):
        self._enter_definition(ctx, SymbolKind.ALLOCATION_DEF)

    def exitAllocationDefinition(self, ctx: SysMLv2Parser.AllocationDefinitionContext):
        self._pop_scope()

    # Calculation definitions
    def enterCalculationDefinition(
        self, ctx: SysMLv2Parser.CalculationDefinitionContext
    ):
        self._enter_definition(ctx, SymbolKind.CALCULATION_DEF)

    def exitCalculationDefinition(
        self, ctx: SysMLv2Parser.CalculationDefinitionContext
    ):
        self._pop_scope()

    # Concern definitions
    def enterConcernDefinition(self, ctx: SysMLv2Parser.ConcernDefinitionContext):
        self._enter_definition(ctx, SymbolKind.CONCERN_DEF)

    def exitConcernDefinition(self, ctx: SysMLv2Parser.ConcernDefinitionContext):
        self._pop_scope()

    # Case definitions (analysis/verification)
    def enterAnalysisCaseDefinition(
        self, ctx: SysMLv2Parser.AnalysisCaseDefinitionContext
    ):
        self._enter_definition(ctx, SymbolKind.CASE_DEF)

    def exitAnalysisCaseDefinition(
        self, ctx: SysMLv2Parser.AnalysisCaseDefinitionContext
    ):
        self._pop_scope()

    def enterVerificationCaseDefinition(
        self, ctx: SysMLv2Parser.VerificationCaseDefinitionContext
    ):
        self._enter_definition(ctx, SymbolKind.CASE_DEF)

    def exitVerificationCaseDefinition(
        self, ctx: SysMLv2Parser.VerificationCaseDefinitionContext
    ):
        self._pop_scope()

    # -- Usage listeners -----------------------------------------------------

    def _enter_usage(self, ctx, kind: SymbolKind) -> None:
        """Generic handler for entering a usage (part, port, etc.)."""
        name = self._extract_usage_name(ctx)
        if name:
            symbol = Symbol(
                name=name,
                kind=kind,
                line=self._get_line(ctx),
                column=self._get_column(ctx),
            )
            err = self.current_scope.define(symbol)
            if err:
                self.errors.append(err)

    # Part usages
    def enterPartUsage(self, ctx: SysMLv2Parser.PartUsageContext):
        self._enter_usage(ctx, SymbolKind.PART_USAGE)

    # Port usages
    def enterPortUsage(self, ctx: SysMLv2Parser.PortUsageContext):
        self._enter_usage(ctx, SymbolKind.PORT_USAGE)

    # Requirement usages
    def enterRequirementUsage(self, ctx: SysMLv2Parser.RequirementUsageContext):
        """Handle `requirement <Name> : <ReqDef>` — add the usage name to the symbol table."""
        name = self._extract_usage_name(ctx)
        if name:
            symbol = Symbol(
                name=name,
                kind=SymbolKind.REQUIREMENT_USAGE,
                line=self._get_line(ctx),
                column=self._get_column(ctx),
            )
            err = self.current_scope.define(symbol)
            if err:
                self.errors.append(err)

        # Track the type reference
        type_name = self._extract_usage_type_ref(ctx)
        if type_name:
            self.references.append(
                Reference(
                    name=type_name,
                    kind="type_ref",
                    line=self._get_line(ctx),
                    column=self._get_column(ctx),
                    context=SymbolKind.REQUIREMENT_USAGE,
                )
            )

    # Attribute usages
    def enterAttributeUsage(self, ctx: SysMLv2Parser.AttributeUsageContext):
        self._enter_usage(ctx, SymbolKind.ATTRIBUTE_USAGE)

    # Item usages
    def enterItemUsage(self, ctx: SysMLv2Parser.ItemUsageContext):
        self._enter_usage(ctx, SymbolKind.ITEM_USAGE)

    # Connection usages
    def enterConnectionUsage(self, ctx: SysMLv2Parser.ConnectionUsageContext):
        self._enter_usage(ctx, SymbolKind.CONNECTION_USAGE)

    # Action usages — push a scope if the action has a body so that nested
    # parameters (in/out) don't collide with the parent scope.
    def enterActionUsage(self, ctx: SysMLv2Parser.ActionUsageContext):
        name = self._extract_usage_name(ctx)
        if name:
            symbol = Symbol(
                name=name,
                kind=SymbolKind.ACTION_USAGE,
                line=self._get_line(ctx),
                column=self._get_column(ctx),
            )
            err = self.current_scope.define(symbol)
            if err:
                self.errors.append(err)
            # Push a child scope only if the action usage has a body
            if ctx.actionBody():
                self._push_scope(name)

    def exitActionUsage(self, ctx: SysMLv2Parser.ActionUsageContext):
        if ctx.actionBody():
            self._pop_scope()

    # Constraint usages
    def enterConstraintUsage(self, ctx: SysMLv2Parser.ConstraintUsageContext):
        self._enter_usage(ctx, SymbolKind.CONSTRAINT_USAGE)

    # -- Helper: extract type reference from usage ---------------------------

    def _extract_usage_type_ref(self, ctx) -> str:
        """Extract the type reference from a usage context.

        Tries multiple declaration paths depending on usage type:
          - Most: usage() -> usageDeclaration() -> featureSpecializationPart()
          - ActionUsage: actionUsageDeclaration() -> usageDeclaration() -> ...
          - ConstraintUsage/RequirementUsage: constraintUsageDeclaration() -> usageDeclaration() -> ...
          - ConnectionUsage: usageDeclaration() -> ...
        """
        # Try usage() -> usageDeclaration() (most usage types)
        try:
            usage = ctx.usage()
            if usage:
                decl = usage.usageDeclaration()
                if decl:
                    spec_part = decl.featureSpecializationPart()
                    if spec_part:
                        return self._extract_type_from_spec_part(spec_part)
        except AttributeError:
            pass

        # Try actionUsageDeclaration() -> usageDeclaration() (ActionUsage)
        try:
            au_decl = ctx.actionUsageDeclaration()
            if au_decl:
                decl = au_decl.usageDeclaration()
                if decl:
                    spec_part = decl.featureSpecializationPart()
                    if spec_part:
                        return self._extract_type_from_spec_part(spec_part)
        except AttributeError:
            pass

        # Try constraintUsageDeclaration() -> usageDeclaration() (ConstraintUsage, RequirementUsage)
        try:
            cu_decl = ctx.constraintUsageDeclaration()
            if cu_decl:
                decl = cu_decl.usageDeclaration()
                if decl:
                    spec_part = decl.featureSpecializationPart()
                    if spec_part:
                        return self._extract_type_from_spec_part(spec_part)
        except AttributeError:
            pass

        # Try usageDeclaration() directly (ConnectionUsage, SatisfyRequirementUsage)
        try:
            decl = ctx.usageDeclaration()
            if decl:
                spec_part = decl.featureSpecializationPart()
                if spec_part:
                    return self._extract_type_from_spec_part(spec_part)
        except AttributeError:
            pass

        return ""

    def _extract_type_from_spec_part(self, spec_part) -> str:
        """Extract type name from a featureSpecializationPart.

        New grammar path: featureSpecialization → typings → typedBy
        → featureTyping → ownedFeatureTyping → qualifiedName
        """
        try:
            for spec in spec_part.featureSpecialization():
                typings = spec.typings()
                if typings:
                    typed_by = typings.typedBy()
                    if typed_by:
                        ft = typed_by.featureTyping()
                        if ft:
                            owned_ft = ft.ownedFeatureTyping()
                            if owned_ft:
                                qname = owned_ft.qualifiedName()
                                # qualifiedName may return a list
                                if isinstance(qname, list):
                                    qname = qname[0] if qname else None
                                if qname:
                                    return qname.getText()
        except AttributeError:
            pass
        return ""

    # -- Satisfy requirement usage listener ---------------------------------

    def enterSatisfyRequirementUsage(
        self, ctx: SysMLv2Parser.SatisfyRequirementUsageContext
    ):
        """Handle `satisfy <requirement> by <subject>` — check that the
        referenced name is a requirement definition or usage."""
        ref_name = ""
        line = self._get_line(ctx)
        col = self._get_column(ctx)

        # New grammar: satisfyRequirementUsage has ownedReferenceSubsetting
        # which contains the qualified name of the requirement being satisfied
        try:
            ref_sub = ctx.ownedReferenceSubsetting()
            if ref_sub:
                if isinstance(ref_sub, list):
                    ref_sub = ref_sub[0] if ref_sub else None
                if ref_sub:
                    qname = ref_sub.qualifiedName()
                    if isinstance(qname, list):
                        qname = qname[0] if qname else None
                    if qname:
                        ref_name = qname.getText()
        except AttributeError:
            pass

        if ref_name:
            self.references.append(
                Reference(
                    name=ref_name,
                    kind="satisfy",
                    line=line,
                    column=col,
                    context=SymbolKind.REQUIREMENT_USAGE,
                )
            )

        # Extract the "by <subject>" part
        try:
            sat_subject = ctx.satisfactionSubjectMember()
            if sat_subject:
                sat_param = sat_subject.satisfactionParameter()
                if sat_param:
                    sat_feat = sat_param.satisfactionFeatureValue()
                    if sat_feat:
                        sat_expr = sat_feat.satisfactionReferenceExpression()
                        if sat_expr:
                            # The expression text contains the subject reference
                            expr_text = sat_expr.getText()
                            if expr_text:
                                self.references.append(
                                    Reference(
                                        name=expr_text,
                                        kind="subject",
                                        line=line,
                                        column=col,
                                        context=SymbolKind.REQUIREMENT_USAGE,
                                    )
                                )
        except AttributeError:
            pass

    # -- Subject member listener (inside requirement definitions) ------------

    def enterSubjectUsage(self, ctx: SysMLv2Parser.SubjectUsageContext):
        """Handle `subject <name> : <Type>` inside requirement definitions."""
        name = self._extract_usage_name(ctx)
        if name:
            symbol = Symbol(
                name=name,
                kind=SymbolKind.PART_USAGE,
                line=self._get_line(ctx),
                column=self._get_column(ctx),
            )
            err = self.current_scope.define(symbol)
            if err:
                self.errors.append(err)

            self.references.append(
                Reference(
                    name=name,
                    kind="subject",
                    line=self._get_line(ctx),
                    column=self._get_column(ctx),
                    context=SymbolKind.REQUIREMENT_DEF,
                )
            )

        # Track the type reference
        type_name = self._extract_usage_type_ref(ctx)
        if type_name:
            self.references.append(
                Reference(
                    name=type_name,
                    kind="type_ref",
                    line=self._get_line(ctx),
                    column=self._get_column(ctx),
                )
            )

    # -- Feature specialization listener (typing, subsetting, redefinition) -

    def enterFeatureSpecialization(
        self, ctx: SysMLv2Parser.FeatureSpecializationContext
    ):
        """Handle feature specializations (typing, subsetting, redefinitions)."""
        # Typings: `: <type>` — path: typings → typedBy → featureTyping → ownedFeatureTyping → qualifiedName
        typings = ctx.typings()
        if typings:
            typed_by = typings.typedBy()
            if typed_by:
                ft = typed_by.featureTyping()
                if ft:
                    owned_ft = ft.ownedFeatureTyping()
                    if owned_ft:
                        qname = owned_ft.qualifiedName()
                        if isinstance(qname, list):
                            qname = qname[0] if qname else None
                        if qname:
                            self.references.append(
                                Reference(
                                    name=qname.getText(),
                                    kind="type_ref",
                                    line=self._get_line(ctx),
                                    column=self._get_column(ctx),
                                )
                            )

        # Subsettings: `subsets <name>` — path: subsettings → subsets → ownedSubsetting → qualifiedName
        subsettings = ctx.subsettings()
        if subsettings:
            subsets = subsettings.subsets()
            if subsets:
                owned_sub = subsets.ownedSubsetting()
                if owned_sub:
                    qname = owned_sub.qualifiedName()
                    if isinstance(qname, list):
                        qname = qname[0] if qname else None
                    if qname:
                        self.references.append(
                            Reference(
                                name=qname.getText(),
                                kind="subsetting",
                                line=self._get_line(ctx),
                                column=self._get_column(ctx),
                            )
                        )

    # -- Subclassification listener -----------------------------------------

    def enterSubclassification(self, ctx: SysMLv2Parser.SubclassificationContext):
        """Handle `specializes <name>` or `:> <name>` as a standalone relationship.

        New grammar: SubclassificationContext has qualifiedName() directly
        (not ownedSubclassification). Path:
        Subclassification → qualifiedName
        """
        qname = ctx.qualifiedName()
        if isinstance(qname, list):
            qname = qname[0] if qname else None
        if qname:
            self.references.append(
                Reference(
                    name=qname.getText(),
                    kind="specialization",
                    line=self._get_line(ctx),
                    column=self._get_column(ctx),
                )
            )

    def enterSubclassificationPart(
        self, ctx: SysMLv2Parser.SubclassificationPartContext
    ):
        """Handle `specializes <name>` or `:> <name>` in definition declarations.

        In the new grammar, `:> Base` inside a definition declaration is parsed
        as SubclassificationPart (not Subclassification). Path:
        SubclassificationPart → ownedSubclassification (list) → qualifiedName
        """
        owned_list = ctx.ownedSubclassification()
        if isinstance(owned_list, list):
            for owned in owned_list:
                qname = owned.qualifiedName()
                if isinstance(qname, list):
                    qname = qname[0] if qname else None
                if qname:
                    self.references.append(
                        Reference(
                            name=qname.getText(),
                            kind="specialization",
                            line=self._get_line(ctx),
                            column=self._get_column(ctx),
                        )
                    )
        elif owned_list:
            qname = owned_list.qualifiedName()
            if isinstance(qname, list):
                qname = qname[0] if qname else None
            if qname:
                self.references.append(
                    Reference(
                        name=qname.getText(),
                        kind="specialization",
                        line=self._get_line(ctx),
                        column=self._get_column(ctx),
                    )
                )

    # -- Perform action usage listener --------------------------------------

    def enterPerformActionUsage(self, ctx: SysMLv2Parser.PerformActionUsageContext):
        """Handle `perform <actionRef>` — track the reference to the action definition."""
        line = self._get_line(ctx)
        col = self._get_column(ctx)

        try:
            decl = ctx.performActionUsageDeclaration()
            if decl:
                # Alternative 1: OwnedReferenceSubsetting
                ref_sub = decl.ownedReferenceSubsetting()
                if ref_sub:
                    # ownedReferenceSubsetting may return a list
                    if isinstance(ref_sub, list):
                        ref_sub = ref_sub[0] if ref_sub else None
                    if ref_sub:
                        qname = ref_sub.qualifiedName()
                        if isinstance(qname, list):
                            qname = qname[0] if qname else None
                        if qname:
                            self.references.append(
                                Reference(
                                    name=qname.getText(),
                                    kind="perform_action",
                                    line=line,
                                    column=col,
                                    context=SymbolKind.ACTION_USAGE,
                                )
                            )
                            return

                # Alternative 2: usageDeclaration with name
                usage_decl = decl.usageDeclaration()
                if usage_decl:
                    ident = usage_decl.identification()
                    if ident:
                        self.references.append(
                            Reference(
                                name=self._extract_name(ident),
                                kind="perform_action",
                                line=line,
                                column=col,
                                context=SymbolKind.ACTION_USAGE,
                            )
                        )
        except AttributeError:
            pass

    def enterStatePerformActionUsage(
        self, ctx: SysMLv2Parser.StatePerformActionUsageContext
    ):
        """Handle `do perform <actionRef>` inside state definitions.

        New grammar: statePerformActionUsage wraps performActionUsageDeclaration.
        This is the `do perform <actionRef>` pattern inside state definitions.
        """
        line = self._get_line(ctx)
        col = self._get_column(ctx)

        try:
            decl = ctx.performActionUsageDeclaration()
            if decl:
                # Alternative 1: OwnedReferenceSubsetting
                ref_sub = decl.ownedReferenceSubsetting()
                if ref_sub:
                    # ownedReferenceSubsetting may return a list
                    if isinstance(ref_sub, list):
                        ref_sub = ref_sub[0] if ref_sub else None
                    if ref_sub:
                        qname = ref_sub.qualifiedName()
                        # qualifiedName may return a list
                        if isinstance(qname, list):
                            qname = qname[0] if qname else None
                        if qname:
                            self.references.append(
                                Reference(
                                    name=qname.getText(),
                                    kind="perform_action",
                                    line=line,
                                    column=col,
                                    context=SymbolKind.ACTION_USAGE,
                                )
                            )
                            return

                # Alternative 2: usageDeclaration with name
                usage_decl = decl.usageDeclaration()
                if usage_decl:
                    ident = usage_decl.identification()
                    if ident:
                        self.references.append(
                            Reference(
                                name=self._extract_name(ident),
                                kind="perform_action",
                                line=line,
                                column=col,
                                context=SymbolKind.ACTION_USAGE,
                            )
                        )
        except AttributeError:
            pass

    # -- Flow end listener --------------------------------------------------

    def enterFlowEnd(self, ctx: SysMLv2Parser.FlowEndContext):
        """Handle flow end references in `flow ... from <part>.<feature> to <part>.<feature>`.

        New grammar: flowEnd -> qualifiedName (DOT qualifiedName)*
        This directly gives us the feature chain as qualifiedName segments.
        """
        line = self._get_line(ctx)
        col = self._get_column(ctx)

        # Collect all qualifiedName segments
        qnames = ctx.qualifiedName()
        if not qnames:
            return

        # qualifiedName() returns a list in the new grammar
        if isinstance(qnames, list):
            qname_list = qnames
        else:
            qname_list = [qnames]

        if not qname_list:
            return

        # If there's only one qualifiedName, it's a simple reference
        if len(qname_list) == 1:
            name = qname_list[0].getText()
            if name:
                self.references.append(
                    Reference(
                        name=name,
                        kind="feature_ref",
                        line=line,
                        column=col,
                        context=SymbolKind.PART_USAGE,
                    )
                )
            return

        # Multiple qualifiedNames separated by DOT — feature chain
        segments = [qn.getText() for qn in qname_list]
        chain_text = ".".join(segments)
        first_segment = segments[0]

        self.references.append(
            Reference(
                name=chain_text,
                kind="flow_end",
                line=line,
                column=col,
                context=SymbolKind.PART_USAGE,
                parent_ref_name=first_segment,
            )
        )

    # -- Connector end listener ---------------------------------------------

    def enterConnectorEnd(self, ctx: SysMLv2Parser.ConnectorEndContext):
        """Handle connector end references in `connect <part>.<port> to <part>.<port>`.

        New grammar: connectorEnd -> ownedReferenceSubsetting -> qualifiedName
        """
        line = self._get_line(ctx)
        col = self._get_column(ctx)

        try:
            ref_sub = ctx.ownedReferenceSubsetting()
            if ref_sub:
                # ownedReferenceSubsetting may return a list
                if isinstance(ref_sub, list):
                    ref_sub = ref_sub[0] if ref_sub else None
                if ref_sub:
                    qname = ref_sub.qualifiedName()
                    if isinstance(qname, list):
                        qname = qname[0] if qname else None
                    if qname:
                        name = qname.getText()
                        if name:
                            self.references.append(
                                Reference(
                                    name=name,
                                    kind="connector_end",
                                    line=line,
                                    column=col,
                                    context=SymbolKind.PORT_USAGE,
                                )
                            )
        except AttributeError:
            pass

    # -- Default reference usage listener (in/out params in action defs) ----

    def enterDefaultReferenceUsage(
        self, ctx: SysMLv2Parser.DefaultReferenceUsageContext
    ):
        """Handle default reference usages — these are `in`/`out` parameters
        inside action definitions (e.g., `in sensorData : BrakePedalData`)
        and `:>>` redefinitions inside action usage bodies
        (e.g., `in :>> sensorData = pedalData`).

        New grammar: defaultReferenceUsage -> refPrefix usage valuePart? usageBody
        refPrefix may contain a featureDirection (in/out/inout).
        usage -> usageDeclaration -> identification (for named params)
        OR usage -> usageDeclaration -> featureSpecializationPart (for :>> redefs)
        """
        name = self._extract_default_ref_name(ctx)
        if not name:
            return

        is_redefinition = self._is_redefinition(ctx)

        if is_redefinition:
            # :>> redefinition — the name references an existing parameter
            self.references.append(
                Reference(
                    name=name,
                    kind="feature_ref",
                    line=self._get_line(ctx),
                    column=self._get_column(ctx),
                    context=SymbolKind.ATTRIBUTE_USAGE,
                )
            )
        else:
            # Regular parameter declaration (e.g., `in sensorData : BrakePedalData`)
            symbol = Symbol(
                name=name,
                kind=SymbolKind.ATTRIBUTE_USAGE,
                line=self._get_line(ctx),
                column=self._get_column(ctx),
            )
            err = self.current_scope.define(symbol)
            if err:
                self.errors.append(err)

            # Track the type reference
            type_name = self._extract_default_ref_type(ctx)
            if type_name:
                self.references.append(
                    Reference(
                        name=type_name,
                        kind="type_ref",
                        line=self._get_line(ctx),
                        column=self._get_column(ctx),
                    )
                )

    def _is_redefinition(self, ctx) -> bool:
        """Check if a DefaultReferenceUsage is a :>> redefinition.

        A redefinition has no identification in the usageDeclaration —
        only a featureSpecializationPart with redefinitions.
        """
        try:
            usage = ctx.usage()
            if usage:
                decl = usage.usageDeclaration()
                if decl:
                    # If there's an identification, it's NOT a redefinition
                    ident = decl.identification()
                    if ident:
                        return False
                    # Check for redefinitions in the specialization part
                    spec_part = decl.featureSpecializationPart()
                    if spec_part:
                        for spec in spec_part.featureSpecialization():
                            redefs = spec.redefinitions()
                            if redefs:
                                return True
        except AttributeError:
            pass
        return False

    def _extract_default_ref_name(self, ctx) -> str:
        """Extract the name from a DefaultReferenceUsage context.

        For named params: usage -> usageDeclaration -> identification -> name
        For :>> redefs: usage -> usageDeclaration -> featureSpecializationPart
          -> featureSpecialization -> redefinitions -> redefines
          -> ownedRedefinition -> qualifiedName
        """
        try:
            usage = ctx.usage()
            if usage:
                decl = usage.usageDeclaration()
                if decl:
                    # Named parameter
                    ident = decl.identification()
                    if ident:
                        return self._extract_name(ident)

                    # :>> redefinition — name is in the redefinition target
                    spec_part = decl.featureSpecializationPart()
                    if spec_part:
                        for spec in spec_part.featureSpecialization():
                            redefs = spec.redefinitions()
                            if redefs:
                                for redef in redefs.redefines():
                                    owned_redef = redef.ownedRedefinition()
                                    if owned_redef:
                                        qname = owned_redef.qualifiedName()
                                        if isinstance(qname, list):
                                            qname = qname[0] if qname else None
                                        if qname:
                                            return qname.getText()
        except AttributeError:
            pass
        return ""

    def _extract_default_ref_type(self, ctx) -> str:
        """Extract the type reference from a DefaultReferenceUsage context."""
        try:
            usage = ctx.usage()
            if usage:
                decl = usage.usageDeclaration()
                if decl:
                    spec_part = decl.featureSpecializationPart()
                    if spec_part:
                        return self._extract_type_from_spec_part(spec_part)
        except AttributeError:
            pass
        return ""

    # -- Owned expression listener (feature chains in new grammar) ----------

    def enterOwnedExpression(self, ctx: SysMLv2Parser.OwnedExpressionContext):
        """Handle owned expressions — detect feature chains.

        New grammar: ownedExpression -> baseExpression (DOT qualifiedName)*
        When there are DOT-qualifiedName segments, this is a feature chain
        like `partName.featureName`. We set a flag to suppress individual
        feature reference creation and handle the chain on exit.
        """
        # Check if this ownedExpression has DOT-qualifiedName chaining
        qnames = ctx.qualifiedName()
        if qnames:
            # qualifiedName() returns a list in the new grammar
            if isinstance(qnames, list) and len(qnames) > 0:
                self._in_feature_chain_expr = True
            elif not isinstance(qnames, list):
                # Single qualifiedName — still a chain if there's a DOT
                self._in_feature_chain_expr = True

    def exitOwnedExpression(self, ctx: SysMLv2Parser.OwnedExpressionContext):
        """Handle exit of owned expression — create feature chain reference if needed."""
        was_in_chain = self._in_feature_chain_expr
        self._in_feature_chain_expr = False

        if not was_in_chain:
            return

        # Build the full chain text from the expression
        chain_text = ctx.getText()
        if chain_text and "." in chain_text:
            first_segment = chain_text.split(".")[0]
            self.references.append(
                Reference(
                    name=chain_text,
                    kind="feature_chain",
                    line=self._get_line(ctx),
                    column=self._get_column(ctx),
                    context=SymbolKind.ATTRIBUTE_USAGE,
                    parent_ref_name=first_segment,
                )
            )

    # -- Feature reference expression listener -------------------------------

    def enterFeatureReferenceExpression(
        self, ctx: SysMLv2Parser.FeatureReferenceExpressionContext
    ):
        """Handle feature reference expressions in action bodies.

        New grammar: featureReferenceExpression -> qualifiedName
        These are bare name references like `sensorData` or `actuationSignal`
        used in expressions within action bodies.

        If _in_feature_chain_expr is True, this is part of a feature chain
        and will be handled by exitOwnedExpression — skip it.
        """
        if self._in_feature_chain_expr:
            return

        line = self._get_line(ctx)
        col = self._get_column(ctx)

        try:
            qname = ctx.qualifiedName()
            if qname:
                # qualifiedName may return a list
                if isinstance(qname, list):
                    qname = qname[0] if qname else None
                if qname:
                    name = qname.getText()
                    if name and "::" not in name:
                        self.references.append(
                            Reference(
                                name=name,
                                kind="feature_ref",
                                line=line,
                                column=col,
                                context=SymbolKind.ATTRIBUTE_USAGE,
                            )
                        )
        except AttributeError:
            pass

    # -- Post-walk semantic checks -------------------------------------------

    def _resolve_feature_chain(
        self, chain: str
    ) -> tuple[Optional[Symbol], Optional[str]]:
        """Resolve a feature chain like 'partName.featureName.subFeature'.

        Type-aware: handles both definitions (own child scopes) and usages
        (inherit scope from their type definition).
        """
        segments = chain.split(".")
        if not segments:
            return None, None

        first = segments[0]
        symbol = self.root_scope.lookup_global(first)
        if symbol is None:
            return None, first

        for segment in segments[1:]:
            child_scope = self._get_child_scope_for_symbol_typed(symbol)
            if child_scope is None:
                return symbol, segment
            if segment not in child_scope.symbols:
                return symbol, segment
            symbol = child_scope.symbols[segment]

        return symbol, None

    def _get_child_scope_for_symbol_typed(self, symbol: Symbol) -> Optional[Scope]:
        """Find the child scope for a symbol, considering type definitions.

        For definitions (part def, action def, etc.), the child scope has the
        same name as the symbol.

        For usages (part, port, etc.) that are typed by a definition, the child
        scope is the scope of the type definition.
        """
        if symbol.scope is None:
            return None

        # First, try to find a child scope with the same name as the symbol
        for child in symbol.scope.children:
            if child.name == symbol.name:
                return child

        # For usages typed by a definition, find the type's scope
        if symbol.kind in (
            SymbolKind.PART_USAGE,
            SymbolKind.PORT_USAGE,
            SymbolKind.ITEM_USAGE,
            SymbolKind.ACTION_USAGE,
            SymbolKind.CONNECTION_USAGE,
            SymbolKind.ATTRIBUTE_USAGE,
        ):
            type_name = self._find_type_for_usage(symbol)
            if type_name:
                type_symbol = self.root_scope.lookup_global(type_name)
                if type_symbol and type_symbol.scope:
                    for child in type_symbol.scope.children:
                        if child.name == type_name:
                            return child

        return None

    def _find_type_for_usage(self, symbol: Symbol) -> Optional[str]:
        """Find the type name for a usage symbol by checking collected references."""
        for ref in self.references:
            if (
                ref.kind == "type_ref"
                and ref.line == symbol.line
                and ref.name != symbol.name
            ):
                return ref.name
        return None

    def validate(self) -> list[SemanticError]:
        """Run all semantic checks after the tree walk."""
        for ref in self.references:
            self._check_reference(ref)
        return self.errors

    def _check_reference(self, ref: Reference) -> None:
        """Check a single reference for semantic validity."""
        if ref.kind == "satisfy":
            self._check_satisfy_reference(ref)
        elif ref.kind == "subject":
            self._check_subject_reference(ref)
        elif ref.kind in ("type_ref", "specialization", "subsetting"):
            self._check_type_reference(ref)
        elif ref.kind == "perform_action":
            self._check_perform_action_reference(ref)
        elif ref.kind == "flow_end":
            self._check_flow_end_reference(ref)
        elif ref.kind == "connector_end":
            self._check_connector_end_reference(ref)
        elif ref.kind == "feature_chain":
            self._check_feature_chain_reference(ref)
        elif ref.kind == "feature_ref":
            self._check_feature_ref_reference(ref)

    def _check_satisfy_reference(self, ref: Reference) -> None:
        """Check that `satisfy <name>` references a requirement definition or usage."""
        symbol = self.root_scope.lookup_global(ref.name)
        if symbol is None:
            simple_name = ref.name.split("::")[-1] if "::" in ref.name else ref.name
            symbol = self.root_scope.lookup_global(simple_name)

        if symbol is None:
            self.errors.append(
                SemanticError(
                    message=f"Couldn't resolve reference to Feature '{ref.name}'.",
                    line=ref.line,
                    column=ref.column,
                    severity="error",
                    error_type="reference",
                )
            )
        elif symbol.kind not in _REQUIREMENT_KINDS:
            self.errors.append(
                SemanticError(
                    message=(
                        f"Must reference a requirement. "
                        f"'{ref.name}' is a {symbol.kind.value}, not a requirement definition or usage."
                    ),
                    line=ref.line,
                    column=ref.column,
                    severity="error",
                    error_type="type",
                )
            )

    def _check_subject_reference(self, ref: Reference) -> None:
        """Check that `subject <name>` references a valid part/element."""
        symbol = self.root_scope.lookup_global(ref.name)
        if symbol is None:
            simple_name = ref.name.split("::")[-1] if "::" in ref.name else ref.name
            symbol = self.root_scope.lookup_global(simple_name)

        if symbol is None:
            self.errors.append(
                SemanticError(
                    message=f"Couldn't resolve reference to Element '{ref.name}'.",
                    line=ref.line,
                    column=ref.column,
                    severity="error",
                    error_type="reference",
                )
            )
        elif symbol.kind not in _SUBJECT_KINDS:
            self.errors.append(
                SemanticError(
                    message=(
                        f"Subject '{ref.name}' is a {symbol.kind.value}. "
                        f"Subjects typically reference part or item definitions."
                    ),
                    line=ref.line,
                    column=ref.column,
                    severity="warning",
                    error_type="type",
                )
            )

    def _check_type_reference(self, ref: Reference) -> None:
        """Check that type references (specialization, typing) resolve."""
        symbol = self.root_scope.lookup_global(ref.name)
        if symbol is None:
            simple_name = ref.name.split("::")[-1] if "::" in ref.name else ref.name
            symbol = self.root_scope.lookup_global(simple_name)

        if symbol is None:
            self.errors.append(
                SemanticError(
                    message=f"Couldn't resolve reference to Feature '{ref.name}'.",
                    line=ref.line,
                    column=ref.column,
                    severity="warning",
                    error_type="reference",
                )
            )

    def _check_perform_action_reference(self, ref: Reference) -> None:
        """Check that `perform <name>` references an action definition or usage."""
        name = ref.name

        if "." in name:
            resolved, failed_segment = self._resolve_feature_chain(name)
            if failed_segment:
                self.errors.append(
                    SemanticError(
                        message=f"Couldn't resolve reference to Feature '{failed_segment}'.",
                        line=ref.line,
                        column=ref.column,
                        severity="error",
                        error_type="reference",
                    )
                )
            return

        symbol = self.root_scope.lookup_global(name)
        if symbol is None:
            simple_name = name.split("::")[-1] if "::" in name else name
            symbol = self.root_scope.lookup_global(simple_name)

        if symbol is None:
            self.errors.append(
                SemanticError(
                    message=f"Couldn't resolve reference to Feature '{name}'.",
                    line=ref.line,
                    column=ref.column,
                    severity="error",
                    error_type="reference",
                )
            )
        elif symbol.kind not in _ACTION_KINDS and symbol.kind != SymbolKind.BUILTIN:
            self.errors.append(
                SemanticError(
                    message=(
                        f"Perform references '{name}' which is a {symbol.kind.value}, "
                        f"not an action definition or usage."
                    ),
                    line=ref.line,
                    column=ref.column,
                    severity="warning",
                    error_type="type",
                )
            )

    def _check_flow_end_reference(self, ref: Reference) -> None:
        """Check that flow end references resolve to valid features."""
        name = ref.name

        if "." in name:
            resolved, failed_segment = self._resolve_feature_chain(name)
            if failed_segment:
                if resolved is None:
                    self.errors.append(
                        SemanticError(
                            message=f"Couldn't resolve reference to Element '{failed_segment}'.",
                            line=ref.line,
                            column=ref.column,
                            severity="error",
                            error_type="reference",
                        )
                    )
                else:
                    self.errors.append(
                        SemanticError(
                            message=f"Couldn't resolve reference to Element '{failed_segment}'.",
                            line=ref.line,
                            column=ref.column,
                            severity="error",
                            error_type="reference",
                        )
                    )
            return

        symbol = self.root_scope.lookup_global(name)
        if symbol is None:
            simple_name = name.split("::")[-1] if "::" in name else name
            symbol = self.root_scope.lookup_global(simple_name)

        if symbol is None:
            self.errors.append(
                SemanticError(
                    message=f"Couldn't resolve reference to Feature '{name}'.",
                    line=ref.line,
                    column=ref.column,
                    severity="error",
                    error_type="reference",
                )
            )

    def _check_connector_end_reference(self, ref: Reference) -> None:
        """Check that connector end references resolve to valid port/part features."""
        name = ref.name

        if "." in name:
            resolved, failed_segment = self._resolve_feature_chain(name)
            if failed_segment:
                if resolved is None:
                    self.errors.append(
                        SemanticError(
                            message=f"Couldn't resolve reference to Element '{failed_segment}'.",
                            line=ref.line,
                            column=ref.column,
                            severity="error",
                            error_type="reference",
                        )
                    )
                else:
                    self.errors.append(
                        SemanticError(
                            message="Must be a valid feature",
                            line=ref.line,
                            column=ref.column,
                            severity="error",
                            error_type="reference",
                        )
                    )
            return

        symbol = self.root_scope.lookup_global(name)
        if symbol is None:
            simple_name = name.split("::")[-1] if "::" in name else name
            symbol = self.root_scope.lookup_global(simple_name)

        if symbol is None:
            self.errors.append(
                SemanticError(
                    message=f"Couldn't resolve reference to Feature '{name}'.",
                    line=ref.line,
                    column=ref.column,
                    severity="error",
                    error_type="reference",
                )
            )

    def _check_feature_chain_reference(self, ref: Reference) -> None:
        """Check that feature chain references resolve through child scopes."""
        name = ref.name

        resolved, failed_segment = self._resolve_feature_chain(name)
        if failed_segment:
            if resolved is None:
                self.errors.append(
                    SemanticError(
                        message=f"Couldn't resolve reference to Element '{failed_segment}'.",
                        line=ref.line,
                        column=ref.column,
                        severity="error",
                        error_type="reference",
                    )
                )
            else:
                self.errors.append(
                    SemanticError(
                        message="Must be a valid feature",
                        line=ref.line,
                        column=ref.column,
                        severity="error",
                        error_type="reference",
                    )
                )

    def _check_feature_ref_reference(self, ref: Reference) -> None:
        """Check that bare feature references resolve to defined symbols."""
        name = ref.name

        symbol = self.root_scope.lookup_global(name)
        if symbol is None:
            simple_name = name.split("::")[-1] if "::" in name else name
            symbol = self.root_scope.lookup_global(simple_name)

        if symbol is None:
            self.errors.append(
                SemanticError(
                    message=f"Couldn't resolve reference to Feature '{name}'.",
                    line=ref.line,
                    column=ref.column,
                    severity="error",
                    error_type="reference",
                )
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def validate_semantics(
    code: str, syntax_errors: list[str] | None = None
) -> list[SemanticError]:
    """Run semantic validation on SysML v2 code.

    Parameters
    ----------
    code : str
        The SysML v2 source code to validate.
    syntax_errors : list[str] | None
        Existing syntax errors. If non-empty, semantic validation is skipped
        because the parse tree may be incomplete.

    Returns
    -------
    list[SemanticError]
        List of semantic errors found.
    """
    # Skip semantic validation if there are syntax errors
    if syntax_errors:
        logger.info(
            "Skipping semantic validation due to %d syntax errors", len(syntax_errors)
        )
        return []

    from antlr4 import CommonTokenStream, InputStream

    from .NewGrammar.SysMLv2Lexer import SysMLv2Lexer
    from .NewGrammar.SysMLv2Parser import SysMLv2Parser as AntlrParser

    try:
        input_stream = InputStream(code)
        lexer = SysMLv2Lexer(input_stream)
        token_stream = CommonTokenStream(lexer)
        parser = AntlrParser(token_stream)
        tree = parser.rootNamespace()

        # Walk the tree to build symbol table and collect references
        validator = SemanticValidator()
        walker = ParseTreeWalker()
        walker.walk(validator, tree)

        # Run semantic checks
        errors = validator.validate()
        return errors

    except Exception as exc:
        logger.warning("Semantic validation failed: %s", exc)
        return [
            SemanticError(
                message=f"Semantic validation error: {exc}",
                severity="warning",
                error_type="semantic",
            )
        ]
