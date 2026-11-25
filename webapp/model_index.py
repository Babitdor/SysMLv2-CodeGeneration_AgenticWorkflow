"""Queryable index of a SysML v2 model, built from the real parse tree.

SemanticValidator already walks the ANTLR tree into a symbol table and a list of
references, but it keeps no element identity or nesting: its usage handlers
define a symbol and never push a scope, so a usage body like
"part vehicle { part w1; }" flattens into the package. This module subclasses
that walker read-only and adds what the UI needs - ids, parents, line ranges and
typed edges.

Ids are qualified names (Vehicle::w1), so they stay stable while the rest of the
model is edited. Analysis lives in webapp/model_analysis.py and takes an index,
so it is testable without ANTLR.
"""

from __future__ import annotations

import hashlib
import logging
import re
from functools import lru_cache
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field

from tools.sysml_parser.SemanticValidator import Reference, SemanticValidator, Symbol

logger = logging.getLogger(__name__)

MAX_ELEMENTS = 4000
MAX_EDGES = 12000
MAX_ISSUES = 200
CACHE_ENTRIES = 8

REQUIREMENT_KINDS = frozenset({"requirement_def", "requirement_usage"})


# ---------------------------------------------------------------------------
# Wire shapes
# ---------------------------------------------------------------------------


class ModelElement(BaseModel):
    """One named element. id is the qualified name, so it is stable."""

    id: str
    name: str
    kind: str
    qualified_name: str
    parent: Optional[str] = None
    line: int = 0
    column: int = 0
    end_line: int = 0


class ModelEdge(BaseModel):
    """A reference from one element to another, of a known kind.

    resolved is false when the name resolved to no symbol at all (a typo, or a
    type the model does not define). A target that is a builtin carries the
    builtin name rather than an element id.
    """

    source: str
    target: str
    kind: str
    label: str
    line: int = 0
    resolved: bool = True


class ModelIssue(BaseModel):
    severity: Literal["error", "warning", "info"] = "warning"
    message: str
    line: int = 0
    column: int = 0
    rule: str = "semantic"
    element: Optional[str] = None


class ModelStats(BaseModel):
    elements: int = 0
    edges: int = 0
    issues: int = 0
    unresolved: int = 0
    requirements: int = 0


class ModelIndex(BaseModel):
    """One model, indexed. Cached, so treat an instance as read-only."""

    revision: str
    validator: str = "antlr"
    elements: List[ModelElement] = Field(default_factory=list)
    edges: List[ModelEdge] = Field(default_factory=list)
    issues: List[ModelIssue] = Field(default_factory=list)
    stats: ModelStats = Field(default_factory=ModelStats)
    truncated: bool = False


# ---------------------------------------------------------------------------
# Which generated rules are elements
# ---------------------------------------------------------------------------

# Rule context class name -> element kind. Only contexts that name a real
# element: modifier contexts (subject, perform, satisfy, transition, ...) are
# deliberately absent so their references stay attributed to the element that
# owns them, which is what impact analysis needs.
_ELEMENT_RULES: Dict[str, str] = {
    "PackageContext": "package",
    "PartDefinitionContext": "part_def",
    "PortDefinitionContext": "port_def",
    "AttributeDefinitionContext": "attribute_def",
    "ItemDefinitionContext": "item_def",
    "ConnectionDefinitionContext": "connection_def",
    "InterfaceDefinitionContext": "interface_def",
    "AllocationDefinitionContext": "allocation_def",
    "FlowDefinitionContext": "flow_def",
    "ActionDefinitionContext": "action_def",
    "StateDefinitionContext": "state_def",
    "CalculationDefinitionContext": "calculation_def",
    "ConstraintDefinitionContext": "constraint_def",
    "RequirementDefinitionContext": "requirement_def",
    "ConcernDefinitionContext": "concern_def",
    "CaseDefinitionContext": "case_def",
    "AnalysisCaseDefinitionContext": "case_def",
    "VerificationCaseDefinitionContext": "case_def",
    "UseCaseDefinitionContext": "case_def",
    "ViewDefinitionContext": "view_def",
    "ViewpointDefinitionContext": "viewpoint_def",
    "RenderingDefinitionContext": "rendering_def",
    "MetadataDefinitionContext": "metadata_def",
    "EnumerationDefinitionContext": "enumeration_def",
    "OccurrenceDefinitionContext": "occurrence_def",
    "IndividualDefinitionContext": "occurrence_def",
    "PartUsageContext": "part_usage",
    "PortUsageContext": "port_usage",
    "AttributeUsageContext": "attribute_usage",
    "ItemUsageContext": "item_usage",
    "ConnectionUsageContext": "connection_usage",
    "BindingConnectorAsUsageContext": "connection_usage",
    "SuccessionAsUsageContext": "connection_usage",
    "InterfaceUsageContext": "interface_usage",
    "AllocationUsageContext": "allocation_usage",
    "FlowUsageContext": "flow_usage",
    "SuccessionFlowUsageContext": "flow_usage",
    "ActionUsageContext": "action_usage",
    "StateUsageContext": "state_usage",
    "ExhibitStateUsageContext": "state_usage",
    "CalculationUsageContext": "calculation_usage",
    "ConstraintUsageContext": "constraint_usage",
    "RequirementUsageContext": "requirement_usage",
    "ConcernUsageContext": "concern_usage",
    "CaseUsageContext": "case_usage",
    "AnalysisCaseUsageContext": "case_usage",
    "VerificationCaseUsageContext": "case_usage",
    "UseCaseUsageContext": "case_usage",
    "ViewUsageContext": "view_usage",
    "ViewpointUsageContext": "viewpoint_usage",
    "RenderingUsageContext": "rendering_usage",
    "EnumerationUsageContext": "enumeration_usage",
    "OccurrenceUsageContext": "occurrence_usage",
    "IndividualUsageContext": "occurrence_usage",
    "EventOccurrenceUsageContext": "occurrence_usage",
    "PortionUsageContext": "occurrence_usage",
    "ReferenceUsageContext": "reference_usage",
    "DefaultReferenceUsageContext": "reference_usage",
    "EndFeatureUsageContext": "reference_usage",
}

# Every other generated rule. tests/test_model_index.py asserts that
# _ELEMENT_RULES plus this set cover the listener exactly, so a regenerated
# grammar fails loudly instead of silently dropping elements from the index.
_IGNORED_RULES: frozenset = frozenset({
    "AcceptNodeContext", "AcceptNodeDeclarationContext", "AcceptParameterPartContext", "ActionBehaviorMemberContext",
    "ActionBodyContext", "ActionBodyItemContext", "ActionBodyParameterContext", "ActionBodyParameterMemberContext",
    "ActionNodeContext", "ActionNodeMemberContext", "ActionNodePrefixContext", "ActionNodeUsageDeclarationContext",
    "ActionTargetSuccessionContext", "ActionTargetSuccessionMemberContext", "ActionUsageDeclarationContext", "ActorMemberContext",
    "ActorUsageContext", "AliasMemberContext", "AllocationUsageDeclarationContext", "AnnotatingElementContext",
    "AnnotatingMemberContext", "AnnotationContext", "ArgumentExpressionMemberContext", "ArgumentListContext",
    "ArgumentMemberContext", "AssertConstraintUsageContext", "AssignmentNodeContext", "AssignmentNodeDeclarationContext",
    "AssignmentTargetBindingContext", "AssignmentTargetMemberContext", "AssignmentTargetParameterContext", "AssociationContext",
    "AssociationStructureContext", "BaseExpressionContext", "BasicDefinitionPrefixContext", "BasicFeaturePrefixContext",
    "BasicUsagePrefixContext", "BehaviorContext", "BehaviorUsageElementContext", "BehaviorUsageMemberContext",
    "BinaryConnectorDeclarationContext", "BinaryConnectorPartContext", "BinaryInterfacePartContext", "BindingConnectorContext",
    "BindingConnectorDeclarationContext", "BodyExpressionContext", "BooleanExpressionContext", "CalculationBodyContext",
    "CalculationBodyItemContext", "CalculationBodyPartContext", "CalculationUsageDeclarationContext", "CaseBodyContext",
    "CaseBodyItemContext", "ChainingPartContext", "ClassContext", "ClassifierContext",
    "ClassifierDeclarationContext", "CommentContext", "ConjugatedPortDefinitionContext", "ConjugatedPortDefinitionMemberContext",
    "ConjugatedPortTypingContext", "ConjugationContext", "ConjugationPartContext", "ConnectorContext",
    "ConnectorDeclarationContext", "ConnectorEndContext", "ConnectorEndMemberContext", "ConnectorPartContext",
    "ConstraintUsageDeclarationContext", "ConstructorExpressionContext", "ControlNodeContext", "ControlNodePrefixContext",
    "CrossesContext", "DataTypeContext", "DecisionNodeContext", "DefaultInterfaceEndContext",
    "DefaultTargetSuccessionContext", "DefinitionBodyContext", "DefinitionBodyItemContentContext", "DefinitionBodyItemContext",
    "DefinitionContext", "DefinitionDeclarationContext", "DefinitionElementContext", "DefinitionExtensionKeywordContext",
    "DefinitionMemberContext", "DefinitionPrefixContext", "DependencyContext", "DependencyDeclarationContext",
    "DifferencingContext", "DifferencingPartContext", "DisjoiningContext", "DisjoiningPartContext",
    "DoActionMemberContext", "DocumentationContext", "EffectBehaviorMemberContext", "EffectBehaviorUsageContext",
    "ElementFilterMemberContext", "EmptyActionUsage_Context", "EmptyEndMemberContext", "EmptyFeature_Context",
    "EmptyMultiplicityMemberContext", "EmptyMultiplicity_Context", "EmptyParameterMemberContext", "EmptyUsage_Context",
    "EndFeaturePrefixContext", "EndOccurrenceUsageElementContext", "EndUsagePrefixContext", "EntryActionMemberContext",
    "EntryTransitionMemberContext", "EnumeratedValueContext", "EnumerationBodyContext", "EnumerationUsageMemberContext",
    "ExitActionMemberContext", "ExposeContext", "ExpressionContext", "ExpressionParameterMemberContext",
    "ExtendedDefinitionContext", "ExtendedUsageContext", "FeatureBindingContext", "FeatureChainMemberContext",
    "FeatureContext", "FeatureDeclarationContext", "FeatureDirectionContext", "FeatureElementContext",
    "FeatureIdentificationContext", "FeatureInvertingContext", "FeatureMemberContext", "FeaturePrefixContext",
    "FeatureReferenceExpressionContext", "FeatureRelationshipPartContext", "FeatureSpecializationContext", "FeatureSpecializationPartContext",
    "FeatureTypingContext", "FeatureValueContext", "FilterPackageContext", "FilterPackageImportContext",
    "FilterPackageImportDeclarationContext", "FilterPackageMemberContext", "FlowContext", "FlowDeclarationContext",
    "FlowEndContext", "FlowEndMemberContext", "FlowPayloadFeatureContext", "FlowPayloadFeatureMemberContext",
    "ForLoopNodeContext", "ForVariableDeclarationMemberContext", "ForkNodeContext", "FramedConcernMemberContext",
    "FramedConcernUsageContext", "FunctionBodyContext", "FunctionBodyPartContext", "FunctionContext",
    "GeneralTypeContext", "GuardExpressionMemberContext", "GuardedSuccessionContext", "GuardedSuccessionMemberContext",
    "GuardedTargetSuccessionContext", "IdentificationContext", "IfNodeContext", "IfNodeParameterMemberContext",
    "ImportDeclarationContext", "ImportRuleContext", "IncludeUseCaseUsageContext", "InitialNodeMemberContext",
    "InteractionContext", "InterfaceBodyContext", "InterfaceBodyItemContext", "InterfaceEndContext",
    "InterfaceEndMemberContext", "InterfaceNonOccurrenceUsageElementContext", "InterfaceNonOccurrenceUsageMemberContext", "InterfaceOccurrenceUsageElementContext",
    "InterfaceOccurrenceUsageMemberContext", "InterfacePartContext", "InterfaceUsageDeclarationContext", "IntersectingContext",
    "IntersectingPartContext", "InvariantContext", "InvertingPartContext", "JoinNodeContext",
    "LibraryPackageContext", "LiteralBooleanContext", "LiteralExpressionContext", "LiteralInfinityContext",
    "LiteralIntegerContext", "LiteralRealContext", "LiteralStringContext", "MemberElementContext",
    "MemberPrefixContext", "MembershipExposeContext", "MembershipImportContext", "MergeNodeContext",
    "MessageContext", "MessageDeclarationContext", "MessageEventContext", "MessageEventMemberContext",
    "MetaclassContext", "MetadataBodyContext", "MetadataBodyElementContext", "MetadataBodyFeatureContext",
    "MetadataBodyFeatureMemberContext", "MetadataBodyUsageContext", "MetadataBodyUsageMemberContext", "MetadataFeatureContext",
    "MetadataFeatureDeclarationContext", "MultiplicityBoundsContext", "MultiplicityContext", "MultiplicityExpressionMemberContext",
    "MultiplicityPartContext", "MultiplicityRangeContext", "MultiplicitySubsetContext", "NameContext",
    "NamedArgumentContext", "NamedArgumentListContext", "NamespaceBodyContext", "NamespaceBodyElementContext",
    "NamespaceContext", "NamespaceDeclarationContext", "NamespaceExposeContext", "NamespaceFeatureMemberContext",
    "NamespaceImportContext", "NamespaceImportDirectContext", "NamespaceMemberContext", "NaryConnectorDeclarationContext",
    "NaryConnectorPartContext", "NaryInterfacePartContext", "NodeParameterContext", "NodeParameterMemberContext",
    "NonBehaviorBodyItemContext", "NonFeatureChainPrimaryExpressionContext", "NonFeatureElementContext", "NonFeatureMemberContext",
    "NonOccurrenceUsageElementContext", "NonOccurrenceUsageMemberContext", "NullExpressionContext", "ObjectiveMemberContext",
    "ObjectiveRequirementUsageContext", "OccurrenceDefinitionPrefixContext", "OccurrenceUsageElementContext", "OccurrenceUsagePrefixContext",
    "OwnedAnnotationContext", "OwnedConjugationContext", "OwnedCrossFeatureContext", "OwnedCrossFeatureMemberContext",
    "OwnedCrossMultiplicityContext", "OwnedCrossMultiplicityMemberContext", "OwnedCrossSubsettingContext", "OwnedDisjoiningContext",
    "OwnedExpressionContext", "OwnedFeatureInvertingContext", "OwnedFeatureMemberContext", "OwnedFeatureTypingContext",
    "OwnedMultiplicityContext", "OwnedMultiplicityRangeContext", "OwnedRedefinitionContext", "OwnedReferenceSubsettingContext",
    "OwnedRelatedElementContext", "OwnedSpecializationContext", "OwnedSubclassificationContext", "OwnedSubsettingContext",
    "OwnedTypeFeaturingContext", "PackageBodyContext", "PackageBodyElementContext", "PackageDeclarationContext",
    "PackageMemberContext", "PayloadFeatureContext", "PayloadFeatureMemberContext", "PayloadFeatureSpecializationPartContext",
    "PayloadParameterContext", "PayloadParameterMemberContext", "PerformActionUsageContext", "PerformActionUsageDeclarationContext",
    "PortConjugationContext", "PortionKindContext", "PositionalArgumentListContext", "PredicateContext",
    "PrefixMetadataAnnotationContext", "PrefixMetadataFeatureContext", "PrefixMetadataMemberContext", "PrefixMetadataUsageContext",
    "QualifiedNameContext", "RedefinesContext", "RedefinitionContext", "RedefinitionsContext",
    "RefPrefixContext", "ReferencesContext", "RelationshipBodyContext", "RelationshipOwnedElementContext",
    "RequirementBodyContext", "RequirementBodyItemContext", "RequirementConstraintMemberContext", "RequirementConstraintUsageContext",
    "RequirementKindContext", "RequirementVerificationMemberContext", "RequirementVerificationUsageContext", "ResultExpressionMemberContext",
    "ReturnFeatureMemberContext", "ReturnParameterMemberContext", "RootNamespaceContext", "SatisfactionFeatureValueContext",
    "SatisfactionParameterContext", "SatisfactionReferenceExpressionContext", "SatisfactionSubjectMemberContext", "SatisfyRequirementUsageContext",
    "SendNodeContext", "SendNodeDeclarationContext", "SenderReceiverPartContext", "SequenceExpressionListContext",
    "SourceEndContext", "SourceEndMemberContext", "SourceSuccessionContext", "SourceSuccessionMemberContext",
    "SpecializationContext", "SpecializationPartContext", "SpecificTypeContext", "StakeholderMemberContext",
    "StakeholderUsageContext", "StateAcceptActionUsageContext", "StateActionUsageContext", "StateAssignmentActionUsageContext",
    "StateBodyItemContext", "StateDefBodyContext", "StatePerformActionUsageContext", "StateSendActionUsageContext",
    "StateUsageBodyContext", "StepContext", "StructureContext", "StructureUsageElementContext",
    "StructureUsageMemberContext", "SubclassificationContext", "SubclassificationPartContext", "SubjectMemberContext",
    "SubjectUsageContext", "SubsetsContext", "SubsettingContext", "SubsettingsContext",
    "SuccessionContext", "SuccessionDeclarationContext", "SuccessionFlowContext", "SuperclassingPartContext",
    "TargetSuccessionContext", "TargetTransitionUsageContext", "TargetTransitionUsageMemberContext", "TerminateNodeContext",
    "TextualRepresentationContext", "TransitionAcceptActionUsageContext", "TransitionAssignmentActionUsageContext", "TransitionPerformActionUsageContext",
    "TransitionSendActionUsageContext", "TransitionSuccessionContext", "TransitionSuccessionMemberContext", "TransitionUsageContext",
    "TransitionUsageMemberContext", "TriggerActionContext", "TriggerActionMemberContext", "TriggerExpressionContext",
    "TriggerFeatureValueContext", "TriggerValuePartContext", "TypeBodyContext", "TypeBodyElementContext",
    "TypeContext", "TypeDeclarationContext", "TypeFeatureMemberContext", "TypeFeaturingContext",
    "TypeFeaturingPartContext", "TypePrefixContext", "TypeReferenceContext", "TypeRelationshipPartContext",
    "TypedByContext", "TypingsContext", "UnextendedUsagePrefixContext", "UnioningContext",
    "UnioningPartContext", "UnreservedKeywordContext", "UsageBodyContext", "UsageCompletionContext",
    "UsageContext", "UsageDeclarationContext", "UsageElementContext", "UsageExtensionKeywordContext",
    "UsagePrefixContext", "ValuePartContext", "VariantReferenceContext", "VariantUsageElementContext",
    "VariantUsageMemberContext", "ViewBodyContext", "ViewBodyItemContext", "ViewDefinitionBodyContext",
    "ViewDefinitionBodyItemContext", "ViewRenderingMemberContext", "ViewRenderingUsageContext", "VisibilityIndicatorContext",
    "WhileLoopNodeContext",
})


# ---------------------------------------------------------------------------
# The walker
# ---------------------------------------------------------------------------


class _ReferenceLog(list):
    """Reference list that stamps each reference with the element open then.

    The inherited handlers append from inside their enter*/exit* methods, where
    the innermost element is still on the stack, so attribution is exact even
    when several elements share a line.
    """

    __slots__ = ("walker",)

    def __init__(self, walker: "_IndexWalker") -> None:
        super().__init__()
        self.walker = walker

    def append(self, reference: Reference) -> None:
        reference.element = self.walker.current_id()
        super().append(reference)


class _IndexWalker(SemanticValidator):
    """SemanticValidator plus element identity, nesting and line ranges."""

    def __init__(self) -> None:
        super().__init__()
        self.references = _ReferenceLog(self)
        self.elements: List[ModelElement] = []
        self.truncated = False
        self._stack: List[ModelElement] = []
        self._open: List[bool] = []
        self._seen: Dict[str, int] = {}

    def current_id(self) -> Optional[str]:
        return self._stack[-1].id if self._stack else None

    def enterEveryRule(self, ctx) -> None:
        kind = _ELEMENT_RULES.get(type(ctx).__name__)
        pushed = False
        if kind is not None:
            if len(self.elements) >= MAX_ELEMENTS:
                self.truncated = True
            else:
                self.elements.append(self._element(ctx, kind))
                pushed = True
        self._open.append(pushed)

    def exitEveryRule(self, ctx) -> None:
        if not self._open or not self._open.pop():
            return
        element = self._stack.pop()
        # An empty context has no stop token; falling back keeps one odd rule
        # from losing the whole index.
        element.end_line = getattr(ctx.stop, "line", 0) or element.line

    def _element(self, ctx, kind: str) -> ModelElement:
        name = self._name_for(ctx) or f"<{kind}@{self._get_line(ctx)}>"
        parent = self._stack[-1].id if self._stack else None
        qualified = f"{parent}::{name}" if parent else name
        count = self._seen.get(qualified, 0) + 1
        self._seen[qualified] = count
        line = self._get_line(ctx)
        element = ModelElement(
            id=qualified if count == 1 else f"{qualified}#{count}",
            name=name,
            kind=kind,
            qualified_name=qualified,
            parent=parent,
            line=line,
            column=self._get_column(ctx),
            end_line=line,
        )
        self._stack.append(element)
        return element

    def _name_for(self, ctx) -> str:
        """The element name, from the inherited per-shape extractors."""
        for extract in (
            self._extract_package_name,
            self._extract_definition_name,
            self._extract_usage_name,
        ):
            try:
                name = extract(ctx) or ""
            except AttributeError:
                continue
            if name:
                return name.strip("'\"")
        return ""


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------


def _revision(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


# The grammar's ErrorLogger formats each diagnostic as
# "Syntax error at line N, column M: message" followed by a source snippet.
_SYNTAX_MESSAGE = re.compile(r"^Syntax error at line (\d+), column (\d+): ([^\n]*)", re.S)


def _syntax_issues(messages: List[str]) -> List[ModelIssue]:
    """ModelIssues from the grammar's own error log."""
    issues: List[ModelIssue] = []
    for message in messages[:MAX_ISSUES]:
        match = _SYNTAX_MESSAGE.match(message)
        if match is None:
            issues.append(ModelIssue(severity="error", message=message, rule="syntax"))
            continue
        issues.append(
            ModelIssue(
                severity="error",
                message=match.group(3).strip(),
                line=int(match.group(1)),
                column=int(match.group(2)),
                rule="syntax",
            )
        )
    return issues


def _resolve(walker: _IndexWalker, reference: Reference) -> Optional[Symbol]:
    """The symbol a collected reference names, or None when it names nothing."""
    root = walker.root_scope
    symbol = root.lookup_global(reference.name)
    if symbol is None and "::" in reference.name:
        symbol = root.lookup_global(reference.name.split("::")[-1])
    if symbol is None and "." in reference.name:
        symbol = root.lookup_feature_chain(reference.name)
    return symbol


def _element_for_line(elements: List[ModelElement], line: int) -> Optional[str]:
    """Innermost element whose line range holds the given line."""
    best: Optional[ModelElement] = None
    for element in elements:
        if element.line <= line <= element.end_line:
            if best is None or element.line >= best.line:
                best = element
    return best.id if best else None


def _edges(walker: _IndexWalker) -> Tuple[List[ModelEdge], bool]:
    by_site = {(element.name, element.line): element.id for element in walker.elements}
    by_name: Dict[str, str] = {}
    for element in walker.elements:
        by_name.setdefault(element.name, element.id)

    edges: List[ModelEdge] = []
    seen = set()
    for reference in walker.references:
        source = getattr(reference, "element", None)
        if source is None:  # a reference declared outside every indexed element
            continue
        symbol = _resolve(walker, reference)
        if symbol is None:
            target = reference.name
        else:
            target = (
                by_site.get((symbol.name, symbol.line))
                or by_name.get(symbol.name)
                or symbol.name
            )
        key = (source, target, reference.kind, reference.name, reference.line)
        if key in seen:  # the inherited handlers report some references twice
            continue
        seen.add(key)
        edges.append(
            ModelEdge(
                source=source,
                target=target,
                kind=reference.kind,
                label=reference.name,
                line=reference.line,
                resolved=symbol is not None,
            )
        )
        if len(edges) >= MAX_EDGES:
            return edges, True
    return edges, False


def _issues(walker: _IndexWalker, errors) -> List[ModelIssue]:
    issues: List[ModelIssue] = []
    for error in errors[:MAX_ISSUES]:
        severity = error.severity
        if severity not in ("error", "warning", "info"):
            severity = "error"
        issues.append(
            ModelIssue(
                severity=severity,
                message=error.message,
                line=error.line,
                column=error.column,
                rule=error.error_type or "semantic",
                element=_element_for_line(walker.elements, error.line) if error.line else None,
            )
        )
    return issues


def _degraded(revision: str, validator: str, issues: List[ModelIssue]) -> ModelIndex:
    """An index that explains why there is nothing to show."""
    kept = issues[:MAX_ISSUES]
    return ModelIndex(
        revision=revision,
        validator=validator,
        issues=kept,
        stats=ModelStats(issues=len(kept)),
    )


def _index_from_code(text: str, revision: str) -> ModelIndex:
    from antlr4 import CommonTokenStream, InputStream, ParseTreeWalker

    from tools.sysml_parser.LoggingParsers.ErrorLogger import ErrorLogger
    from tools.sysml_parser.NewGrammar.SysMLv2Lexer import SysMLv2Lexer
    from tools.sysml_parser.NewGrammar.SysMLv2Parser import SysMLv2Parser as AntlrParser

    # The grammar's own error log is the syntax verdict: the ANTLR backend in
    # webapp/validation reads an attribute this ErrorLogger does not expose, so
    # it accepts a model the grammar recovered from. Attaching the logger also
    # keeps ANTLR from writing recoveries to stderr.
    syntax_errors = ErrorLogger(input_string=text)
    lexer = SysMLv2Lexer(InputStream(text))
    lexer.removeErrorListeners()
    lexer.addErrorListener(syntax_errors)
    parser = AntlrParser(CommonTokenStream(lexer))
    parser.removeErrorListeners()
    parser.addErrorListener(syntax_errors)
    tree = parser.rootNamespace()

    walker = _IndexWalker()
    ParseTreeWalker().walk(walker, tree)

    syntax = _syntax_issues(syntax_errors.get_errors_list())
    issues = (syntax + _issues(walker, walker.validate()))[:MAX_ISSUES]
    edges, edges_cut = _edges(walker)
    stats = ModelStats(
        elements=len(walker.elements),
        edges=len(edges),
        issues=len(issues),
        unresolved=sum(1 for edge in edges if not edge.resolved),
        requirements=sum(1 for e in walker.elements if e.kind in REQUIREMENT_KINDS),
    )
    return ModelIndex(
        revision=revision,
        validator="syntax-error" if syntax else "antlr",
        elements=walker.elements,
        edges=edges,
        issues=issues,
        stats=stats,
        truncated=walker.truncated or edges_cut,
    )


@lru_cache(maxsize=CACHE_ENTRIES)
def _cached(text: str) -> ModelIndex:
    revision = _revision(text)
    try:
        return _index_from_code(text, revision)
    except Exception as exc:  # grammar missing, or it crashed on hostile input
        logger.warning("model index failed (%s); returning a degraded index", exc)
        return _degraded(
            revision,
            "unavailable",
            [ModelIssue(message=f"No index was built: {exc}", rule="backend")],
        )


def build_index(code: str) -> ModelIndex:
    """Index one model.

    Syntax is checked first, so a broken model reports its errors instead of
    looking like an empty model. Cached by source text; the returned index is
    shared, so callers must treat it as read-only.
    """
    from webapp.validation import validate_sysml

    text = (code or "").strip()
    revision = _revision(text)
    if not text:
        return _degraded(
            revision,
            "empty",
            [ModelIssue(severity="error", message="The model is empty.", line=1, rule="syntax")],
        )

    report = validate_sysml(text)
    if not report.ok:
        return _degraded(
            revision,
            "syntax-error",
            [
                ModelIssue(
                    severity=issue.severity,
                    message=issue.message,
                    line=issue.line,
                    column=issue.column,
                    rule=issue.rule or "syntax",
                )
                for issue in report.issues
                if issue.severity != "info"
            ],
        )
    return _cached(text)
