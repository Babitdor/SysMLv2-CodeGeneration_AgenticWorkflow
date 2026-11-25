# A **SysML Chat Agent** should act as an intelligent Model-Based Systems Engineering (MBSE) assistant that can understand, create, modify, validate, analyze, and explain SysML models through natural-language conversations

The goal should be to move beyond a chatbot that answers SysML questions toward an **AI engineering copilot that works directly with a system model**.

For a modern implementation, I would recommend designing primarily for **SysML v2**, while providing SysML v1 support where needed. SysML v2 has standardized textual notation and an API and Services specification that make programmatic model interaction more practical.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.omg.org&sz=32)

Object Management Group+1

## **1. Core features of a SysML Chat Agent**

Here are the 15 capabilities I would consider.

| #   | Feature                             | What the agent should do                                             | Priority |
| --- | ----------------------------------- | -------------------------------------------------------------------- | -------- |
| 1   | Natural-language interaction        | Understand engineering questions and commands                        | P0       |
| 2   | Model understanding                 | Read elements, relationships, hierarchies and properties             | P0       |
| 3   | Model navigation                    | Find and locate elements in large models                             | P0       |
| 4   | Model creation                      | Generate valid SysML elements from natural language                  | P0       |
| 5   | Model modification                  | Add, update, delete and refactor model elements                      | P0       |
| 6   | Requirements engineering            | Create, refine and connect requirements                              | P0       |
| 7   | Traceability analysis               | Analyze relationships between requirements, architecture and tests   | P0       |
| 8   | Model validation                    | Detect semantic, syntactic and consistency issues                    | P0       |
| 9   | Diagram generation                  | Generate and explain graphical views of models                       | P1       |
| 10  | Impact analysis                     | Predict model dependencies affected by a change                      | P1       |
| 11  | Model explanation                   | Explain complex architecture in simple terms                         | P1       |
| 12  | Engineering reasoning               | Identify missing interfaces, functions and constraints               | P1       |
| 13  | Document integration                | Extract model information from PDFs, specifications and spreadsheets | P1       |
| 14  | Simulation and analysis integration | Connect models with engineering analysis tools                       | P2       |
| 15  | Collaboration and governance        | Support reviews, approvals, versioning and audit trails              | P1       |

P0 = essential MVP, P1 = next release, P2 = advanced capability. These are suggested product priorities, not SysML standard requirements.

## **2. Key engineering workflows**

A useful agent should support complete workflows, rather than isolated questions.

### **Example SysML agent workflow**

#### **Engineer request**

"Add a redundant power supply to the system and check which requirements and interfaces are affected."

1

Understand intent

Identify requested change

2

Query model

Locate relevant elements

3

Analyze dependencies

Find affected requirements and interfaces

4

Propose changes

Prepare valid SysML modifications

#### **Engineer reviews and approves**

Preview model differences, affected relationships and validation results before committing.

#### **Validated model update**

Commit changes, preserve history and provide a traceable summary.

## **3. Advanced AI features that differentiate the agent**

These are the features that could make a SysML agent substantially more valuable than traditional modeling tools.

### **Model-aware reasoning**

Answer questions such as "Why does this subsystem exist?", "Which requirements are not satisfied?" and "What happens if this component fails?" using actual model evidence.

### **Natural language to SysML**

Translate engineering descriptions into standards-compliant SysML v2 textual definitions, validate them and integrate them into the existing model.

### **Graph-based reasoning**

Traverse model relationships to answer dependency, completeness, traceability and impact-analysis questions.

### **Requirements quality assistant**

Identify ambiguous, duplicated, conflicting or potentially unverifiable requirements, and suggest improvements with engineer approval.

### **Automated model reviews**

Evaluate completeness, disconnected elements, unresolved references, missing verification links, organizational modeling rules and architecture consistency.

### **Explainable and grounded answers**

Every model-specific answer should reference actual model elements, stable IDs, relationships and versions. The agent should distinguish verified facts from AI suggestions.

## **4. Recommended technical architecture**

For SysML v2, I would structure the agent around a deterministic modeling backend, with an LLM handling interpretation and explanation.

### **Chat interface**

Natural-language requests, model references, change previews

### **AI orchestration layer**

Intent detection, tool selection, grounded responses, safety and approval control

Model query engine

Read and navigate

Graph reasoning

Trace relationships

SysML generator

Generate valid changes

Validation engine

Check and verify

### **SysML API and modeling services**

Model CRUD, versioning, semantic validation, transactions

### **Authoritative model repository**

SysML elements, relationships, requirements, versions and metadata

A few important architecture decisions:

- **Model repository as source of truth:** The LLM should not invent model states or maintain an independent authoritative copy.
- **Structured tools:** Use explicit query, create, update, validate and commit operations rather than relying on generated text alone.
- **Read-only by default:** Require preview and approval for modifications.
- **Transactional updates:** Prevent partially applied or inconsistent changes.
- **Grounded retrieval:** Combine semantic search for documents with structured model queries for relationships.
- **Version awareness:** Every answer and change should refer to an identifiable model version.

SysML v2's standardized API and Services specification provides a foundation for interoperability with modeling tools, although actual capabilities will depend on the tool implementation.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.omg.org&sz=32)

omg.org+1

## **5. Example conversations your agent should support**

| User prompt                                             | Agent capability         |
| ------------------------------------------------------- | ------------------------ |
| "Explain the system architecture."                      | Model understanding      |
| "Show all subsystems connected to the controller."      | Model navigation         |
| "Create a new battery management subsystem."            | Model generation         |
| "Which requirements have no verification relationship?" | Traceability             |
| "Find unused or disconnected components."               | Model quality checks     |
| "What will be affected if I remove this sensor?"        | Impact analysis          |
| "Generate SysML v2 code from these requirements."       | Text-to-model generation |
| "Compare architecture version A and B."                 | Model differencing       |
| "Review the architecture for missing interfaces."       | Engineering reasoning    |
| "Generate a model review report."                       | Reporting                |

## **6. Suggested MVP**

I would start with seven features that deliver useful engineering value while keeping implementation complexity manageable.

## **MVP feature checklist**

Select the features you want to include in your first release.

7 / 7

Chat and natural-language understanding

Understand engineering intent and follow-up context.

Model querying and navigation

Read actual repository elements, properties and relationships.

SysML v2 code generation

Create structured model change proposals.

Validation and error explanation

Check syntax, semantic correctness and references.

Requirements traceability

Identify linked and unlinked engineering elements.

Change preview and approval

Show differences before committing changes.

Source-grounded answers

Reference model element IDs and versions.

&#x20;Copy selected requirements

## **7. How to measure success**

| KPI                                      | Suggested initial target               |
| ---------------------------------------- | -------------------------------------- |
| Model query correctness                  | ≥95% on a benchmark                    |
| Valid SysML generated                    | ≥90% on supported tasks                |
| Traceability query correctness           | ≥95%                                   |
| Unauthorized model changes               | Zero                                   |
| Answers with verifiable model references | 100% for model-specific factual claims |
| Successful model update workflow         | ≥90% on tested scenarios               |

These are proposed engineering acceptance targets, not industry benchmarks. They should be evaluated against a representative set of real SysML models.

## **8. Important distinction: SysML v1 versus v2**

| Area                      | SysML v1                                                | SysML v2                                        |
| ------------------------- | ------------------------------------------------------- | ----------------------------------------------- |
| Standard textual notation | No equivalent comprehensive standardized textual syntax | Yes                                             |
| Model interaction         | Often vendor-specific APIs                              | Standardized API specification                  |
| Primary representation    | UML-derived graphical modeling                          | Formally defined textual and graphical modeling |
| Model generation          | Often tool-dependent                                    | Textual generation and semantic validation      |
| AI integration            | Requires more tool-specific adapters                    | More standardized integration opportunities     |

SysML v2 is therefore a strong foundation for a new AI-native modeling assistant.&#x20;

[image](https://www.google.com/s2/favicons?domain=https://www.omg.org&sz=32)

Object Management Group+1

**My recommendation:** The three most important differentiators are **model-aware reasoning, natural-language model modification, and intelligent validation/traceability**. Together, these transform a SysML chatbot from a documentation assistant into a productive MBSE engineering tool.

One question that would substantially influence the feature set: **Are you building the chat agent for SysML v1 (such as Cameo/MagicDraw), SysML v2, or both?** The integration architecture and achievable automation features differ significantly.
