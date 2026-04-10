"""
Unified Player class for the multi-agent system.

A Player is a self-contained agent that can:
1. Execute tasks using tools
2. Participate in debates (generate work, critique, revise)
3. Synthesize results from multiple sources

Each player has a role/persona defined by a prompt, and a set of tools
it can use to accomplish tasks.

Uses the unified ExecutionContext abstraction for all data access.
"""
from typing import List, Dict, Any, Optional, Union, Type, get_origin

from pydantic import BaseModel
from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..config import PLAYER_TEMPERATURE, create_llm, LLM_PROVIDER, structured_output_kwargs


def _primary_list_field_name(model_cls: Type[BaseModel]) -> Optional[str]:
    """Name of the first ``list``-typed field (wrapper record list), if any."""
    for name, finfo in model_cls.model_fields.items():
        if get_origin(finfo.annotation) is list:
            return name
    return None


class Player:
    """
    A unified player agent capable of executing tasks and participating in debates.
    
    Attributes:
        name: Unique identifier for this player instance
        role_prompt: The persona/role description that guides the player's behavior
        tools: List of tools available to this player
        llm: The language model instance for this player
    """
    
    def __init__(
        self,
        name: str,
        role_prompt: str,
        tools: Optional[List[BaseTool]] = None,
        model_name: str = None,
        temperature: float = None,
        provider: str = None
    ):
        """
        Initialize a Player with a role and tools.
        
        Args:
            name: Unique identifier for this player
            role_prompt: Description of the player's role/persona
            tools: List of LangChain tools available to this player
            model_name: The LLM model to use (default from config)
            temperature: LLM temperature (default from config)
            provider: LLM provider to use (default from config)
        """
        # Use config defaults if not specified
        temperature = temperature if temperature is not None else PLAYER_TEMPERATURE
        provider = provider or LLM_PROVIDER
        
        self.name = name
        self.role_prompt = role_prompt
        self.tools = tools or []
        self.llm = create_llm(
            model_name=model_name,
            temperature=temperature,
            provider=provider
        )
        self._output_parser = StrOutputParser()
    
    def get_tool_manifest(self) -> str:
        """
        Generates a string manifest of the tools available to this player.
        Used by the orchestrator for planning.
        """
        if not self.tools:
            return f"Player: {self.name}\n  Description: {self.role_prompt}\n  Tools: None"
        
        manifest = f"Player: {self.name}\n"
        manifest += f"  Description: {self.role_prompt}\n"
        tasks = [f"{tool.name}: {tool.description}" for tool in self.tools]
        manifest += f"  Tools:\n" + "\n".join([f"    - {task}" for task in tasks])
        return manifest
    
    def execute_task(
        self,
        task: str,
        context_key: str,
        context_info: Dict[str, Any],
        workspace: Dict[str, Any],
        inputs: Dict[str, str],
        target_resources: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Execute a specific task using available tools.
        
        This is the main execution method where the player uses its tools
        to accomplish a task from the plan.
        
        Args:
            task: The task description to execute
            context_key: Key for the ExecutionContext in the tool registry
            context_info: Serialized info about the ExecutionContext
            workspace: Dictionary of artifacts from previous steps
            inputs: Mapping of parameter names to artifact names in workspace
            target_tables: List of specific resources/tables this task targets
            
        Returns:
            Dictionary containing the execution result and any produced artifacts
        """
        # Resolve input artifacts from workspace
        resolved_inputs = {}
        for param_name, artifact_name in inputs.items():
            if artifact_name in workspace:
                resolved_inputs[param_name] = workspace[artifact_name]
            else:
                resolved_inputs[param_name] = f"[MISSING: {artifact_name}]"
        
        # Build the execution prompt
        tool_descriptions = "\n".join([
            f"- {tool.name}: {tool.description}" 
            for tool in self.tools
        ]) if self.tools else "No tools available."
        
        # Build context info section
        is_multi_resource = context_info.get("is_multi_resource", False)
        resources = context_info.get("resources", [])
        target_resources = target_resources or []
        
        # Decide once whether this player should behave as a pure labeller.
        is_labeller = "labeller" in self.name.lower() or "label" in task.lower()
        
        # Document content resolution strategy:
        # - For the labeller, ALWAYS read the full original document from the context.
        # - For direct_extractor, ALWAYS read the raw paper from context (same input family
        #   as standalone ``extract_meta_analysis``), matching the planner task text.
        # - For other players, if labeled_text exists, use it as primary document content.
        from ..tools.context_tools import get_context
        document_contents = {}

        is_direct_extractor = "direct_extractor" in self.name.lower()

        try:
            if (
                not is_labeller
                and not is_direct_extractor
                and "labeled_text" in workspace
            ):
                labeled_value = workspace["labeled_text"]
                # Allow either a direct string or a dict wrapper like {"labeled_text": "..."}
                if isinstance(labeled_value, dict) and "labeled_text" in labeled_value:
                    labeled_value = labeled_value["labeled_text"]
                document_contents = {"labeled_text": str(labeled_value)}
            else:
                # Fall back to reading from the ExecutionContext.
                actual_context = get_context(context_key)
                
                # Determine which resources to read
                resources_to_read = target_resources if target_resources else resources
                if not resources_to_read and actual_context.primary_resource:
                    resources_to_read = [actual_context.primary_resource]
                
                for resource in resources_to_read:
                    try:
                        content = actual_context.read_resource(resource)
                        if isinstance(content, str):
                            pass
                        elif isinstance(content, list):
                            content = "\n\n".join(content)
                        else:
                            content = str(content)
                        if is_direct_extractor and isinstance(content, str):
                            from ..experimentutils.eval_utils import (
                                highlight_numbers_and_tables,
                            )
                            content = highlight_numbers_and_tables(content)
                        document_contents[resource] = content
                    except Exception as e:
                        document_contents[resource] = f"[Error reading resource: {str(e)}]"
        except Exception as e:
            document_contents = {"error": f"Could not access context: {str(e)}"}
        
        if is_multi_resource:
            ctx_info = f"Multi-resource Context: {context_info.get('name', 'context')}\n"
            ctx_info += f"Context type: {context_info.get('context_type', 'unknown')}\n"
            ctx_info += f"Resources: {', '.join(resources)}\n"
            if target_resources:
                ctx_info += f"Target resources for this step: {', '.join(target_resources)}\n"
        else:
            resource_name = resources[0] if resources else "unknown"
            ctx_info = f"Context: {context_info.get('name', 'context')}\n"
            ctx_info += f"Context type: {context_info.get('context_type', 'unknown')}\n"
            ctx_info += f"Resource: {resource_name}\n"
        
        # Build document content section
        document_content_section = "\n\n--- Document Content ---\n"
        if len(document_contents) == 1:
            resource_name = list(document_contents.keys())[0]
            content = list(document_contents.values())[0]
            document_content_section += f"Content from '{resource_name}':\n\n{content}"
        else:
            for resource_name, content in document_contents.items():
                document_content_section += f"\n--- Content from '{resource_name}' ---\n{content}\n"
        
        # Adjust system message for labeller to avoid encouraging analysis/summarization
        system_task_instruction = (
            "Your task is to add XML tags to the original document text exactly as provided. "
            "Return the complete original text with tags inserted - do NOT analyze, summarize, or restructure."
            if is_labeller
            else "Your task is to analyze the context and provide a detailed response.\n"
                 "When you need to use a tool, describe what you would do and provide your analysis."
        )
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are {self.name}. {self.role_prompt}

You have access to the following tools:
{tool_descriptions}

{system_task_instruction}

{ctx_info}

For multi-resource contexts (e.g. multiple tables), consider:
- How resources might relate to each other
- Common fields that could be foreign keys
- Data integrity across resources
"""),
            ("human", """Task: {task}

Target resources for this step: {target_tables}

{document_content}

Input context from previous steps:
{input_context}

Execute this task and provide your output.""")
        ])
        
        # Build input context strictly from the artifacts the planner chose to wire
        # into this step via `inputs`. This gives the planner full control over what
        # each player can see from previous steps or from initial artifacts.
        input_context_parts = []
        if resolved_inputs:
            for k, v in resolved_inputs.items():
                input_context_parts.append(f"- {k}: {v}")
        
        input_context = "\n".join(input_context_parts) if input_context_parts else "No inputs from previous steps."
        
        # No extra schema or objective text is injected automatically here; the planner
        # must explicitly include those artifacts in `inputs` if a step should see them.
        
        # Execute with LLM
        chain = prompt | self.llm | self._output_parser
        
        # Actually invoke tools if available
        tool_results = {}
        
        # Invoke tools with context_key (and additional parameters when needed)
        for tool in self.tools:
            tool_name = tool.name.lower()
            try:
                # Determine which resources to analyze
                if target_resources:
                    resources_to_analyze = target_resources
                else:
                    resources_to_analyze = resources
                
                # Special handling for XML tagging tool: it needs field_value_pairs
                if "xml_tag_from_field_values" in tool_name:
                    field_value_pairs = workspace.get("field_value_pairs", [])
                    if not field_value_pairs:
                        tool_results[tool.name] = "Error: 'field_value_pairs' artifact not found in workspace."
                        continue

                    # The value_identifier stores field_value_pairs as a JSON string
                    # (from its synthesis output).  Parse it into a Python list so the
                    # tool's Pydantic args_schema validation succeeds.
                    if isinstance(field_value_pairs, str):
                        import json as _json
                        try:
                            field_value_pairs = _json.loads(field_value_pairs)
                        except Exception:
                            # Try extracting a JSON array from within the string
                            import re as _re
                            _m = _re.search(r'\[.*\]', field_value_pairs, _re.DOTALL)
                            if _m:
                                try:
                                    field_value_pairs = _json.loads(_m.group(0))
                                except Exception:
                                    tool_results[tool.name] = "Error: could not parse field_value_pairs JSON."
                                    continue
                            else:
                                tool_results[tool.name] = "Error: field_value_pairs is not valid JSON."
                                continue

                    # Run on the relevant resources (usually the primary document)
                    for resource in resources_to_analyze or [resources[0] if resources else ""]:
                        if not resource:
                            continue
                        try:
                            result = tool.invoke(
                                {
                                    "context_key": context_key,
                                    "resource": resource,
                                    "field_value_pairs": field_value_pairs,
                                }
                            )
                            tool_results[f"{resource}:{tool.name}"] = result
                        except Exception as e:
                            tool_results[f"{resource}:{tool.name}"] = f"Error: {str(e)}"
                    continue

                # Special handling for record-level XML tagging: reads labeled_text and
                # candidate_records from the workspace rather than from context, because
                # these artifacts are produced by previous plan steps, not stored in the
                # original ExecutionContext.
                if "xml_tag_records" in tool_name:
                    labeled_text_artifact = workspace.get("labeled_text", "")
                    candidate_records_artifact = workspace.get("candidate_records", [])
                    if not labeled_text_artifact:
                        tool_results[tool.name] = "Error: 'labeled_text' artifact not found in workspace."
                        continue
                    if not candidate_records_artifact:
                        tool_results[tool.name] = "Error: 'candidate_records' artifact not found in workspace."
                        continue

                    # Ensure labeled_text is a plain str (not a LangChain wrapper).
                    labeled_text_artifact = str(labeled_text_artifact)

                    # candidate_records may arrive as a JSON string from the
                    # record_grouper's string synthesis — parse it to a Python list.
                    if isinstance(candidate_records_artifact, str):
                        import json as _json
                        import re as _re
                        try:
                            candidate_records_artifact = _json.loads(candidate_records_artifact)
                        except Exception:
                            _m = _re.search(r'\[.*\]', candidate_records_artifact, _re.DOTALL)
                            if _m:
                                try:
                                    candidate_records_artifact = _json.loads(_m.group(0))
                                except Exception:
                                    tool_results[tool.name] = "Error: could not parse candidate_records JSON."
                                    continue
                            else:
                                tool_results[tool.name] = "Error: candidate_records is not valid JSON."
                                continue

                    # candidate_records may also be a Pydantic model — dump to a list of dicts.
                    from pydantic import BaseModel as _BaseModel
                    if isinstance(candidate_records_artifact, _BaseModel):
                        dumped = candidate_records_artifact.model_dump()
                        # Unwrap the first list-valued field (e.g. yield_records)
                        for v in dumped.values():
                            if isinstance(v, list):
                                candidate_records_artifact = v
                                break
                        else:
                            candidate_records_artifact = [dumped]

                    try:
                        result = tool.invoke(
                            {
                                "labeled_text": labeled_text_artifact,
                                "candidate_records": candidate_records_artifact,
                            }
                        )
                        tool_results[tool.name] = result
                    except Exception as e:
                        tool_results[tool.name] = f"Error: {str(e)}"
                    continue

                # Generic resource-specific tools
                if any(
                    kw in tool_name
                    for kw in [
                        "resource_info",
                        "item_count",
                        "field",
                        "sample",
                        "statistics",
                        "missing",
                        "unique",
                    ]
                ):
                    # Resource-specific tools - run on each target resource
                    for resource in resources_to_analyze:
                        try:
                            result = tool.invoke({
                                "context_key": context_key,
                                "resource": resource
                            })
                            tool_results[f"{resource}:{tool.name}"] = result
                        except Exception as e:
                            # Try without table parameter
                            try:
                                result = tool.invoke({"context_key": context_key})
                                tool_results[tool.name] = result
                                break
                            except Exception:
                                tool_results[f"{resource}:{tool.name}"] = f"Error: {str(e)}"
                else:
                    # Context-level tools
                    result = tool.invoke({"context_key": context_key})
                    tool_results[tool.name] = result
                    
            except Exception as e:
                tool_results[tool.name] = f"Error: {str(e)}"
        
        # Get LLM analysis
        target_info = ", ".join(target_resources) if target_resources else (
            "All resources" if is_multi_resource else "N/A"
        )
        
        # For labeller, if XML tagging tool produced a tagged document, prefer that
        # deterministic output over an LLM-generated one.
        is_labeller = "labeller" in self.name.lower() or "label" in task.lower()
        tagged_result = None
        if is_labeller:
            # Prefer deterministic tool output over LLM generation.
            # Check field-value tagger first, then record-level tagger.
            for key, value in tool_results.items():
                if isinstance(value, str) and not value.startswith("Error:"):
                    if "xml_tag_from_field_values" in key or "xml_tag_records" in key:
                        tagged_result = value
                        break

        if tagged_result is not None:
            llm_response = tagged_result
        else:
            llm_response = chain.invoke({
                "task": task,
                "target_tables": target_info,
                "document_content": document_content_section,
                "input_context": input_context + "\n\nTool Results:\n" + str(tool_results)
            })
        
        return {
            "player": self.name,
            "task": task,
            "tool_results": tool_results,
            "analysis": llm_response,
            "success": True,
            "is_multi_table": is_multi_resource,
        }
    
    def generate_initial_work(
        self,
        task: str,
        context_info: Dict[str, Any],
        context: Dict[str, Any]
    ) -> str:
        """
        Generate initial work/analysis for a debate round.
        
        Args:
            task: The task to work on
            context_info: Info about the ExecutionContext
            context: Additional context (workspace, tool results, etc.)
            
        Returns:
            The player's initial analysis as a string
        """
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    f"""You are {self.name}. {self.role_prompt}

You are participating in a multi-agent analysis of a context (dataset, API, etc.).
Your goal is to provide your unique perspective and insights.""",
                ),
                (
                    "human",
                    """Task: {task}

Context: {context_name} ({context_type})
Resources: {resources}

Context and available information:
{context}

Provide your initial analysis. Be thorough and specific.
Focus on what you can contribute based on your role.""",
                ),
            ]
        )
        
        chain = prompt | self.llm | self._output_parser
        
        return chain.invoke({
            "task": task,
            "context_name": context_info.get("name", "context"),
            "context_type": context_info.get("context_type", "unknown"),
            "resources": ", ".join(context_info.get("resources", [])),
            "context": str(context),
        })
    
    def critique_work(
        self,
        task: str,
        other_players_work: Dict[str, str]
    ) -> str:
        """
        Critique the work of other players.
        
        Args:
            task: The task being worked on
            other_players_work: Dictionary mapping player names to their work
            
        Returns:
            Critique as a string
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are {self.name}. {self.role_prompt}

You are reviewing the work of other analysts. Provide constructive criticism
that helps improve the overall analysis. Be specific about what could be
improved, what's missing, or what might be incorrect."""),
            ("human", """Task: {task}

Work from other players to critique:
{other_work}

Provide your critique. Focus on:
1. Accuracy and correctness
2. Completeness
3. Clarity and specificity
4. Suggestions for improvement""")
        ])
        
        chain = prompt | self.llm | self._output_parser
        
        other_work_str = "\n\n".join([
            f"=== {name} ===\n{work}" 
            for name, work in other_players_work.items()
        ])
        
        return chain.invoke({
            "task": task,
            "other_work": other_work_str
        })
    
    def revise_work(
        self,
        task: str,
        my_original_work: str,
        critiques: List[str]
    ) -> str:
        """
        Revise work based on critiques received.
        
        Args:
            task: The task being worked on
            my_original_work: This player's original work
            critiques: List of critiques from other players
            
        Returns:
            Revised work as a string
        """
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are {self.name}. {self.role_prompt}

You are revising your work based on feedback from other analysts.
Incorporate valid criticisms while maintaining your unique perspective."""),
            ("human", """Task: {task}

Your original work:
{original_work}

Critiques received:
{critiques}

Provide your revised analysis. Address the valid points raised in the critiques
while maintaining accuracy and your analytical perspective.""")
        ])
        
        chain = prompt | self.llm | self._output_parser
        
        critiques_str = "\n\n".join([
            f"Critique {i+1}:\n{c}" 
            for i, c in enumerate(critiques)
        ])
        
        return chain.invoke({
            "task": task,
            "original_work": my_original_work,
            "critiques": critiques_str
        })
    
    def synthesize_results(
        self,
        task: str,
        all_results: List[Dict[str, Any]],
        output_schema: Optional[Type[BaseModel]] = None,
        baseline_record_count: Optional[int] = None,
    ) -> Union[str, BaseModel]:
        """
        Synthesize multiple results into a consolidated output.
        Uses this player's role/expertise to consolidate debate results.
        
        Args:
            task: The task that was worked on
            all_results: List of results from all players
            output_schema: Optional Pydantic model class for structured output.
                          If provided, returns a validated Pydantic model instance.
                          If None, returns a string (legacy behavior).
            
        Returns:
            Synthesized result as a string or Pydantic model instance
        """
        # Deterministic tool-based players (labeller, record_labeller) must NOT go
        # through an LLM synthesis step — the LLM would corrupt or truncate the
        # carefully tagged XML document.  Pass the tool output through directly.
        is_deterministic = "labeller" in self.name.lower()
        if is_deterministic:
            for r in all_results:
                if r.get("player") == "schema_reference":
                    continue
                analysis = r.get("analysis", "")
                if analysis and not str(analysis).startswith("Error:"):
                    return analysis
            # If all analyses are errors/empty, fall through to LLM as last resort.

        results_str = "\n\n".join([
            f"=== {r.get('player', 'Unknown')} ===\n{r.get('analysis', str(r))}"
            for r in all_results
        ])
        
        if output_schema is not None:
            # Use structured output with Pydantic schema
            return self._synthesize_structured(
                task,
                results_str,
                output_schema,
                baseline_record_count=baseline_record_count,
            )
        else:
            # Legacy string output
            return self._synthesize_string(task, results_str)
    
    def _synthesize_string(self, task: str, results_str: str) -> str:
        """Synthesize results as a string (legacy behavior)."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are {self.name}. {self.role_prompt}

You are now synthesizing results from multiple analysts who worked on the same task.

**Your job:**
- Consolidate the findings into a single, authoritative result
- Resolve any conflicts by choosing the most accurate/complete information
- Preserve important details while removing redundancy
- Output a clear, concise result appropriate for the task

**Output requirements:**
- Output ONLY the consolidated result
- NO meta-commentary like "Based on the analyses..." or "The players found..."
- NO explanations of your synthesis process
- Keep the format appropriate for the task (e.g., numbers for counts, lists for columns)"""),
            ("human", """Task: {task}

Results from all analysts:
{all_results}

Provide the consolidated result for this task. Output only the result, no commentary.""")
        ])
        
        chain = prompt | self.llm | self._output_parser
        
        return chain.invoke({
            "task": task,
            "all_results": results_str
        })
    
    def _synthesize_structured(
        self,
        task: str,
        results_str: str,
        output_schema: Type[BaseModel],
        baseline_record_count: Optional[int] = None,
    ) -> BaseModel:
        """
        Synthesize results into a structured Pydantic model.
        
        Uses LangChain's with_structured_output() for guaranteed schema compliance.
        """
        rk = _primary_list_field_name(output_schema)
        shape_hint = ""
        if rk:
            shape_hint = (
                f'**Output shape:** Return one JSON object whose top-level key is exactly '
                f'"{rk}" (array of records). Do not use "records", "data", or a bare array.\n\n'
            )

        is_record_refine = self.name == "record_extractor" or (
            baseline_record_count is not None and baseline_record_count > 0
        )

        cardinality_block = ""
        if baseline_record_count is not None and baseline_record_count > 0 and rk:
            cardinality_block = (
                f"\n**HARD RECORD COUNT:** Baseline `direct_records` has **{baseline_record_count}** experiment rows. "
                f'The "{rk}" array MUST have length **at least {baseline_record_count}** (same baseline rows in order '
                f"as indices 0..{baseline_record_count - 1}, refined in place). "
                "Do NOT merge them into one record. You may append additional records only for clearly missing table rows.\n\n"
            )

        if is_record_refine and rk:
            system_body = f"""You are {self.name}. {self.role_prompt}

You are converting the analyst's per-record refinement work into the final structured schema.

**This is NOT summarization.** Do NOT consolidate multiple experiment rows into one object.

**Your job:**
- Emit **one schema record per baseline experiment row** (plus any clearly necessary append-only rows).
- Copy forward each baseline row, then apply field-level fixes/null-fills from the analyst text.
- Use null only where the paper truly lacks a value for that row.

**CRITICAL:**
- Output MUST conform exactly to the provided schema
- The "{rk}" list must preserve baseline cardinality (see HARD RECORD COUNT if present)
- Use actual values, not placeholders
- Never replace many rows with a single aggregate or "typical" row"""
        else:
            system_body = f"""You are {self.name}. {self.role_prompt}

You are synthesizing results from multiple analysts into a structured format.

**Your job:**
- Extract and consolidate all relevant information from the analyses
- Fill in ALL fields in the schema with concrete values from the gathered information
- Use null/None for fields where information is truly unavailable
- Resolve conflicts by choosing the most accurate/complete information

**CRITICAL:**
- Output MUST conform exactly to the provided schema
- Use actual values, not placeholders like "..." 
- Be specific and concrete
- Produce every non-empty record from the analyses; do not return an empty list if records were extracted."""

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_body),
            ("human", """Task: {task}

Results from all analysts:
{all_results}

""" + shape_hint + cardinality_block + """Generate the final structured output."""),
        ])
        
        structured_llm = self.llm.with_structured_output(
            output_schema,
            **structured_output_kwargs(),
        )
        chain = prompt | structured_llm
        
        return chain.invoke({
            "task": task,
            "all_results": results_str
        })
    
    def __repr__(self):
        return f"Player(name={self.name}, tools={len(self.tools)})"


def create_player_from_config(
    config: Dict[str, Any], 
    name: str,
    provider: str = None
) -> Player:
    """
    Factory function to create a Player from a configuration dictionary.
    
    Args:
        config: Dictionary with 'role_prompt', 'tools', and optional 'model_name', 'temperature'
        name: The name to assign to this player instance
        provider: LLM provider to use (default from config)
        
    Returns:
        Configured Player instance
    """
    return Player(
        name=name,
        role_prompt=config.get("role_prompt", "You are a helpful analyst."),
        tools=config.get("tools", []),
        model_name=config.get("model_name"),  # None means use config default
        temperature=config.get("temperature"),  # None means use config default
        provider=provider
    )
