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
from typing import List, Dict, Any, Optional

from langchain_core.tools import BaseTool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from ..config import PLAYER_TEMPERATURE, create_llm, LLM_PROVIDER


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
        target_tables: List[str] = None
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
        target_tables = target_tables or []
        
        if is_multi_resource:
            ctx_info = f"Multi-resource Context: {context_info.get('name', 'context')}\n"
            ctx_info += f"Context type: {context_info.get('context_type', 'unknown')}\n"
            ctx_info += f"Resources: {', '.join(resources)}\n"
            if target_tables:
                ctx_info += f"Target resources for this step: {', '.join(target_tables)}\n"
            ctx_info += f"\nTo use tools, pass context_key='{context_key}'"
        else:
            resource_name = resources[0] if resources else "unknown"
            ctx_info = f"Context: {context_info.get('name', 'context')}\n"
            ctx_info += f"Context type: {context_info.get('context_type', 'unknown')}\n"
            ctx_info += f"Resource: {resource_name}\n"
            ctx_info += f"\nTo use tools, pass context_key='{context_key}'"
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are {self.name}. {self.role_prompt}

You have access to the following tools:
{tool_descriptions}

Your task is to analyze the context and provide a detailed response.
When you need to use a tool, describe what you would do and provide your analysis.

{ctx_info}

For multi-resource contexts (e.g. multiple tables), consider:
- How resources might relate to each other
- Common fields that could be foreign keys
- Data integrity across resources
"""),
            ("human", """Task: {task}

Target tables for this step: {target_tables}

Input context from previous steps:
{input_context}

Execute this task and provide a comprehensive response. Include:
1. Your approach to the task
2. Any relevant observations or findings
3. The result of your analysis""")
        ])
        
        input_context = "\n".join([
            f"- {k}: {v}" for k, v in resolved_inputs.items()
        ]) if resolved_inputs else "No inputs from previous steps."
        
        # Execute with LLM
        chain = prompt | self.llm | self._output_parser
        
        # Actually invoke tools if available
        tool_results = {}
        
        # Invoke tools with context_key
        for tool in self.tools:
            tool_name = tool.name.lower()
            try:
                # Determine which tables to analyze
                if target_tables:
                    resources_to_analyze = target_tables
                else:
                    resources_to_analyze = resources
                
                # Check if tool needs table parameter
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
        target_info = ", ".join(target_tables) if target_tables else (
            "All resources" if is_multi_resource else "N/A"
        )
        llm_response = chain.invoke({
            "task": task,
            "target_tables": target_info,
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
        all_results: List[Dict[str, Any]]
    ) -> str:
        """
        Synthesize multiple results into a consolidated output.
        Uses this player's role/expertise to consolidate debate results.
        
        Args:
            task: The task that was worked on
            all_results: List of results from all players
            
        Returns:
            Synthesized result as a string
        """
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
        
        results_str = "\n\n".join([
            f"=== {r.get('player', 'Unknown')} ===\n{r.get('analysis', str(r))}"
            for r in all_results
        ])
        
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
