import asyncio
import os
from dataclasses import dataclass
from datetime import datetime
from typing import List, TextIO, TypedDict

from openai import OpenAI
from pyppeteer import launch
from pyppeteer.errors import PageError, TimeoutError as PyppeteerTimeoutError

from dotenv import load_dotenv

try:
    from langgraph.graph import END, START, StateGraph
except ImportError as exc:  # pragma: no cover - informative import error
    raise ImportError(
        "LangGraph is required for lg_agent. Install it with `pip install langgraph`."
    ) from exc

try:
    from cli import parse_args
except ModuleNotFoundError:  # Running as a module package
    from agent.cli import parse_args  # type: ignore

load_dotenv()


@dataclass
class Task:
    """A high-level description of an action to complete."""

    description: str


class AgentState(TypedDict, total=False):
    goal: str
    tasks: List[str]
    history: List[str]
    requires_replan: bool
    last_command: str


class LangGraphWebAgent:
    """LangGraph-driven web agent that executes an agentic workflow in the browser."""

    def __init__(
        self,
        api_key: str,
        endpoint: str,
        headless: bool = False,
        *,
        verbose: bool = False,
        log_path: str | None = None,
    ) -> None:
        self.api_key = api_key
        self.endpoint = f"{endpoint}"
        self.client = OpenAI(base_url=endpoint, api_key=api_key)
        self.headless = headless
        self.browser = None
        self.page = None
        self.verbose = verbose
        self.log_path = log_path or "langgraph_agent.log"
        self.log_file: TextIO | None = None
        if self.verbose:
            resolved_path = os.path.abspath(self.log_path)
            directory = os.path.dirname(resolved_path)
            if directory and not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
            self.log_file = open(resolved_path, "w", encoding="utf-8")
            self.log(f"Verbose logging enabled. Writing to {resolved_path}")
        self.workflow = self._build_workflow()

    def log(self, message: str) -> None:
        """Emit a timestamped log entry when verbose mode is enabled."""
        if not self.verbose:
            return
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        formatted = f"[{timestamp}] {message}"
        print(formatted)
        if self.log_file:
            self.log_file.write(formatted + "\n")
            self.log_file.flush()

    async def start_browser(self) -> None:
        """Launch a new browser session."""
        self.log("Launching browser session.")
        CHROME_PATH = r"C:\\Program Files\\Google\\Chrome\\Application\\chrome.exe"
        self.browser = await launch(
            headless=self.headless,
            executablePath=CHROME_PATH,
            args=[
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-dev-shm-usage",
            ],
        )
        self.page = await self.browser.newPage()
        self.log("Browser session started.")

    async def close_browser(self) -> None:
        """Close the browser session."""
        if self.browser:
            await self.browser.close()
            self.log("Browser session closed.")
            self.browser = None
            self.page = None

    def close_log(self) -> None:
        """Close the verbose log file if it was opened."""
        if self.log_file:
            self.log("Closing verbose log.")
            self.log_file.close()
            self.log_file = None

    def _build_workflow(self):
        graph = StateGraph(AgentState)
        graph.add_node("plan", self._plan_node)
        graph.add_node("act", self._act_node)
        graph.add_edge(START, "plan")
        graph.add_conditional_edges("plan", self._after_plan, {"continue": "act", "stop": END})
        graph.add_conditional_edges(
            "act",
            self._after_act,
            {
                "continue": "act",
                "replan": "plan",
                "stop": END,
            },
        )
        return graph.compile()

    def _after_plan(self, state: AgentState) -> str:
        if state.get("tasks"):
            return "continue"
        return "stop"

    def _after_act(self, state: AgentState) -> str:
        if state.get("requires_replan"):
            return "replan"
        if state.get("tasks"):
            return "continue"
        return "stop"

    async def _plan_node(self, state: AgentState) -> AgentState:
        goal = state["goal"]
        self.log(f"Planning tasks for goal: {goal}")
        response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Break the user's goal into an ordered list of web tasks.",
                },
                {"role": "user", "content": goal},
            ],
        )
        lines = [
            line.strip("- ")
            for line in response.choices[0].message.content.splitlines()
            if line.strip()
        ]
        tasks = [Task(description=line).description for line in lines]
        history = list(state.get("history", []))
        if tasks:
            for task in tasks:
                self.log(f"Planned task: {task}")
            history.append(f"Planner proposed {len(tasks)} task(s).")
        else:
            self.log("Planner returned no tasks.")
            history.append("Planner returned no tasks.")
        return {
            "tasks": tasks,
            "history": history,
            "requires_replan": False,
        }

    async def _act_node(self, state: AgentState) -> AgentState:
        tasks = list(state.get("tasks", []))
        history = list(state.get("history", []))
        if not tasks:
            self.log("No queued tasks to execute; requesting replan.")
            history.append("No tasks to execute; requesting replan.")
            return {"tasks": tasks, "history": history, "requires_replan": True}
        task_description = tasks.pop(0)
        self.log(f"Executing task: {task_description}")
        history.append(f"Executing task: {task_description}")
        if not self.page:
            raise RuntimeError("Browser page is not initialized.")
        html = await self.page.content()
        prompt = (
            "You control a browser. Given the HTML and a task, "
            "respond with an action in one of these forms:\n"
            "NAVIGATE <url>\nCLICK <css selector>\nTYPE <css selector> <text>\nDONE\n"
            f"HTML:\n{html[:2000]}\nTask: {task_description}"
        )
        response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": prompt}],
        )
        command = response.choices[0].message.content.strip()
        self.log(f"Model command: {command}")
        history.append(f"Model command: {command}")

        async def ensure_selector(selector: str) -> bool:
            try:
                await self.page.waitForSelector(selector, {"timeout": 5000})
                return True
            except PyppeteerTimeoutError:
                self.log(f"Element not found for selector: {selector}")
                history.append(f"Element not found for selector: {selector}")
                return False

        if command.upper().startswith("NAVIGATE"):
            url = command.split(maxsplit=1)[1]
            self.log(f"Navigating to {url}")
            history.append(f"Navigating to {url}")
            await self.page.goto(url)
        elif command.upper().startswith("CLICK"):
            selector = command.split(maxsplit=1)[1]
            self.log(f"Clicking element: {selector}")
            history.append(f"Clicking element: {selector}")
            if not await ensure_selector(selector):
                tasks.insert(0, task_description)
                return {"tasks": tasks, "history": history, "requires_replan": True}
            try:
                await self.page.click(selector)
            except PageError as click_error:
                self.log(f"Click failed for {selector}: {click_error}")
                history.append(f"Click failed for {selector}: {click_error}")
                tasks.insert(0, task_description)
                return {"tasks": tasks, "history": history, "requires_replan": True}
        elif command.upper().startswith("TYPE"):
            _, selector, text = command.split(" ", 2)
            self.log(f"Typing into {selector}: {text}")
            history.append(f"Typing into {selector}: {text}")
            if not await ensure_selector(selector):
                tasks.insert(0, task_description)
                return {"tasks": tasks, "history": history, "requires_replan": True}
            await self.page.type(selector, text)
        elif command.upper().startswith("DONE"):
            self.log("Task marked as DONE by model.")
            history.append("Task marked as DONE by model.")
        else:
            self.log(f"Unknown command received: {command}")
            history.append(f"Unknown command received: {command}")
            raise ValueError(f"Unknown command: {command}")

        html = await self.page.content()
        followup_prompt = (
            f"Goal: {state['goal']}\nHTML:\n{html[:2000]}\n"
            "What is the next task? Reply DONE if the goal is complete."
        )
        followup = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{"role": "user", "content": followup_prompt}],
        ).choices[0].message.content.strip()
        normalized = followup.strip()
        if normalized.upper().startswith("DONE"):
            self.log("Received DONE after follow-up; goal considered complete.")
            history.append("Received DONE after follow-up; goal considered complete.")
        else:
            new_tasks = [
                line.strip("- ")
                for line in followup.splitlines()
                if line.strip()
            ]
            for new_task in new_tasks:
                self.log(f"Follow-up task queued: {new_task}")
                history.append(f"Follow-up task queued: {new_task}")
            tasks.extend(new_tasks)

        return {
            "tasks": tasks,
            "history": history,
            "requires_replan": False,
            "last_command": command,
        }

    async def run(self, goal: str) -> AgentState:
        self.log(f"Starting LangGraph workflow with goal: {goal}")
        await self.start_browser()
        try:
            result: AgentState = await self.workflow.ainvoke(
                {"goal": goal, "tasks": [], "history": []}
            )
            self.log("Workflow completed.")
            return result
        finally:
            await self.close_browser()
            self.close_log()


async def main() -> None:
    args = parse_args()
    goal = args.goal
    api_key = os.getenv("OPENAI_API_KEY")
    endpoint = os.getenv("ENDPOINT")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    verbose_env = os.getenv("AGENT_VERBOSE", "")
    verbose = args.verbose or verbose_env.lower() in {"1", "true", "yes", "on"}
    log_path = args.log_file or os.getenv("AGENT_LOG_PATH")
    agent = LangGraphWebAgent(api_key, endpoint, verbose=verbose, log_path=log_path)
    await agent.run(goal)


if __name__ == "__main__":
    asyncio.run(main())
