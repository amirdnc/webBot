import asyncio
import os
from dataclasses import dataclass
from typing import List

import openai
from pyppeteer import launch

from dotenv import load_dotenv
load_dotenv()
@dataclass
class Task:
    """A high-level description of an action to complete."""

    description: str


class WebAgent:
    """Goal-oriented web assistant using Pyppeteer and OpenAI."""

    def __init__(self, api_key: str, endpoint: str, headless: bool = False) -> None:
        self.api_key = api_key
        openai.api_key = api_key
        self.headless = headless
        self.browser = None
        self.page = None
        self.endpoint = endpoint
        self.tasks: List[Task] = []
        self.goal: str | None = None

    async def start_browser(self) -> None:
        """Launch a new browser session."""
        self.browser = await launch(headless=self.headless)
        self.page = await self.browser.newPage()

    async def close_browser(self) -> None:
        """Close the browser session."""
        if self.browser:    
            await self.browser.close()

    def plan_tasks(self, goal: str) -> None:
        """Break a goal into an initial list of tasks."""
        self.goal = goal
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
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
            for line in response["choices"][0]["message"]["content"].splitlines()
            if line.strip()
        ]
        for line in lines:
            self.tasks.append(Task(line))

    async def execute_task(self, task: Task) -> None:
        """Ask the model how to satisfy a task and perform the action."""
        html = await self.page.content()
        prompt = (
            "You control a browser. Given the HTML and a task, "
            "respond with an action in one of these forms:\n"
            "NAVIGATE <url>\nCLICK <css selector>\nTYPE <css selector> <text>\nDONE\n"
            f"HTML:\n{html[:2000]}\nTask: {task.description}"
        )
        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        command = response["choices"][0]["message"]["content"].strip()
        if command.upper().startswith("NAVIGATE"):
            url = command.split(maxsplit=1)[1]
            await self.page.goto(url)
        elif command.upper().startswith("CLICK"):
            selector = command.split(maxsplit=1)[1]
            await self.page.click(selector)
        elif command.upper().startswith("TYPE"):
            _, selector, text = command.split(" ", 2)
            await self.page.type(selector, text)
        elif command.upper().startswith("DONE"):
            return
        else:
            raise ValueError(f"Unknown command: {command}")

        # After performing the action, ask for the next step.
        html = await self.page.content()
        followup = (
            f"Goal: {self.goal}\nHTML:\n{html[:2000]}\n"
            "What is the next task? Reply DONE if the goal is complete."
        )
        next_step = openai.ChatCompletion.create(
                    model="gpt-5",
                    messages=[{"role": "user", "content": followup}],
                )["choices"][0]["message"]["content"].strip()
        if next_step.upper() != "DONE":
            self.tasks.append(Task(next_step))

    async def run(self, goal: str) -> None:
        """Start the agent with a goal and execute tasks sequentially."""
        await self.start_browser()
        try:
            self.plan_tasks(goal)
            while self.tasks:
                task = self.tasks.pop(0)
                await self.execute_task(task)
        finally:
            await self.close_browser()


def parse_args() -> str:
    import argparse
    return "what is the weather in tokyo?"
    parser = argparse.ArgumentParser(description="Goal-oriented web agent")
    parser.add_argument("goal", help="Goal the agent should accomplish")
    parsed = parser.parse_args()
    return parsed.goal


async def main() -> None:
    goal = parse_args()
    api_key = os.getenv("OPENAI_API_KEY")
    endpoiont = os.getenv("ENDPOINT")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY environment variable not set")
    agent = WebAgent(api_key, endpoiont)
    await agent.run(goal)


if __name__ == "__main__":
    asyncio.run(main())