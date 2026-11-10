"""A scripted stand-in for the model, so no test ever calls Ollama.

Testing agents: the graph only calls ``decider.invoke(messages)``, so a fake that returns a fixed
list of Decisions lets a test script a whole conversation ("run X, then repeat X, then answer") and
check what the graph does with it: fast, free and the same on every run.
"""

from __future__ import annotations

from shellmate.agent import Decision


def run(command: str, why: str = "test step") -> Decision:
    return Decision(action="run", command=command, explanation=why)


def answer(text: str) -> Decision:
    return Decision(action="answer", answer=text)


class ScriptedDecider:
    """Returns the given decisions in order; the last one repeats. An exception in the list is raised."""

    def __init__(self, *decisions):
        self.decisions = list(decisions)
        self.calls: list[list] = []

    def invoke(self, messages):
        self.calls.append(messages)
        item = self.decisions.pop(0) if len(self.decisions) > 1 else self.decisions[0]
        if isinstance(item, BaseException):
            raise item
        return item

    def last_message(self, call: int = -1) -> str:
        return self.calls[call][-1][1]


def tool(action: str, path: str = "", why: str = "test tool") -> Decision:
    return Decision(action=action, path=path, explanation=why)
