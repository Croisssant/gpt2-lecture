def render_mc(question:str, letters:str|list[str], choices:list[str]) -> str:
    rendered = [f"Multiple Choice question: {question}\n"]

    for letter, choice in zip(letters, choices):
        rendered.append(f"- {choice}={letter}\n")

    rendered.append("\nRespond only with the letter of the correct answer.")

    return "".join(rendered)