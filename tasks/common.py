def render_mc(question:str, letters:str|list[str], choices:list[str]) -> str:
    rendered = [f"Multiple Choice question: {question}\n"]

    for letter, choice in zip(letters, choices):
        rendered.append(f"- {choice}={letter}\n")

    rendered.append("\nRespond only with the letter of the correct answer.")

    return "".join(rendered)


if __name__ == "__main__":
    question = "Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q."
    expected = "Multiple Choice question: Find the degree for the given field extension Q(sqrt(2), sqrt(3), sqrt(18)) over Q.\n- 0=A\n- 4=B\n- 2=C\n- 6=D\n\nRespond only with the letter of the correct answer."
    assert render_mc(question, "ABCD", ["0", "4", "2", "6"]) == expected

    print("all tests passed")
