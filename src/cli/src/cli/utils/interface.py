import inquirer


def interactive_select(options: dict[str, str] | list[str], prompt: str = "Select an option") -> str:
    """Interactive selection with arrow keys using inquirer.

    Args:
        options: Either a dict (display_key -> return_value) or list of strings
        prompt: The prompt message to show
    """
    if not options:
        return ""

    # Handle both dict and list inputs
    if isinstance(options, dict):
        if len(options) == 1:
            return list(options.values())[0]

        # Convert dict to display format for inquirer
        display_choices = list(options.keys())
        questions = [inquirer.List("choice", message=prompt, choices=display_choices, carousel=True)]
        answers = inquirer.prompt(questions)

        if answers and answers["choice"] in options:
            return options[answers["choice"]]
        return ""
    else:
        # Handle list input (original behavior)
        if len(options) == 1:
            return options[0]

        questions = [inquirer.List("choice", message=prompt, choices=options, carousel=True)]
        answers = inquirer.prompt(questions)
        return answers["choice"] if answers else ""
