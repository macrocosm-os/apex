import typer

from cli.commands.link import link
from cli.commands.submit import submit
from cli.commands.dashboard import dashboard
from cli.commands.version import get_version
from cli.commands.docs import docs
from cli.commands.competitions import competitions
from cli.commands.list_submissions import list_submissions
from cli.commands.result import result


app = typer.Typer(
    help="Apex CLI — interact with competitions and submissions.",
    add_completion=False,
)
app.command("link")(link)
app.command("submit")(submit)
app.command("dashboard")(dashboard)
app.command("version")(get_version)
app.command("docs")(docs)
app.command("competitions")(competitions)
app.command("list")(list_submissions)
app.command("result")(result)


if __name__ == "__main__":
    app()
