# -*- coding: utf-8 -*-
"""
Custom exceptions for Oríon
===========================

"""


NO_CONFIGURATION_FOUND = """\
No commandline configuration found for new experiment."""


NO_EXP_NAME_PROVIDED = """\
No name provided for the experiment."""


class NoConfigurationError(Exception):
    """Raise when commandline configuration is empty."""

    def __init__(self, message=NO_CONFIGURATION_FOUND):
        super().__init__(message)


class NoNameError(Exception):
    """Raise when no name is provided for an experiment."""

    def __init__(self, message=NO_EXP_NAME_PROVIDED):
        super().__init__(message)


class CheckError(Exception):
    """Raise when a check has failed."""

    pass


class RaceCondition(Exception):
    """Raise when a race condition occured."""

    pass


MISSING_RESULT_FILE = """
Cannot parse result file.

Make sure to report results in file `$ORION_RESULTS_PATH`.
This can be done with `orion.client.cli.report_objective()`.
"""


class MissingResultFile(Exception):
    """Raise when no result file (or empty) at end of trial execution."""

    def __init__(self, message=MISSING_RESULT_FILE):
        super().__init__(message)


BRANCHING_ERROR_MESSAGE = """\
Configuration is different and generates a branching event:
{}

Hint
----

This error is typically caused by the following 2 reasons:
  1) Commandline calls where arguments are different from one worker to another
     (think of paths that are worker specific). There will be --cli-change-type
     in the error message above if it is the case.
  2) User script that writes to the repository of the script, causing changes in the code
     and therefore leading to branching events. There will be --code-change-type
     in the error message above if it is the case.

For each case you should:
  1) Use --non-monitored-arguments [ARGUMENT_NAME]
     (where you argument would be --argument-name, note the lack of dashes at
      the beginning and the underscores instead of dashes between words)
     The commandline argument only support one entry. To ignore many arguments,
     you can use the option in a local config file, or in the global config file:
     ```
     evc:
         non_monitored_arguments: ['FIRST_ARG', 'ANOTHER_ARG']
     ```

  2) Avoid writing data in your repository. It should only be code anyway, right? :)
     Otherwise, you can ignore code changes altogether with option --ignore-code-changes.

"""


class BranchingEvent(Exception):
    """Raise when conflicts could not be automatically resolved."""

    def __init__(self, status, message=BRANCHING_ERROR_MESSAGE):
        super().__init__(message.format(status))


class SampleTimeout(Exception):
    """Raised when the algorithm is not able to sample new unique points in time"""

    pass


class WaitingForTrials(Exception):
    """Raised when the algorithm needs to wait for some trials to complete before it can suggest new
    ones
    """

    pass


class CompletedExperiment(Exception):
    """Raised when attempting to suggest new trials for an experiment that is completed"""

    pass


class BrokenExperiment(Exception):
    """Raised when too many trials failed in an experiment and it is now considered broken"""

    pass


INEXECUTABLE_USER_SCRIPT_ERROR_MESSAGE = """\
User script is not executable:
{}

Use chmod +x <your script> to make it executable.
If your script is written in python, you can otherwise
use `python <your script>` instead of `./<your script>`.

"""


class InexecutableUserScript(PermissionError):
    """Raised during consumption when user script is not executable"""

    def __init__(self, cmdline, message=INEXECUTABLE_USER_SCRIPT_ERROR_MESSAGE):
        super().__init__(message.format(cmdline))


class UnsupportedOperation(Exception):
    """The operation is not supported with current access rights"""


class InvalidResult(ValueError):
    """The format of trial result is invalid."""
