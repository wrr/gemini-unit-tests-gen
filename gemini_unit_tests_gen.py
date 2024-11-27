# Copyright 2024 - Jan Wrobel jan@mixedbit.org

import contextlib
import datetime
import os
import re
import subprocess
import sys

import google.generativeai as genai

from filestotest import FILES_TO_TEST

PROJECT_TO_TEST = 'algorithms'

DEBUG = False
GEMINI_API_KEY = os.getenv('GEMINI_API_KEY')


# version suffix is required for cache to work.
# Gemini-pro model runs out of quota with the Gemini free tire, paid
# account is needed.
MODEL_ID = 'gemini-1.5-pro-001'
USE_CONTEXT_CACHE = True


# 1.5-flash works with a Gemini free tier, but produces worse results
# (sometimes even fails to follow the communication protocol).
# MODEL_ID = 'gemini-1.5-flash'
# USE_CONTEXT_CACHE = False # Not supported by the free tier

SINGLE_FILE_GENERATION_ATTEMPTS_LIMIT = 5

PROMPT = """You are a programmer working on a Python project. Your job
is to add missing unit tests to the project. All the project files are
attached. Study carefully the existing source files and unit tests.
Your added tests need to match the style and naming conventions used
in the project. Be consistent.

You are interacting with a computer that will be writing to disk,
executing and committing to git repository tests created by you.

First, wait for the computer to issue the following command:

###
ADD_TEST_FOR: file_path
TEST_COVERAGE:
###

In this command, file_path is a path of the python file to test
relative to the project root directory. Examine carefully the source
code of this file to understand the functionality that needs to be
tested. Also examine the existing tests for the file (if any), add
tests only for logic that is not already tested by the existing
files. If a file is called foo_bar.py, the tests for such file are
usually called test_foo_bar.py or tests_for_bar.py and are often
placed in a separate 'tests' folder.

TEST_COVERAGE: is followed by a source code of the file with each line
prefixed with a coverage information like this:

  # some function
> def h(x):
-     if 0:  # pragma: no cover
-         pass
>     if x == 1:
!         a = 1
>     else:
>         a = 2

'>' prefix means that the line is covered by tests.
'!' prefix means that the line is not covered by test. Focus on these
lines, your tests should include logic to execute them.
Lack of prefix means that line is not executable and doesn't need to
be covered (for example a comment or an empty line).
 '-' prefix means that line is explicitly excluded from coverage testing
 and doesn't need to be covered by tests.

Special response "TEST_COVERAGE: WHOLE_FILE_NOT_COVERED" means
that not a single line of the file is currently covered by unit tests.

Next, you need to create a single unit test file for the requested
file to test. Write the following command (enclosed in ### below) for
the computer:

###
WRITE_TEST_FILE: file_path_with_name

file_content

END_TEST_FILE
your_comments
###

Where:
file_path_with_name is a path of the added test file relative to the
project root directory. Important: all added file names (but not
directories) must start with the test_gemini_ prefix (for example
tests/test_gemini_quick_sort.py is correct). Do not create new tests
directories, put tests in the already existing ones.

file_content is a Python code of the test file to be added. Do not use
any special encoding or character escaping for the Python code. Also
do not wrap the code in any quotes and do not start it with the
'python' prefix. All names of added TestCase classes need to have
GeminiTest suffix (for example QuickSortGeminiTest). All added test
methods need to have test_gemini_ prefix (for example
test_gemini_quick_sort_empty_list). Follow the project programming
style and PEP 8 code style recommendations.

your_comments can include any description of what you are doing, what
problems did you encounter with your tests and how do you try to solve
them.

The computer (not you) will respond with a message in the following
format:

TEST_RUN_STATUS:
FAILURE_MESSAGE:
TEST_COVERAGE:
PROMPT:

Where TEST_RUN_STATUS: is followed by either 'OK' if the added test
file executed successfully or 'FAILED' in other cases.
FAILURE_MESSAGE: is present only if the tests failed and is followed by
the errors printed by the Python interpreter. This can include stack
trace or syntax errors.
TEST_COVERAGE: is followed by an updated information about the
coverage of the tested file. It has the form already described above.
PROMPT: is optional and can be followed by additional instructions.

Please don't just add a single test function to cover all untested
functionality. Instead, add a separate test function for each tested
case.

Examine carefully the returned values, if there are any syntax errors
or test assertion failures, continue to modify the test file and write
it again until there are no more errors. Not all test assertion code
can be made to be passing. It is possible that your test will uncover
an actual error in the program logic. In such case the unit test added
by you will be failing. Annotate such tests with a comment and comment
out the failing test assertion like this:

# TODO: fix the program logic to make this test pass:
# self.assertEqual([1, 2, 3], quick_sort([1, 3, 2]))

FAILURE_MESSAGE result of the WRITE_TEST_FILE command must be
inspected to ensure it does not contain any syntax errors. Continue to
modify the test file to fix any syntax errors.

Once your test file is ready, runs without any errors, with any
assertions that are failing due to a bug in a program commented out,
and has good, preferably 100% coverage (no lines marked with !)
commit it using the following command (enclosed in ### below):

###
COMMIT: file_path_with_name
commit_message
END_COMMIT_MESSAGE
###

Where commit_message is a git commit message describing the
change. Separate subject from body with a blank line. Start the
subject with 'Gemini: ' prefix. Wrap lines at 72
characters. Capitalize the subject line and each paragraph. Use the
imperative mood in the subject line. Include information in the body
that the test was generated automatically.

Issue commands one by one, never issue more than one command in a
single message. COMMIT command should be issued only after you have
examined the computer response from the last WRITE_TEST_FILE command
and the computer responded with "TEST_RUN_STATUS: OK" and most of the
tested file lines are covered.

After committing the test file, wait for instructions to create a next
test file or to TERMINATE.

Do not write any text that does not respect the communication protocol
defined above. The computer only understands these two commands from
you: WRITE_TEST_FILE and COMMIT. The computer will not be able to parse
any other messages correctly and can fail if you provide other input.
"""


def run_command(command, capture_stderr=False):
    """Runs a command.

    Args:
        command: The command to run as a string.
        capture_stderr: if True stderr is captured.

    Returns:
        A tuple containing:
        - return_code: The return code of the process.
        - stderr: The standard error as a string (decoded) or None.
    """
    default_output = subprocess.DEVNULL
    if DEBUG:
        print(f'Executing {command}')
        default_output = None
    process = subprocess.run(
        command,
        shell=True,
        stdout=default_output,
        stderr=(subprocess.PIPE if capture_stderr else default_output),
        text=True,
        check=False
    )
    if capture_stderr:
        return process.returncode, process.stderr
    return process.returncode, None

def get_coverage(path):
    """Return a test coverage for a given file."""
    command = ('coverage run '
               '--omit=tests*,*init*,test* '
               '-m unittest discover tests')
    run_command(command)
    command = f'coverage annotate --include {path}'
    status_code, _ = run_command(command)
    if status_code != 0:
        # coverage command fails if the file is not even imported by
        # any executed file.
        return 'WHOLE_FILE_NOT_COVERED'
    with open(path + ',cover', 'r', encoding='utf8') as cover_file:
        return cover_file.read()

def write_string_to_file(file_path, file_string):
    with open(file_path, 'w', encoding='utf8') as f:
        f.write(file_string)

def add_test_file(tested_file_path:str, test_file_path:str, file_content:str):
    """Adds and executes a test file.

    If a file with such path already exists, overwrites the existing file.

    Args:
        tested_file_path: A string path to the tested source code file.
        test_file_path: A string path of the added test file relative to
           the project root directory.
        file_content: Content of the test file.

    Returns:
         A dict with three keys: 'success', 'stderr' and 'coverage':
         The value of the 'success' key is True, if the test executed
         successfully. The value of the 'stderr' key is a string: a
         standard error output captured while running the added test
         file. The value of the 'coverage' key is a string describing
         the test coverage of the tested file after the new test was
         added.
    """
    if DEBUG:
        print(f'add_test_file {test_file_path}')
    write_string_to_file(test_file_path, file_content)
    module_to_test = test_file_path.replace('/', '.').replace('.py', '')
    command = f'python3 -m unittest {module_to_test}'
    return_code, stderr_str = run_command(command, capture_stderr=True)
    is_success = return_code == 0
    return {
        'success': is_success,
        'stderr': stderr_str,
        'coverage': get_coverage(tested_file_path),
    }


def git_commit_test_file(test_file_path:str, commit_message:str):
    """Creates a git commit with a previously added test file.

    Args:
       test_file_path: A string path of the previously added test
          file to be committed.
       commit_message: A git commit message describing the change.
    """
    if DEBUG:
        print(f'git_commit_test_file {test_file_path}')
    return_code, _ = run_command(f'git add {test_file_path}')
    assert return_code == 0
    return_code, _ = run_command(f'git commit -m "{commit_message}"')
    assert return_code == 0

def list_python_files(root_folder):
    for dirpath, dirnames, filenames in os.walk(root_folder):
        # Exclude venv dirs
        dirnames[:] = [d for d in dirnames if d != 'venv']

        for filename in filenames:
            # do not include top level python scripts as these
            # are helpers, not parts of modules and usually are not
            # tested.
            if dirpath != '.' and filename.endswith('.py'):
                full_path = os.path.join(dirpath, filename)
                yield full_path.lstrip('./')

def list_uploaded_files():
    return genai.list_files()

def delete_uploaded_files():
    for f in list_uploaded_files():
        f.delete()

def upload_source_files():
    # delete_uploaded_files()
    for path in list_python_files('.'):
        content_type = 'text/x-python'
        yield genai.upload_file(path=path, display_name=path,
                                mime_type=content_type)

def parse_add_file_command(input_string):
    """Parses a string representing the WRITE_TEST_FILE command.

    Returns:
        A tuple containing the file path and file content, None if the
        input string is not a WRITE_TEST_FILE command.
    """
    pattern = r'.*WRITE_TEST_FILE:\s+([^\n\s]+)\n(.+?)\nEND_TEST_FILE.*'
    match = re.match(pattern, input_string, re.DOTALL)

    if not match:
        return None
    file_path = match.group(1)
    file_content = match.group(2)
    return file_path, file_content

def parse_commit_command(input_string):
    """Parses a string representing the COMMIT command.

    Returns:
        A tuple containing the file path and commit, None if the
        input string is not a COMMIT command.
    """
    pattern = r'.*COMMIT:\s+([^\n]+_gemini_[^\n\s]+)\n(.+?)\nEND_COMMIT_MESSAGE.*'
    match = re.match(pattern, input_string, re.DOTALL)
    if not match:
        return None
    file_path = match.group(1)
    commit_message = match.group(2)
    return file_path, commit_message

def chat_request_test_generation(chat_session, generation_config, file_to_test):
    horizontal_bar = '=' * 20
    user_message = (f'ADD_TEST_FOR: {file_to_test}\n'
                    f'TEST_COVERAGE: {get_coverage(file_to_test)}\n')
    for attempt in range(0, SINGLE_FILE_GENERATION_ATTEMPTS_LIMIT):
        print(f'{horizontal_bar}User message{horizontal_bar}\n{user_message}')
        response = chat_session.send_message(
            user_message,
            generation_config=generation_config)

        command = response.text
        print(f'{horizontal_bar}Gemini message{horizontal_bar}\n{command}\n'
              f'Metadata: {response.usage_metadata}')
        add_file_args = parse_add_file_command(command)
        commit_args = parse_commit_command(command)
        if add_file_args and commit_args:
            # This has happened a couple of times.
            user_message = ('ERROR: You cannot send WRITE_TEST_FILE and COMMIT '
                            'commands in a single message. '
                            'always first send the WRITE_TEST_FILE commands '
                            'alone and once the file is ready send the COMMIT '
                            'command alone.')
        elif add_file_args:
            test_file_path, file_content = add_file_args[0], add_file_args[1]
            if 'test_gemini_' not in test_file_path:
                user_message = ('ERROR: test file names must start with the '
                                'test_gemini_ prefix')
                continue
            result = add_test_file(file_to_test,
                                   test_file_path,
                                   file_content)
            user_message = \
                f'TEST_RUN_STATUS: {"OK" if result["success"] else "FAILED"}\n'
            if not result['success']:
                user_message += f'FAILURE_MESSAGE: {result["stderr"]}\n'
            user_message += f'TEST_COVERAGE:\n{result["coverage"]}\n'
            if attempt == SINGLE_FILE_GENERATION_ATTEMPTS_LIMIT - 3:
                user_message += (
                    'PROMPT: you have one more test creation attempt left. '
                    'After the next attempt you need to COMMIT the test.\n')
                if not result['success']:
                    user_message += (
                        'If the test assertions are still failing, please '
                        'comment out the failing assertions and add a '
                        'comment in the code for the humans to review them.\n')
            if attempt == SINGLE_FILE_GENERATION_ATTEMPTS_LIMIT - 2:
                user_message += (
                    'PROMPT: this was your last attempt to create this test'
                    'file, if the test works please COMMIT it now.\n')
        elif commit_args:
            git_commit_test_file(commit_args[0], commit_args[1])
            user_message = 'Thank you, the test is committed'
            return
        else:
            user_message = 'Unrecognized command, try again'


@contextlib.contextmanager
def create_chat_session(uploaded_files):
    if USE_CONTEXT_CACHE:
        cache = genai.caching.CachedContent.create(
            model=MODEL_ID,
            display_name='unit-test-generator', # used to identify the cache
            system_instruction=PROMPT,
            contents=uploaded_files,
            ttl=datetime.timedelta(minutes=180),
        )
        try:
            model = genai.GenerativeModel.from_cached_content(
                cached_content=cache)
            yield model.start_chat()
        finally:
            cache.delete()
    else:
        model = genai.GenerativeModel(MODEL_ID, system_instruction=PROMPT)
        yield  model.start_chat(history=[
            {
                'role': 'user',
                'parts': uploaded_files,
            }
        ])


def fatal(msg):
    print(msg, file=sys.stderr)
    sys.exit(1)

def ensure_no_cache():
    for c in genai.caching.CachedContent.list():
        fatal(f'Have active cache (should not happen) {c}')

def main():
    if not GEMINI_API_KEY:
        fatal('You need to set GEMINI_API_KEY environment variable')

    os.chdir(PROJECT_TO_TEST)
    genai.configure(api_key=GEMINI_API_KEY)
    # Just in case, to make sure there are no active caches left after
    # the previous run).
    ensure_no_cache()
    uploaded_files = list(upload_source_files())
    # Files do not need to be uploaded each run, once uploaded, they
    # can be just listed:
    #uploaded_files = list(list_uploaded_files())
    if DEBUG:
        print(uploaded_files)
    generation_config = genai.GenerationConfig(temperature=0.1)
    with create_chat_session(uploaded_files) as chat_session:
        for file_to_test in FILES_TO_TEST:
            chat_request_test_generation(chat_session,
                                         generation_config,
                                         file_to_test)
    ensure_no_cache()

if __name__ == '__main__':
    main()
