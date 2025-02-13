# Contributing to MeetNote

We're thrilled that you're interested in contributing to MeetNote! This document provides guidelines for contributing to both the master branch and the dev-combiner-testing branch. By participating in this project, you agree to abide by its terms.

## Table of Contents

- [Contributing to MeetNote](#contributing-to-meetnote)
  - [Table of Contents](#table-of-contents)
  - [Code of Conduct](#code-of-conduct)
  - [Getting Started](#getting-started)
  - [How to Contribute](#how-to-contribute)
  - [Coding Standards](#coding-standards)
  - [Commit Messages](#commit-messages)
  - [Pull Requests](#pull-requests)
  - [Testing](#testing)
  - [Documentation](#documentation)
  - [Community](#community)

## Code of Conduct

This project and everyone participating in it is governed by our Code of Conduct. By participating, you are expected to uphold this code. Please report unacceptable behavior to [project_email@example.com].

## Getting Started

1. Fork the repository on GitHub.
2. Clone your fork locally:
   ```
   git clone https://github.com/your-username/meetnote.git
   cd meetnote
   ```
3. Create a branch for your feature or bug fix:
   ```
   git checkout -b feature-or-fix-name
   ```
4. Make your changes and commit them with a clear commit message.
5. Push your changes to your fork on GitHub.

## How to Contribute

1. Ensure your code adheres to the project's coding standards.
2. Add or update tests as necessary.
3. Update documentation to reflect any changes.
4. Submit a pull request with a clear description of the changes.

For the dev-combiner-testing branch:
- Focus on improving existing combiners or adding new combining strategies.
- Ensure any new combiners are thoroughly tested and documented.
- Update the combiner testing utilities as necessary.

## Coding Standards

- Follow PEP 8 style guide for Python code.
- Use meaningful variable and function names.
- Keep functions small and focused on a single task.
- Comment your code where necessary, especially for complex logic.
- Use type hints for function arguments and return values.

## Commit Messages

- Use the present tense ("Add feature" not "Added feature").
- Use the imperative mood ("Move cursor to..." not "Moves cursor to...").
- Limit the first line to 72 characters or less.
- Reference issues and pull requests liberally after the first line.

## Pull Requests

1. Ensure the PR description clearly describes the problem and solution.
2. Include the relevant issue number if applicable.
3. Do not include unrelated changes.
4. Ensure all tests pass before submitting the PR.

## Testing

- Write unit tests for new functionality.
- Ensure all tests pass locally before submitting a pull request.
- For combiner development, use the `combiner_testing.py` utility to evaluate performance.

## Documentation

- Update the README.md if you change functionality.
- Add or update docstrings for new or modified functions and classes.
- If you add a new combiner, create or update the corresponding documentation in the `Docs/` directory.

## Community

- Feel free to ask questions on the project's [GitHub Discussions](https://github.com/helLf1nGer/meetnote/discussions) page.
- Be welcoming and inclusive in your interactions with other community members.

Thank you for contributing to MeetNote!
