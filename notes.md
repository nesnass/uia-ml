# Environemnts

## Debugging
Run as:
`python3 main.py`
This will use Flask with the development web server called Werkzeug
OR:
Use VS Code launch config in .vscode dir

## Production (Docker)
Dockerfile will instruct Docker to start as:
`gunicorn", "-b", ":80", "main:app`
This will run in production mode using Gunicorn web sever

## Production (local)
First configure a virtual environment, else Google Cloud products can't be found
`pip install virtualenv`
`virtualenv <your-env>`
`source <your-env>/bin/activate`
`<your-env>/bin/pip install google-cloud-error-reporting`

Run as:
`gunicorn", "-b", ":80", "main:app`

## Pushing code
Pusing code to 'main' branch will auto-build an image from code, and add to a Docker container using Google Cloud Build
VM Container looks for the build tagged 'latest' e.g. `gcr.io/organic-nation-267514/uia-p3-ml:latest`
However the VM may need to be restarted to use the new image/container
Refer to Cloud Build Triggers.

