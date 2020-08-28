FROM python:3.7.6-buster
# Copy our application code
WORKDIR /var/app
COPY . .
COPY requirements.txt .
# Fetch app specific dependencies
RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
# Expose port
EXPOSE 80
# Start the app
#ENV FLASK_APP=main.py

ENTRYPOINT ["gunicorn"]
#CMD ["app.py"]
#CMD ["python3", "-m", "flask", "run", "--host=0.0.0.0", "--port=80"]
CMD [-b", ":80", "main:app"]
