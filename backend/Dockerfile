# Use Python 3.12 slim base image
FROM python:3.12-slim

# Set working directory in the container
WORKDIR /app

# Copy requirements.txt from the parent directory
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --trusted-host pypi.org --trusted-host pypi.python.org --trusted-host files.pythonhosted.org -r requirements.txt

# Copy the backend source code files
COPY backend/server.py .
COPY backend/process.py .
COPY backend/display.py .
COPY backend/app.py .

# Copy the sample data from parent directory to expected location
COPY sample_data ../sample_data

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "server.py"]