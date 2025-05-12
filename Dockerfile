# Base image with Python
FROM python:3.10-slim

# Set workdir
WORKDIR /app

# Copy only requirements.txt first to leverage Docker caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy server and client code
COPY server /app/server
COPY client /app/client

# Install Node.js (for building React client)
RUN apt-get update && apt-get install -y curl gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_18.x | bash - && \
    apt-get install -y nodejs && \
    cd client && npm install && npm run build && \
    cp -r build ../server/build && \
    apt-get remove -y curl gnupg nodejs && apt-get autoremove -y && apt-get clean

# Set workdir to server
WORKDIR /app/server

# Expose the port
EXPOSE 5001

# Start the server
CMD ["gunicorn", "app:app", "--bind", "0.0.0.0:5001"]
