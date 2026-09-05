FROM python:3.12-slim

WORKDIR /app

# Layer dependency install separately from source so editing application code
# doesn't invalidate the (slow) dependency layer.
COPY pyproject.toml ./
COPY src ./src
RUN pip install --no-cache-dir -e .

COPY data ./data

ENTRYPOINT ["jobsearch"]
CMD ["--help"]
