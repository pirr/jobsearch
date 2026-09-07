FROM python:3.12-slim

WORKDIR /app

# Install real dependencies from pyproject.toml before `src/` even exists in
# the image, so editing application code never invalidates this (slow,
# network-bound) layer -- only an actual dependency change does. `-e .`
# can't be run yet: an editable install of a src-layout package needs the
# real source tree present for setuptools to discover packages. tomllib is
# stdlib on this base image, so no extra tooling is needed to read the list.
COPY pyproject.toml ./
RUN python3 -c "import tomllib; print('\n'.join(tomllib.load(open('pyproject.toml', 'rb'))['project']['dependencies']))" > /tmp/requirements.txt \
    && pip install --no-cache-dir -r /tmp/requirements.txt

COPY src ./src
# --no-deps: every real dependency is already installed above, so this step
# just registers the local package -- no network calls, so re-running it
# after `COPY src` invalidates it (i.e. on every code change) is cheap.
RUN pip install --no-cache-dir --no-deps -e .

COPY data ./data

ENTRYPOINT ["jobsearch"]
CMD ["--help"]
