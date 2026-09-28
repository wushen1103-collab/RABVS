FROM mambaorg/micromamba:1.5.10

WORKDIR /workspace/RABVS

COPY environment.yml /tmp/environment.yml
RUN micromamba install -y -n base -f /tmp/environment.yml && micromamba clean -a -y

COPY . /workspace/RABVS

ENV PYTHONPATH=/workspace/RABVS/src
CMD ["python", "scripts/smoke_test.py"]

