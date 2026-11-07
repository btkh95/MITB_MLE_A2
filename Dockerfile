# Airflow image with Java for PySpark
FROM apache/airflow:2.10.5-python3.12

USER root

ENV DEBIAN_FRONTEND=noninteractive

# Install Java (OpenJDK 17 headless), procps (for 'ps') and bash
RUN apt-get update && \
    apt-get install -y --no-install-recommends openjdk-17-jdk-headless procps bash && \
    rm -rf /var/lib/apt/lists/* && \
    # Ensure Spark's scripts run with bash instead of dash
    ln -sf /bin/bash /bin/sh && \
    # On arm64 hosts (e.g. Apple Silicon) create the amd64 JAVA_HOME path Spark expects
    if [ ! -e /usr/lib/jvm/java-17-openjdk-amd64/bin/java ]; then \
        mkdir -p /usr/lib/jvm/java-17-openjdk-amd64/bin && \
        ln -s "$(readlink -f "$(which java)")" /usr/lib/jvm/java-17-openjdk-amd64/bin/java; \
    fi

ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-amd64
ENV PATH=$PATH:$JAVA_HOME/bin

USER airflow

# Pin Airflow so pip does not upgrade or downgrade it while resolving our packages
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir "apache-airflow==${AIRFLOW_VERSION}" -r /requirements.txt
