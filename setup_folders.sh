#!/bin/bash

# Definiamo le directory di base
BASE_DIR="$(pwd)"
DATA_DIR="$BASE_DIR/data"
PROCESSED_DIR="$BASE_DIR/processed_data"

# Creazione della struttura per i dataset
mkdir -p "$DATA_DIR/Train"
mkdir -p "$DATA_DIR/Val"
mkdir -p "$DATA_DIR/Test-Dev"
mkdir -p "$DATA_DIR/Test-Challenge"
mkdir -p "$DATA_DIR/dataset"

# Creazione della struttura per i dati processati
mkdir -p "$PROCESSED_DIR/Train/real"
mkdir -p "$PROCESSED_DIR/Train/fake"
mkdir -p "$PROCESSED_DIR/Val/real"
mkdir -p "$PROCESSED_DIR/Val/fake"
mkdir -p "$PROCESSED_DIR/Test-Dev/real"
mkdir -p "$PROCESSED_DIR/Test-Dev/fake"
mkdir -p "$PROCESSED_DIR/Test-Challenge/real"
mkdir -p "$PROCESSED_DIR/Test-Challenge/fake"

echo "✅ Struttura delle cartelle creata con successo!"