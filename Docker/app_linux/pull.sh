#!/usr/bin/env bash

REPOSITORY="uwwee/svm-diagnosis"
TAG="app_linux"

IMG="${REPOSITORY}:${TAG}"

docker pull "${IMG}"
