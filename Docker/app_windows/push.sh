#!/usr/bin/env bash

REPOSITORY="uwwee/svm-diagnosis"
TAG="app_windows"

IMG="${REPOSITORY}:${TAG}"

docker image push "${IMG}"
