#!/usr/bin/env bash

REPOSITORY="uwwee/svm-diagnosis"
TAG="ipc-cpu"

IMG="${REPOSITORY}:${TAG}"

docker image push "${IMG}"
