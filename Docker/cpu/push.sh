#!/usr/bin/env bash

REPOSITORY="uwwee/svm-diagnosis"
TAG="cpu"

IMG="${REPOSITORY}:${TAG}"

docker image push "${IMG}"
