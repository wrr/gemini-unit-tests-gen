#!/bin/sh

docker build -t gemini-unit-test-gen .

docker run -v ~/.gitconfig:/etc/gitconfig  -v `pwd`:/chat --rm -it gemini-unit-test-gen bash
